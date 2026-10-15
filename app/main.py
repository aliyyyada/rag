"""HTTP API и веб-интерфейс."""
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from . import config, contradictions, rag
from .ingest.pdf import extract_pdf
from .ollama_client import OllamaError
from .service import Service
from .textproc import chunk_pages, clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("rag")
STATIC = Path(__file__).parent / "static"

svc: Service | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global svc
    svc = Service()
    svc.start()
    yield
    svc.stop()


app = FastAPI(title="RAG: поиск по документам предприятия", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=config.TOP_K, ge=1, le=15)
    detailed: bool = config.ANSWER_DETAILED


class ScanRequest(BaseModel):
    doc_ids: list[int] | None = None
    max_pairs: int = Field(default=config.CONTRA_MAX_PAIRS, ge=1, le=10000)


class StatusRequest(BaseModel):
    status: str = Field(pattern="^(new|confirmed|dismissed)$")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    try:
        missing = svc.missing_models()
    except Exception as e:
        return {"ok": False, "ollama": f"недоступна: {e}"}
    return {"ok": not missing, "ollama": "ok", "llm": config.LLM_MODEL,
            "embed": config.EMBED_MODEL, "missing_models": missing}


# --- документы ---
@app.get("/api/documents")
def list_documents():
    return svc.store.list_documents()


@app.post("/api/documents")
async def upload(files: list[UploadFile] = File(...)):
    out = []
    for f in files:
        name = Path(f.filename or "document.pdf").name
        if not name.lower().endswith(".pdf"):
            out.append({"title": name, "error": "Поддерживаются только PDF"})
            continue
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=config.FILES_DIR)
        with tmp:
            while chunk := await f.read(1 << 20):
                tmp.write(chunk)
        out.append(await run_in_threadpool(svc.add_pdf, Path(tmp.name), Path(name).stem))
    return out


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int):
    doc = svc.store.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Документ не найден")
    svc.store.delete_document(doc_id)
    if doc["source"] == "pdf":
        (config.FILES_DIR / doc["source_ref"]).unlink(missing_ok=True)
    return {"deleted": doc_id}


@app.post("/api/documents/{doc_id}/retry")
def retry_document(doc_id: int):
    if not svc.retry(doc_id):
        raise HTTPException(409, "Документ нельзя обработать повторно сейчас")
    return svc.store.get_document(doc_id)


@app.get("/api/documents/{doc_id}/file")
def document_file(doc_id: int):
    doc = svc.store.get_document(doc_id)
    if not doc or doc["source"] != "pdf":
        raise HTTPException(404, "Файл не найден")
    return FileResponse(config.FILES_DIR / doc["source_ref"], media_type="application/pdf",
                        filename=f"{doc['title']}.pdf")


@app.post("/api/sql/sync")
def sql_sync():
    if not config.SQL_SOURCE_URL:
        raise HTTPException(400, "SQL-источник не настроен (SQL_SOURCE_URL)")
    try:
        return svc.sync_sql()
    except Exception as e:
        raise HTTPException(502, f"Ошибка SQL-источника: {e}")


@app.post("/api/watch/scan")
def watch_scan():
    if not config.WATCH_DIR:
        raise HTTPException(400, "Папка WATCH_DIR не настроена")
    return {"added": svc.scan_watch_dir()}


# --- поиск и ответы ---
@app.post("/api/search")
def search(req: AskRequest):
    try:
        return rag.retrieve(svc.store, svc.index, svc.llm, req.question, req.top_k)
    except OllamaError as e:
        raise HTTPException(503, str(e))


@app.post("/api/ask")
def ask(req: AskRequest):
    """Ответ целиком: {answer, sources (только использованные), contradictions, contradiction_note}."""
    try:
        return rag.answer(svc.store, svc.index, svc.llm, req.question, req.top_k, req.detailed)
    except OllamaError as e:
        raise HTTPException(503, str(e))


# --- противоречия ---
@app.get("/api/contradictions")
def list_contradictions(status: str | None = None):
    return svc.store.list_contradictions(status=status)


@app.patch("/api/contradictions/{cid}")
def set_contradiction_status(cid: int, req: StatusRequest):
    svc.store.set_contradiction_status(cid, req.status)
    return {"id": cid, "status": req.status}


@app.post("/api/contradictions/scan")
def start_scan(req: ScanRequest):
    if not svc.start_scan(req.doc_ids, req.max_pairs):
        raise HTTPException(409, "Проверка уже идёт")
    return svc.scan_state.as_dict()


@app.get("/api/contradictions/status")
def scan_status():
    return svc.scan_state.as_dict()


@app.post("/api/contradictions/cancel")
def cancel_scan():
    svc.scan_state.cancel.set()
    return svc.scan_state.as_dict()


@app.post("/api/check")
async def check_draft(text: str = Form(default=""), file: UploadFile | None = File(default=None),
                      max_pairs: int = Form(default=8)):
    """Проверить проект документа на противоречия с базой, не добавляя его."""
    if file is not None and file.filename:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp.flush()
            pages, _ = await run_in_threadpool(extract_pdf, tmp.name)
    else:
        pages = [text]
    pages = [clean_text(p) for p in pages]
    chunks = chunk_pages(pages)
    if not chunks:
        raise HTTPException(400, "Нет текста для проверки")

    def run():
        vecs = svc.llm.embed([c["text"] for c in chunks])
        return contradictions.check_draft(chunks, vecs, svc.index, svc.store, svc.llm,
                                          max_pairs=max(1, min(max_pairs, 50)))

    try:
        return await run_in_threadpool(run)
    except OllamaError as e:
        raise HTTPException(503, str(e))
