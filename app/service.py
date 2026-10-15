"""Фоновая обработка: загрузка документов, эмбеддинги, поиск противоречий.

Все тяжёлые задачи идут через ОДИН рабочий поток: на 4-ядерном CPU параллельные
запросы к модели только мешают друг другу.
"""
import hashlib
import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config, contradictions
from .db import Store
from .ingest.pdf import extract_pdf
from .ingest.sql_source import fetch_documents
from .ollama_client import Ollama
from .search import Index
from .textproc import chunk_pages, clean_text, extract_meta, normalize_date

log = logging.getLogger(__name__)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Service:
    def __init__(self, store: Store | None = None, llm: Ollama | None = None):
        config.FILES_DIR.mkdir(parents=True, exist_ok=True)
        self.store = store or Store(config.DB_PATH)
        self.llm = llm or Ollama()
        self.index = Index(self.store)
        self.scan_state = contradictions.ScanState()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-worker")
        self._stop = threading.Event()
        self._pending_scan: set[int] = set()
        self._pending_lock = threading.Lock()
        self._seen_files: dict[str, float] = {}  # путь -> mtime, чтобы не хешировать папку заново

    # --- запуск/остановка ---
    def start(self):
        # PDF, обработка которых не завершилась до перезапуска. SQL-документы
        # переиндексируются при следующей синхронизации.
        for doc in self.store.documents_with_status("queued"):
            if doc["source"] == "pdf":
                self.worker.submit(self._process, doc["id"])
        # PDF, упавшие раньше из-за ещё не скачанной модели, обработаем заново.
        for doc in self.store.documents_with_status("error"):
            if doc["source"] == "pdf" and "not found" in (doc["error"] or ""):
                self.store.update_document(doc["id"], status="waiting", error=None)
        threading.Thread(target=self._waiting_loop, daemon=True, name="rag-wait").start()
        if config.WATCH_DIR:
            threading.Thread(target=self._watch_loop, daemon=True, name="rag-watch").start()

    def stop(self):
        self._stop.set()
        self.scan_state.cancel.set()
        self.worker.shutdown(wait=False, cancel_futures=True)

    # --- добавление документов ---
    def add_pdf(self, src: Path, title: str, move: bool = True) -> dict:
        """Регистрирует PDF и ставит его в очередь. Повторная загрузка того же файла игнорируется."""
        digest = file_hash(src)
        existing = self.store.find_by_hash(digest)
        if existing:
            if move:
                src.unlink(missing_ok=True)
            return {**existing, "duplicate": True}
        dest = config.FILES_DIR / f"{digest[:16]}.pdf"
        if move:
            shutil.move(str(src), dest)
        else:
            shutil.copyfile(src, dest)
        doc_id = self.store.add_document(title, "pdf", dest.name, digest)
        self.worker.submit(self._process, doc_id)
        return self.store.get_document(doc_id)

    def missing_models(self) -> list[str]:
        """Какие из нужных моделей ещё не скачаны в Ollama (пусто — всё готово)."""
        models = self.llm.models()

        def present(name: str) -> bool:
            base = name if ":" in name else f"{name}:latest"
            return any(m in (name, base) for m in models)

        return [m for m in (config.LLM_MODEL, config.EMBED_MODEL) if not present(m)]

    def _embed_model_ready(self) -> bool:
        try:
            return config.EMBED_MODEL not in self.missing_models()
        except Exception:
            return False  # Ollama ещё не поднялась

    def _waiting_loop(self):
        """Документы со статусом waiting ставятся в очередь, как только модель эмбеддингов доступна."""
        while not self._stop.wait(config.MODEL_WAIT_SEC):
            waiting = self.store.documents_with_status("waiting")
            if waiting and self._embed_model_ready():
                for doc in waiting:
                    self.store.update_document(doc["id"], status="queued")
                    self.worker.submit(self._process, doc["id"])

    def retry(self, doc_id: int) -> bool:
        doc = self.store.get_document(doc_id)
        if not doc or doc["source"] != "pdf" or doc["status"] in ("queued", "processing"):
            return False
        self.store.update_document(doc_id, status="queued", error=None)
        self.worker.submit(self._process, doc_id)
        return True

    def _process(self, doc_id: int):
        doc = self.store.get_document(doc_id)
        if not doc:
            return
        if not self._embed_model_ready():
            # Модель ещё качается (первый запуск) или Ollama не поднялась — подождём, а не упадём.
            self.store.update_document(doc_id, status="waiting",
                                       error=f"Ждём, пока в Ollama появится модель {config.EMBED_MODEL}")
            return
        self.store.update_document(doc_id, status="processing", error=None)
        try:
            if doc["source"] != "pdf":
                raise ValueError("Повторная обработка поддерживается только для PDF")
            pages, ocr_pages = extract_pdf(str(config.FILES_DIR / doc["source_ref"]))
            pages = [clean_text(p) for p in pages]
            if not any(pages):
                raise ValueError("В PDF не найден текст (скан без OCR?)")
            meta = extract_meta(pages[0])
            self._index_document(doc_id, doc["title"], pages)
            self.store.update_document(doc_id, status="ready", pages=len(pages), **meta)
            log.info("Документ %s готов: %s стр., OCR: %s", doc_id, len(pages), ocr_pages)
            self._after_ingest(doc_id)
        except Exception as e:
            log.exception("Ошибка обработки документа %s", doc_id)
            self.store.update_document(doc_id, status="error", error=str(e)[:500])

    def _index_document(self, doc_id: int, title: str, pages: list[str]):
        chunks = chunk_pages(pages)
        if not chunks:
            raise ValueError("Документ пустой")
        # Название документа добавляем к тексту фрагмента — так поиск лучше понимает контекст.
        vectors = self.llm.embed([f"{title}\n{c['text']}" for c in chunks])
        self.store.replace_chunks(doc_id, chunks, vectors)

    def _after_ingest(self, doc_id: int):
        if not config.CONTRA_AUTO:
            return
        # Поиск противоречий ставим в конец очереди и группируем: если загрузили пачку
        # документов, сначала проиндексируются все, затем пройдёт одна проверка.
        with self._pending_lock:
            first = not self._pending_scan
            self._pending_scan.add(doc_id)
        if first:
            self.worker.submit(self._run_pending_scan)

    def _run_pending_scan(self):
        with self._pending_lock:
            ids = sorted(self._pending_scan)
            self._pending_scan.clear()
        if ids:
            contradictions.scan(self.store, self.index, self.llm, self.scan_state, doc_ids=ids)

    def start_scan(self, doc_ids: list[int] | None = None, max_pairs: int = config.CONTRA_MAX_PAIRS) -> bool:
        if self.scan_state.running:
            return False
        self.scan_state.running = True  # чтобы второй клик не поставил вторую задачу
        self.scan_state.message = "В очереди"
        self.worker.submit(contradictions.scan, self.store, self.index, self.llm, self.scan_state,
                           doc_ids, max_pairs)
        return True

    # --- SQL-источник ---
    def sync_sql(self) -> dict:
        added = updated = unchanged = 0
        for d in fetch_documents():
            text = clean_text(d.text)
            digest = hashlib.sha256(text.encode()).hexdigest()
            existing = self.store.find_document("sql", d.ref)
            if existing and existing["content_hash"] == digest and existing["status"] == "ready":
                unchanged += 1
                continue
            if existing:
                doc_id = existing["id"]
                self.store.update_document(doc_id, title=d.title, content_hash=digest, status="queued")
                updated += 1
            else:
                doc_id = self.store.add_document(d.title, "sql", d.ref, digest)
                added += 1
            meta = extract_meta(text)
            self.store.update_document(
                doc_id,
                doc_date=normalize_date(d.doc_date) or meta["doc_date"],
                doc_number=d.doc_number or meta["doc_number"],
            )
            self.worker.submit(self._process_text, doc_id, d.title, text)
        return {"added": added, "updated": updated, "unchanged": unchanged}

    def _process_text(self, doc_id: int, title: str, text: str):
        self.store.update_document(doc_id, status="processing", error=None)
        try:
            self._index_document(doc_id, title, [text])
            self.store.update_document(doc_id, status="ready", pages=1)
            self._after_ingest(doc_id)
        except Exception as e:
            log.exception("Ошибка обработки документа %s", doc_id)
            self.store.update_document(doc_id, status="error", error=str(e)[:500])

    # --- папка для автоматической загрузки ---
    def scan_watch_dir(self) -> int:
        root = Path(config.WATCH_DIR)
        if not root.is_dir():
            return 0
        added = 0
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() != ".pdf" or not path.is_file():
                continue
            mtime = path.stat().st_mtime
            if self._seen_files.get(str(path)) == mtime:
                continue
            self._seen_files[str(path)] = mtime
            res = self.add_pdf(path, path.stem, move=False)
            added += 0 if res.get("duplicate") else 1
        return added

    def _watch_loop(self):
        while not self._stop.is_set():
            try:
                n = self.scan_watch_dir()
                if n:
                    log.info("Из папки %s добавлено документов: %s", config.WATCH_DIR, n)
            except Exception:
                log.exception("Ошибка при просмотре папки %s", config.WATCH_DIR)
            self._stop.wait(config.WATCH_INTERVAL_SEC)
