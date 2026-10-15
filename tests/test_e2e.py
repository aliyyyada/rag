"""Сквозной тест на поддельной Ollama: python -m pytest tests -q"""
import json
import os
import shutil
import socket
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from tests.fake_ollama import app as fake

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(fake, port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)

    data = tmp_path_factory.mktemp("data")
    os.environ.update(OLLAMA_URL=f"http://127.0.0.1:{port}", DATA_DIR=str(data),
                      CONTRA_SIM_THRESHOLD="0.05", CONTRA_NEIGHBORS="5",
                      MODEL_WAIT_SEC="1")
    samples = ROOT / "samples"
    if not samples.exists():
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_sample_pdfs.py")], check=True)

    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c, samples
    server.should_exit = True


def _wait(c, cond, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.2)
    raise AssertionError("не дождались")


def test_full_flow(client):
    c, samples = client
    assert c.get("/api/health").json()["ok"] is True

    files = [("files", (p.name, p.read_bytes(), "application/pdf")) for p in sorted(samples.glob("*.pdf"))]
    res = c.post("/api/documents", files=files).json()
    assert len(res) == 5 and not any(r.get("duplicate") for r in res)

    # повторная загрузка того же файла не создаёт дубль
    again = c.post("/api/documents", files=files[:1]).json()
    assert again[0]["duplicate"] is True

    _wait(c, lambda: all(d["status"] == "ready" for d in c.get("/api/documents").json()))
    docs = {d["title"]: d for d in c.get("/api/documents").json()}
    assert docs["Приказ 47 о фотофиксации дефектов"]["doc_date"] == "2024-06-03"
    assert docs["Скан - инструкция по охране труда"]["chunks"] >= 1  # OCR сработал

    # автопроверка противоречий после загрузки
    _wait(c, lambda: (s := c.get("/api/contradictions/status").json())["total"] and not s["running"])
    found = c.get("/api/contradictions").json()
    titles = {frozenset((x["a"]["title"], x["b"]["title"])) for x in found}
    assert frozenset(("Приказ 12 об информационной безопасности", "Приказ 47 о фотофиксации дефектов")) in titles
    assert frozenset(("Положение об отпусках", "Приказ 61 о графике отпусков")) in titles

    # повторный скан не проверяет уже проверенные пары
    c.post("/api/contradictions/scan", json={})
    _wait(c, lambda: not c.get("/api/contradictions/status").json()["running"])
    assert c.get("/api/contradictions/status").json()["total"] == 0

    # вопрос: ответ целиком, только использованный источник, противоречия только по нему
    r = c.post("/api/ask", json={"question": "Можно ли использовать iPhone для фотофиксации?"}).json()
    assert r["answer"] == "Нет, нельзя [1]."
    assert len(r["sources"]) == 1  # несуществующий номер 99 отброшен
    used = r["sources"][0]["chunk_id"]
    assert all(used in (x["chunk_a"], x["chunk_b"]) for x in r["contradictions"])
    assert r["contradictions"]  # первый источник — один из приказов про Apple

    # проверка проекта приказа
    draft = "Приказываю: заявление на отпуск подавать не позднее чем за 7 календарных дней."
    out = c.post("/api/check", data={"text": draft}).json()
    assert any("отпуск" in x["other"]["text"] for x in out)

    # отклонение и удаление документа каскадно удаляет его противоречия
    cid = found[0]["id"]
    c.patch(f"/api/contradictions/{cid}", json={"status": "dismissed"})
    assert c.get("/api/contradictions?status=dismissed").json()[0]["id"] == cid
    doc_id = docs["Приказ 61 о графике отпусков"]["id"]
    c.delete(f"/api/documents/{doc_id}")
    left = c.get("/api/contradictions").json()
    assert all(doc_id not in (x["a"]["doc_id"], x["b"]["doc_id"]) for x in left)


def test_sql_source(client, tmp_path):
    import sqlite3

    c, _ = client
    db = tmp_path / "src.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE documents (id INTEGER, title TEXT, text TEXT, doc_date TEXT, doc_number TEXT)")
    con.execute("INSERT INTO documents VALUES (1, 'Регламент пропусков', 'Пропуск выдаётся службой безопасности.', '05.05.2024', '9')")
    con.commit(); con.close()

    from app import config
    config.SQL_SOURCE_URL = f"sqlite:///{db}"
    r = c.post("/api/sql/sync").json()
    assert r["added"] == 1
    _wait(c, lambda: any(d["source"] == "sql" and d["status"] == "ready" for d in c.get("/api/documents").json()))
    d = next(d for d in c.get("/api/documents").json() if d["source"] == "sql")
    assert d["doc_date"] == "2024-05-05"
    assert c.post("/api/sql/sync").json()["unchanged"] == 1


def test_waits_for_embed_model(client):
    """Документ, загруженный до скачивания bge-m3, ждёт модель и обрабатывается сам."""
    from tests import fake_ollama

    c, _ = client
    fake_ollama.MODELS.remove("bge-m3:latest")
    try:
        import pymupdf
        doc = pymupdf.open()
        doc.new_page().insert_htmlbox(pymupdf.Rect(50, 50, 550, 800), "<p>Пропуск на территорию выдаёт охрана.</p>")
        res = c.post("/api/documents", files=[("files", ("Пропуска.pdf", doc.tobytes(), "application/pdf"))]).json()
        doc_id = res[0]["id"]
        _wait(c, lambda: c.get("/api/documents").json() and
              next(d for d in c.get("/api/documents").json() if d["id"] == doc_id)["status"] == "waiting")
        assert c.get("/api/health").json()["missing_models"] == ["bge-m3"]
    finally:
        fake_ollama.MODELS.append("bge-m3:latest")
    _wait(c, lambda: next(d for d in c.get("/api/documents").json() if d["id"] == doc_id)["status"] == "ready")
