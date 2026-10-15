"""Ответ на вопрос по найденным фрагментам (retrieval-augmented generation).

Модель отвечает строго по JSON-схеме: так она не «рассуждает вслух», а сразу даёт ответ
и сама указывает, из каких фрагментов он взят — их и показываем как источники.
"""
import json
import logging
import re

from . import config
from .contradictions import _describe
from .db import Store
from .ollama_client import Ollama
from .search import Index

log = logging.getLogger(__name__)

SYSTEM = """Ты — справочная система по внутренним документам предприятия. Отвечай по-русски.
Используй ТОЛЬКО приведённые фрагменты документов, ничего не придумывай.
Не рассуждай и не пересказывай фрагменты: сразу дай ответ на вопрос.
Если во фрагментах ответа нет, ответь: «В загруженных документах ответ не найден» и оставь sources пустым.

Ответ в JSON:
- answer: ответ на вопрос;
- sources: номера фрагментов, из которых ДЕЙСТВИТЕЛЬНО взят ответ (обычно 1–2), например [1];
- contradiction: если фрагменты разных документов противоречат друг другу ИМЕННО по этому вопросу —
  одно предложение: какие документы и в чём конфликт; иначе пустая строка."""

SHORT = "Длина answer: 1–2 предложения."
DETAILED = "Длина answer: подробно, по пунктам, с указанием документов."

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "integer"}},
        "contradiction": {"type": "string"},
    },
    "required": ["answer", "sources", "contradiction"],
}

NOT_FOUND = "В загруженных документах ответ не найден."


def retrieve(store: Store, index: Index, llm: Ollama, question: str, top_k: int = config.TOP_K,
             doc_ids: list[int] | None = None) -> list[dict]:
    qvec = llm.embed([question])[0]
    hits = index.hybrid(question, qvec, top_k=top_k, candidates=config.CANDIDATES, doc_ids=doc_ids)
    chunks = store.chunks_by_ids([cid for cid, _ in hits])
    sources = []
    for n, (cid, score) in enumerate(hits, start=1):
        c = chunks.get(cid)
        if c:
            sources.append({"n": n, "chunk_id": cid, "score": round(score, 3), **c})
    return sources


def build_messages(question: str, sources: list[dict], detailed: bool = False) -> list[dict]:
    blocks = []
    for s in sources:
        page = f", стр. {s['page']}" if s.get("page") else ""
        blocks.append(f"[{s['n']}] {_describe(s)}{page}\n{s['text']}")
    context = "\n\n".join(blocks)
    user = f"Фрагменты документов:\n\n{context}\n\nВопрос: {question}"
    system = SYSTEM + "\n" + (DETAILED if detailed else SHORT)
    if not config.LLM_THINK and config.LLM_MODEL.startswith("qwen3"):
        system += "\n/no_think"  # «мягкий» выключатель рассуждений Qwen3
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse(raw: str) -> dict:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Ответ модели не JSON: %r", raw[:200])
        return {"answer": raw, "sources": [], "contradiction": ""}
    ids = []
    for n in data.get("sources") or []:
        try:
            ids.append(int(n))
        except (TypeError, ValueError):
            pass
    return {"answer": str(data.get("answer") or "").strip(), "sources": ids,
            "contradiction": str(data.get("contradiction") or "").strip()}


def answer(store: Store, index: Index, llm: Ollama, question: str, top_k: int = config.TOP_K,
           detailed: bool = False) -> dict:
    """Ответ целиком: текст, только использованные источники и противоречия по теме вопроса."""
    found = retrieve(store, index, llm, question, top_k)
    if not found:
        return {"answer": "В базе пока нет обработанных документов.", "sources": [], "contradictions": []}
    max_tokens = config.ANSWER_MAX_TOKENS_DETAILED if detailed else config.ANSWER_MAX_TOKENS
    raw = llm.chat(build_messages(question, found, detailed), fmt=ANSWER_SCHEMA, num_predict=max_tokens)
    data = _parse(raw)
    by_n = {s["n"]: s for s in found}
    used = [by_n[n] for n in dict.fromkeys(data["sources"]) if n in by_n]
    if not data["answer"]:
        data["answer"] = NOT_FOUND
    # Противоречия показываем только те, что касаются фрагментов, на которых построен ответ.
    used_ids = [s["chunk_id"] for s in used]
    contradictions = [c for c in store.list_contradictions(chunk_ids=used_ids)
                      if c["status"] != "dismissed"] if used_ids else []
    # Замечание модели о противоречии показываем, только если ответ опирается на разные документы
    # и в базе ещё нет проверенного противоречия по ним (маленькая модель склонна к ложным тревогам).
    note = data["contradiction"]
    if contradictions or len({s["doc_id"] for s in used}) < 2:
        note = ""
    return {"answer": data["answer"], "sources": used, "contradictions": contradictions,
            "contradiction_note": note}
