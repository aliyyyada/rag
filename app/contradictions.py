"""Поиск противоречий между документами.

Схема (рассчитана на слабый CPU):
1. Кандидаты: для каждого «нормативного» фрагмента берём ближайшие по смыслу фрагменты
   ИЗ ДРУГИХ документов (cosine >= порога). Это дешёвая матричная операция.
2. Каждую пару-кандидат проверяет LLM и отвечает строго по JSON-схеме.
3. Проверенные пары запоминаются: при добавлении нового документа проверяются только его пары.
"""
import json
import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from . import config
from .db import Store
from .ollama_client import Ollama
from .search import Index

log = logging.getLogger(__name__)

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        # Порядок важен: сначала цитаты и рассуждение, потом вердикт — маленькие модели так точнее.
        "topic": {"type": "string"},
        "quote_a": {"type": "string"},
        "quote_b": {"type": "string"},
        "explanation": {"type": "string"},
        "contradiction": {"type": "boolean"},
    },
    "required": ["topic", "quote_a", "quote_b", "explanation", "contradiction"],
}

SYSTEM = """Ты — юрист предприятия. Тебе дают два фрагмента из РАЗНЫХ внутренних документов (приказы, инструкции, положения).
Определи, противоречат ли они друг другу.

Противоречие — это когда требования невозможно выполнить одновременно: один документ запрещает то, что другой требует или разрешает; указаны разные сроки, суммы, ответственные, порядок действий для одной и той же ситуации.
Пример противоречия: «запрещено использовать технику Apple» и «для фотофиксации использовать только iPhone» (iPhone — техника Apple).

НЕ противоречие: фрагменты о разных вещах; один документ дополняет или уточняет другой; требования относятся к явно разным подразделениям, объектам или периодам; документ прямо отменяет предыдущий.

Ответь JSON:
- topic: о чём оба фрагмента (2–5 слов);
- quote_a: короткая дословная цитата из фрагмента A, на которой основан вывод (или пустая строка);
- quote_b: короткая дословная цитата из фрагмента B (или пустая строка);
- explanation: 1–2 предложения, в чём конфликт или почему его нет;
- contradiction: true или false.
Отвечай по-русски."""


def _describe(c: dict) -> str:
    parts = [c.get("title") or "Документ"]
    if c.get("doc_number"):
        parts.append(f"№ {c['doc_number']}")
    if c.get("doc_date"):
        parts.append(f"от {c['doc_date']}")
    return " ".join(parts)


def judge_pair(llm: Ollama, a: dict, b: dict) -> dict:
    """Спрашивает LLM, противоречат ли два фрагмента. a и b — dict с text/title/doc_*."""
    user = (
        f"Фрагмент A ({_describe(a)}):\n\"\"\"\n{a['text']}\n\"\"\"\n\n"
        f"Фрагмент B ({_describe(b)}):\n\"\"\"\n{b['text']}\n\"\"\"\n\n"
        "Противоречат ли фрагменты A и B друг другу?"
    )
    raw = llm.chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        fmt=VERDICT_SCHEMA,
        temperature=0,
        num_predict=400,
    )
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("LLM вернула не-JSON: %r", raw[:200])
        return {"contradiction": False, "explanation": "Не удалось разобрать ответ модели", "error": True}
    verdict["contradiction"] = bool(verdict.get("contradiction"))
    return verdict


def candidate_pairs(index: Index, doc_ids: list[int] | None = None,
                    threshold: float = config.CONTRA_SIM_THRESHOLD,
                    neighbors: int = config.CONTRA_NEIGHBORS,
                    only_normative: bool = config.CONTRA_ONLY_NORMATIVE) -> list[tuple[int, int, float]]:
    """Пары (chunk_a, chunk_b, сходство) из разных документов, самые похожие первыми."""
    index.refresh()
    n = len(index)
    if n < 2:
        return []
    eligible = index.normative if only_normative else np.ones(n, dtype=bool)
    sources = eligible.copy()
    if doc_ids:
        sources &= np.isin(index.doc_ids, doc_ids)
    pairs: dict[tuple[int, int], float] = {}
    src_idx = np.flatnonzero(sources)
    for start in range(0, len(src_idx), 256):  # блоками, чтобы не держать в памяти матрицу n x n
        block = src_idx[start : start + 256]
        sims = index.matrix[block] @ index.matrix.T
        for row, i in enumerate(block):
            s = sims[row].copy()
            s[index.doc_ids == index.doc_ids[i]] = -1  # только другие документы
            s[~eligible] = -1
            for j in np.argsort(-s)[:neighbors]:
                if s[j] < threshold:
                    break
                a, b = int(index.ids[i]), int(index.ids[j])
                key = (min(a, b), max(a, b))
                pairs[key] = max(pairs.get(key, -1.0), float(s[j]))
    return sorted(((a, b, s) for (a, b), s in pairs.items()), key=lambda p: -p[2])


@dataclass
class ScanState:
    running: bool = False
    total: int = 0
    done: int = 0
    found: int = 0
    skipped: int = 0
    message: str = ""
    cancel: threading.Event = field(default_factory=threading.Event)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("running", "total", "done", "found", "skipped", "message")}


def scan(store: Store, index: Index, llm: Ollama, state: ScanState,
         doc_ids: list[int] | None = None, max_pairs: int = config.CONTRA_MAX_PAIRS):
    """Проверяет пары-кандидаты LLM-кой и сохраняет найденные противоречия."""
    state.running, state.done, state.found, state.skipped = True, 0, 0, 0
    state.cancel.clear()
    try:
        state.message = "Подбор пар-кандидатов"
        pairs = [p for p in candidate_pairs(index, doc_ids) if not store.is_checked(p[0], p[1])]
        state.skipped = max(0, len(pairs) - max_pairs)
        pairs = pairs[:max_pairs]
        state.total = len(pairs)
        state.message = "Проверка пар моделью" if pairs else "Новых пар для проверки нет"
        for a, b, sim in pairs:
            if state.cancel.is_set():
                state.message = "Остановлено пользователем"
                break
            chunks = store.chunks_by_ids([a, b])
            if a in chunks and b in chunks:
                verdict = judge_pair(llm, chunks[a], chunks[b])
                if verdict.get("contradiction"):
                    store.add_contradiction(a, b, sim, verdict)
                    state.found += 1
                if not verdict.get("error"):
                    store.mark_checked(a, b)
            state.done += 1
        else:
            if pairs:
                state.message = "Готово"
    except Exception as e:
        log.exception("Ошибка поиска противоречий")
        state.message = f"Ошибка: {e}"
    finally:
        state.running = False


def check_draft(chunks: list[dict], qvecs: np.ndarray, index: Index, store: Store, llm: Ollama,
                max_pairs: int = 8) -> list[dict]:
    """Проверка проекта документа (ещё не загруженного в базу) против всех документов."""
    index.refresh()
    if not len(index):
        return []
    sims = qvecs @ index.matrix.T
    cands = []
    for ci, row in enumerate(sims):
        for j in np.argsort(-row)[: config.CONTRA_NEIGHBORS]:
            if row[j] >= config.CONTRA_SIM_THRESHOLD:
                cands.append((ci, int(index.ids[j]), float(row[j])))
    cands.sort(key=lambda c: -c[2])
    seen, results = set(), []
    for ci, cid, sim in cands:
        if len(seen) >= max_pairs:
            break
        if (ci, cid) in seen:
            continue
        seen.add((ci, cid))
        other = store.chunks_by_ids([cid]).get(cid)
        if not other:
            continue
        draft = {"text": chunks[ci]["text"], "title": "Проверяемый проект"}
        verdict = judge_pair(llm, draft, other)
        if verdict.get("contradiction"):
            results.append({**verdict, "similarity": sim, "draft_text": draft["text"], "other": other})
    return results
