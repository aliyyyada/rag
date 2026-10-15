"""Гибридный поиск: векторный (эмбеддинги) + лексический BM25, объединение через RRF.

Всё хранится в памяти процесса: 10 000 фрагментов x 1024 измерения — около 40 МБ.
Индексы перестраиваются автоматически, когда меняется набор фрагментов.
"""
import math
import re
import threading
from collections import Counter

import numpy as np

from .db import Store

STOPWORDS = set(
    "и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне "
    "было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до "
    "вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя "
    "их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого "
    "какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда "
    "можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая много "
    "разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда "
    "конечно всю между the of and to in a is for on".split()
)
# «нельзя» и «только» важны для поиска требований — не выкидываем их.
STOPWORDS -= {"нельзя", "только", "не"}


def tokenize(text: str) -> list[str]:
    """Нижний регистр + грубый стемминг усечением до 6 символов (работает для русских окончаний)."""
    words = re.findall(r"[a-zа-яё0-9]+", text.lower().replace("ё", "е"))
    return [w[:6] for w in words if w not in STOPWORDS and len(w) > 1]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf), dtype=np.float32)
        terms = [t for t in set(query) if t in self.idf]
        if not terms:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.len / max(self.avg, 1e-6))
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in terms:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm[i])
            out[i] = s
        return out


class Index:
    def __init__(self, store: Store):
        self.store = store
        self._lock = threading.Lock()
        self._version = -1
        self.ids = np.zeros(0, dtype=np.int64)
        self.doc_ids = np.zeros(0, dtype=np.int64)
        self.normative = np.zeros(0, dtype=bool)
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        self.bm25 = BM25([])

    def refresh(self):
        with self._lock:
            if self._version == self.store.version:
                return
            version = self.store.version
            rows = [r for r in self.store.all_chunks() if r["embedding"]]
            self.ids = np.array([r["id"] for r in rows], dtype=np.int64)
            self.doc_ids = np.array([r["doc_id"] for r in rows], dtype=np.int64)
            self.normative = np.array([bool(r["normative"]) for r in rows], dtype=bool)
            self.matrix = (
                np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
                if rows else np.zeros((0, 0), dtype=np.float32)
            )
            self.bm25 = BM25([tokenize(r["text"]) for r in rows])
            self._version = version

    def __len__(self):
        return len(self.ids)

    def vector_scores(self, qvec: np.ndarray) -> np.ndarray:
        if not len(self):
            return np.zeros(0, dtype=np.float32)
        return self.matrix @ qvec.astype(np.float32)

    def hybrid(self, query: str, qvec: np.ndarray, top_k: int, candidates: int = 30,
               doc_ids: list[int] | None = None) -> list[tuple[int, float]]:
        """Возвращает [(chunk_id, score)] по убыванию релевантности."""
        self.refresh()
        if not len(self):
            return []
        vec = self.vector_scores(qvec)
        lex = self.bm25.scores(tokenize(query))
        mask = np.ones(len(self), dtype=bool)
        if doc_ids:
            mask = np.isin(self.doc_ids, doc_ids)
        rrf = np.zeros(len(self), dtype=np.float32)
        for scores, use_zero in ((vec, True), (lex, False)):
            s = np.where(mask, scores, -np.inf)
            order = np.argsort(-s)[:candidates]
            for rank, i in enumerate(order):
                if not np.isfinite(s[i]) or (not use_zero and s[i] <= 0):
                    continue
                rrf[i] += 1.0 / (60 + rank)
        order = [i for i in np.argsort(-rrf)[:top_k] if rrf[i] > 0]
        return [(int(self.ids[i]), float(vec[i])) for i in order]
