"""Поддельный Ollama для тестов без модели: детерминированные эмбеддинги и примитивный «судья».

Проверяет механику (загрузка, индекс, API, поток, хранение противоречий), а не качество модели.
"""
import hashlib
import json
import re

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()
DIM = 256


def _vec(text: str) -> list[float]:
    v = np.zeros(DIM, dtype=np.float32)
    for w in re.findall(r"[a-zа-яё0-9]+", text.lower()):
        h = int(hashlib.md5(w[:5].encode()).hexdigest(), 16)
        v[h % DIM] += 1.0
    return (v / max(np.linalg.norm(v), 1e-9)).tolist()


MODELS = ["qwen3:4b", "bge-m3:latest"]  # тест может временно убрать модель, имитируя загрузку


@app.get("/api/tags")
def tags():
    return {"models": [{"name": m} for m in MODELS]}


@app.post("/api/embed")
async def embed(req: Request):
    body = await req.json()
    return {"embeddings": [_vec(t) for t in body["input"]]}


def _judge(text: str) -> dict:
    a, _, b = text.partition("Фрагмент B")
    apple_ban = lambda s: "apple" in s.lower() and "запрет" in s.lower()
    apple_only = lambda s: "iphone" in s.lower() and "только" in s.lower()
    days = lambda s: set(re.findall(r"за (\d+) календарн", s))
    conflict = (apple_ban(a) and apple_only(b)) or (apple_ban(b) and apple_only(a)) or \
        bool(days(a) and days(b) and days(a) != days(b))
    return {"topic": "тест", "quote_a": "", "quote_b": "", "explanation": "фейковый судья",
            "contradiction": conflict}


@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    user = body["messages"][-1]["content"]
    fmt = body.get("format") or {}
    if isinstance(fmt, dict) and "answer" in fmt.get("properties", {}):
        # Имитация «болтливой» модели: рассуждения в <think> должны быть вырезаны.
        content = "<think>рассуждаю</think>" + json.dumps(
            {"answer": "Нет, нельзя [1].", "sources": [1, 99], "contradiction": ""}, ensure_ascii=False)
    elif fmt:
        content = json.dumps(_judge(user), ensure_ascii=False)
    else:
        content = "Ответ по источникам [1]."
    if not body.get("stream"):
        return {"message": {"role": "assistant", "content": content}, "done": True}

    def gen():
        for word in content.split(" "):
            yield json.dumps({"message": {"content": word + " "}, "done": False}) + "\n"
        yield json.dumps({"message": {"content": ""}, "done": True}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")
