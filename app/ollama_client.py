"""Минимальный клиент Ollama HTTP API (embed, chat, список моделей)."""
import json
from collections.abc import Iterator

import httpx
import numpy as np

from . import config


class OllamaError(RuntimeError):
    pass


class _RetryWithoutThink(Exception):
    def __init__(self, payload: dict):
        self.payload = payload


class Ollama:
    def __init__(self, url: str = config.OLLAMA_URL):
        self.url = url
        self._http = httpx.Client(base_url=url, timeout=httpx.Timeout(config.OLLAMA_TIMEOUT, connect=10))
        # Qwen3 и подобные умеют «думать» перед ответом — на CPU это в разы медленнее, отключаем.
        # Если модель не поддерживает параметр think, Ollama вернёт ошибку — тогда больше его не шлём.
        self._send_think = not config.LLM_THINK

    def _options(self, **extra) -> dict:
        opts = {
            "num_ctx": config.LLM_NUM_CTX,
            "temperature": config.LLM_TEMPERATURE,
        }
        if config.LLM_NUM_THREAD > 0:
            opts["num_thread"] = config.LLM_NUM_THREAD
        opts.update(extra)
        return opts

    def _chat_payload(self, messages: list[dict], stream: bool, options: dict) -> dict:
        payload = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "stream": stream,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": self._options(**options),
        }
        if self._send_think:
            payload["think"] = False
        return payload

    def _think_unsupported(self, status: int, body: str) -> bool:
        if self._send_think and status == 400 and "think" in body.lower():
            self._send_think = False
            return True
        return False

    def _post(self, path: str, payload: dict) -> dict:
        try:
            r = self._http.post(path, json=payload)
            if path == "/api/chat" and self._think_unsupported(r.status_code, r.text):
                payload.pop("think", None)
                r = self._http.post(path, json=payload)
        except httpx.HTTPError as e:
            raise OllamaError(f"Ollama недоступна ({self.url}): {e}") from e
        if r.status_code != 200:
            raise OllamaError(f"Ollama {path}: HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def models(self) -> list[str]:
        r = self._http.get("/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def embed(self, texts: list[str]) -> np.ndarray:
        """Эмбеддинги, нормированные по длине (cosine = скалярное произведение)."""
        out = []
        for i in range(0, len(texts), config.EMBED_BATCH):
            batch = texts[i : i + config.EMBED_BATCH]
            data = self._post(
                "/api/embed",
                {"model": config.EMBED_MODEL, "input": batch, "keep_alive": config.OLLAMA_KEEP_ALIVE},
            )
            out.extend(data["embeddings"])
        if not out:
            return np.zeros((0, 0), dtype=np.float32)
        arr = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-12)

    def chat(self, messages: list[dict], fmt: dict | str | None = None, **options) -> str:
        payload = self._chat_payload(messages, False, options)
        if fmt is not None:
            payload["format"] = fmt
        return self._post("/api/chat", payload)["message"]["content"]

    def chat_stream(self, messages: list[dict], **options) -> Iterator[str]:
        try:
            yield from self._chat_stream(self._chat_payload(messages, True, options))
        except _RetryWithoutThink as retry:
            yield from self._chat_stream(retry.payload)

    def _chat_stream(self, payload: dict) -> Iterator[str]:
        try:
            with self._http.stream("POST", "/api/chat", json=payload) as r:
                if r.status_code != 200:
                    r.read()
                    if self._think_unsupported(r.status_code, r.text):
                        payload.pop("think", None)
                        raise _RetryWithoutThink(payload)
                    raise OllamaError(f"Ollama /api/chat: HTTP {r.status_code}: {r.text[:300]}")
                for line in r.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise OllamaError(data["error"])
                    piece = data.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if data.get("done"):
                        break
        except httpx.HTTPError as e:
            raise OllamaError(f"Ollama недоступна ({self.url}): {e}") from e
