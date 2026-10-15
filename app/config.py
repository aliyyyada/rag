"""Настройки берутся из переменных окружения (см. .env.example)."""
import os
from pathlib import Path


def _str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# Ollama
OLLAMA_URL = _str("OLLAMA_URL", "http://localhost:11434").rstrip("/")
LLM_MODEL = _str("LLM_MODEL", "qwen3:4b")
LLM_THINK = _bool("LLM_THINK", False)  # режим рассуждений Qwen3: точнее, но в разы медленнее
EMBED_MODEL = _str("EMBED_MODEL", "bge-m3")
OLLAMA_KEEP_ALIVE = _str("OLLAMA_KEEP_ALIVE", "30m")
LLM_NUM_CTX = _int("LLM_NUM_CTX", 4096)
LLM_NUM_THREAD = _int("LLM_NUM_THREAD", 4)  # i3-12100: 4 физических ядра
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.1)
OLLAMA_TIMEOUT = _float("OLLAMA_TIMEOUT", 600)
EMBED_BATCH = _int("EMBED_BATCH", 16)
MODEL_WAIT_SEC = _int("MODEL_WAIT_SEC", 20)  # как часто проверять, докачались ли модели

# Хранилище
DATA_DIR = Path(_str("DATA_DIR", "./data"))
DB_PATH = DATA_DIR / "rag.sqlite3"
FILES_DIR = DATA_DIR / "files"
WATCH_DIR = _str("WATCH_DIR", "")  # папка, из которой PDF подхватываются автоматически
WATCH_INTERVAL_SEC = _int("WATCH_INTERVAL_SEC", 300)

# Разбиение и поиск
CHUNK_SIZE = _int("CHUNK_SIZE", 900)  # символов
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)
TOP_K = _int("TOP_K", 5)
ANSWER_DETAILED = _bool("ANSWER_DETAILED", False)  # по умолчанию краткие ответы
ANSWER_MAX_TOKENS = _int("ANSWER_MAX_TOKENS", 250)
ANSWER_MAX_TOKENS_DETAILED = _int("ANSWER_MAX_TOKENS_DETAILED", 700)
CANDIDATES = _int("CANDIDATES", 30)

# OCR для сканов (Tesseract)
OCR_ENABLED = _bool("OCR_ENABLED", True)
OCR_LANG = _str("OCR_LANG", "rus+eng")
OCR_MIN_CHARS = _int("OCR_MIN_CHARS", 30)  # если на странице меньше текста — распознаём как скан

# Поиск противоречий
CONTRA_AUTO = _bool("CONTRA_AUTO", True)  # проверять каждый новый документ против базы
CONTRA_SIM_THRESHOLD = _float("CONTRA_SIM_THRESHOLD", 0.55)
CONTRA_NEIGHBORS = _int("CONTRA_NEIGHBORS", 3)
CONTRA_ONLY_NORMATIVE = _bool("CONTRA_ONLY_NORMATIVE", True)
CONTRA_MAX_PAIRS = _int("CONTRA_MAX_PAIRS", 300)

# SQL-источник уже распарсенных документов
SQL_SOURCE_URL = _str("SQL_SOURCE_URL", "")
SQL_SOURCE_QUERY = _str(
    "SQL_SOURCE_QUERY",
    "SELECT id, title, text, doc_date, doc_number FROM documents",
)
