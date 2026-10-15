"""Очистка текста, разбиение на фрагменты, извлечение реквизитов документа."""
import re

from . import config

# Слова, по которым фрагмент считается «нормативным» (содержит требование, запрет, разрешение).
# Противоречия ищем в первую очередь среди таких фрагментов: это сильно сокращает число проверок на CPU.
NORMATIVE_RE = re.compile(
    r"запрещ|запрет|не допуска|допуска|не разреша|разреша|обязан|обязательн|необходимо|"
    r"должн|следует|требуется|нельзя|только|исключительно|приказываю|не допускается|"
    r"не позднее|не ранее|не более|не менее|возложить|установить|утвердить|отменить",
    re.IGNORECASE,
)

MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
DATE_NUM_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
DATE_TXT_RE = re.compile(r"\b(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(\d{4})", re.IGNORECASE)
NUMBER_RE = re.compile(r"№\s*([0-9A-Za-zА-Яа-я\-/.]+)")


def clean_text(text: str) -> str:
    text = text.replace("­", "")  # мягкие переносы
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # перенос слова по слогам
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_normative(text: str) -> bool:
    return bool(NORMATIVE_RE.search(text))


def extract_meta(text: str) -> dict:
    """Номер и дата документа из начала текста (обычно шапка приказа)."""
    head = text[:1500]
    meta: dict = {"doc_number": None, "doc_date": None}
    if m := NUMBER_RE.search(head):
        meta["doc_number"] = m.group(1).rstrip(".")
    dates = []
    for m in DATE_NUM_RE.finditer(head):
        d, mo, y = map(int, m.groups())
        dates.append((m.start(), y, mo, d))
    for m in DATE_TXT_RE.finditer(head):
        dates.append((m.start(), int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1))))
    for _, y, mo, d in sorted(dates):
        if 1 <= mo <= 12 and 1 <= d <= 31 and 1950 <= y <= 2100:
            meta["doc_date"] = f"{y:04d}-{mo:02d}-{d:02d}"
            break
    return meta


def _split_long(paragraph: str, size: int) -> list[str]:
    """Режем слишком длинный абзац по предложениям, а если и они длинные — по словам."""
    sentences = re.split(r"(?<=[.!?;])\s+", paragraph)
    parts, cur = [], ""
    for s in sentences:
        while len(s) > size:
            cut = s.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(s[:cut].strip())
            s = s[cut:].strip()
        if len(cur) + len(s) + 1 > size and cur:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts


def chunk_pages(pages: list[str], size: int = config.CHUNK_SIZE,
                overlap: int = config.CHUNK_OVERLAP) -> list[dict]:
    """Склеивает абзацы в фрагменты ~size символов с перекрытием; запоминает номер страницы."""
    units: list[tuple[int, str]] = []
    for page_no, page in enumerate(pages, start=1):
        for para in re.split(r"\n\s*\n|\n(?=\s*(?:\d+[.)]|[-•–]) )", page):
            para = re.sub(r"\s*\n\s*", " ", para).strip()
            if not para:
                continue
            for piece in (_split_long(para, size) if len(para) > size else [para]):
                units.append((page_no, piece))

    chunks: list[dict] = []
    cur_text, cur_page = "", None
    for page_no, piece in units:
        if cur_text and len(cur_text) + len(piece) + 1 > size:
            chunks.append({"page": cur_page, "text": cur_text})
            tail = cur_text[-overlap:] if overlap else ""
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1 :]  # не начинаем с обрывка слова
            cur_text, cur_page = (f"{tail} {piece}".strip() if tail else piece), page_no
        else:
            cur_text = f"{cur_text}\n{piece}" if cur_text else piece
            cur_page = cur_page or page_no
    if cur_text:
        chunks.append({"page": cur_page, "text": cur_text})
    for c in chunks:
        c["normative"] = is_normative(c["text"])
    return chunks


def normalize_date(value: str | None) -> str | None:
    """'2024-03-15', '15.03.2024', '15 марта 2024' -> '2024-03-15'."""
    if not value:
        return None
    value = str(value).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}", value):
        return value[:10]
    return extract_meta(value)["doc_date"]
