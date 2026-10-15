"""Извлечение текста из PDF. Текстовый слой — через PyMuPDF, сканы — через Tesseract OCR."""
import logging
import os

import pymupdf

from .. import config

log = logging.getLogger(__name__)


def _tessdata() -> str | None:
    if os.environ.get("TESSDATA_PREFIX"):
        return os.environ["TESSDATA_PREFIX"]
    for path in ("/usr/share/tesseract-ocr/5/tessdata", "/usr/share/tesseract-ocr/4.00/tessdata",
                 "/usr/share/tessdata"):
        if os.path.isdir(path):
            return path
    return None


def extract_pdf(path: str) -> tuple[list[str], int]:
    """Возвращает (тексты страниц, сколько страниц распознано через OCR)."""
    pages: list[str] = []
    ocr_pages = 0
    tessdata = _tessdata() if config.OCR_ENABLED else None
    with pymupdf.open(path) as doc:
        for page in doc:
            text = page.get_text("text", sort=True)
            if len(text.strip()) < config.OCR_MIN_CHARS and tessdata:
                try:
                    tp = page.get_textpage_ocr(language=config.OCR_LANG, dpi=300, full=True,
                                               tessdata=tessdata)
                    text = page.get_text("text", textpage=tp, sort=True)
                    ocr_pages += 1
                except Exception as e:  # OCR не обязателен: пропускаем страницу, но не падаем
                    log.warning("OCR не удался для %s стр. %s: %s", path, page.number + 1, e)
            pages.append(text)
    return pages, ocr_pages
