"""Источник уже распарсенных документов из SQL-базы.

Запрос (SQL_SOURCE_QUERY) должен вернуть колонки:
    id, title, text            — обязательно
    doc_date, doc_number       — необязательно (дата в формате yyyy-mm-dd или dd.mm.yyyy)
Если один документ хранится несколькими строками (например, по страницам или пунктам),
объедините их в запросе (string_agg / GROUP_CONCAT) или верните одинаковый id: строки склеятся.
"""
from dataclasses import dataclass

from .. import config


@dataclass
class SqlDocument:
    ref: str
    title: str
    text: str
    doc_date: str | None = None
    doc_number: str | None = None


def fetch_documents(url: str | None = None, query: str | None = None) -> list[SqlDocument]:
    url = url or config.SQL_SOURCE_URL
    query = query or config.SQL_SOURCE_QUERY
    if not url:
        raise ValueError("SQL_SOURCE_URL не задан")
    from sqlalchemy import create_engine, text  # импорт здесь: SQL-источник не обязателен

    engine = create_engine(url)
    docs: dict[str, SqlDocument] = {}
    try:
        with engine.connect() as conn:
            for row in conn.execute(text(query)).mappings():
                ref = str(row["id"])
                body = str(row["text"] or "")
                if ref in docs:
                    docs[ref].text += "\n\n" + body
                    continue
                docs[ref] = SqlDocument(
                    ref=ref,
                    title=str(row.get("title") or f"Документ {ref}"),
                    text=body,
                    doc_date=str(row["doc_date"]) if row.get("doc_date") else None,
                    doc_number=str(row["doc_number"]) if row.get("doc_number") else None,
                )
    finally:
        engine.dispose()
    return list(docs.values())
