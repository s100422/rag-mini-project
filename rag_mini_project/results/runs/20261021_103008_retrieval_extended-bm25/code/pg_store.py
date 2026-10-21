"""parent 원문을 PostgreSQL(RAG_POSTGRES_DB)에 저장한다."""
import json

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from .config import PgSettings, pg_settings
from .md_parser import Parent

SCHEMA = """
CREATE TABLE IF NOT EXISTS parents (
    parent_id      TEXT PRIMARY KEY,         -- 예: 04_신용거래약관::제3조
    doc_id         TEXT NOT NULL,            -- md 파일 이름(확장자 제외)
    doc_type       TEXT NOT NULL,            -- standard_terms 등
    source_file    TEXT NOT NULL,            -- 원본 PDF 파일 이름
    doc_title      TEXT NOT NULL,
    seq            INTEGER NOT NULL,         -- 문서 안 순서
    heading        TEXT NOT NULL,            -- 제3조(신용거래의 제한)
    article_no     TEXT,                     -- 제3조
    article_title  TEXT,                     -- 신용거래의 제한
    content        TEXT NOT NULL,            -- 제목 줄을 포함한 원문
    page_start     INTEGER NOT NULL,
    page_end       INTEGER NOT NULL,
    page_breaks    JSONB NOT NULL DEFAULT '[]',  -- [[content 안 위치, 쪽], ...]
    source_url     TEXT,
    effective_date TEXT,                     -- 원문 기준일
    version        TEXT,
    char_len       INTEGER NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- 골든셋 근거 ID (03_answers의 expected_source_ids와 같은 형식)
ALTER TABLE parents ADD COLUMN IF NOT EXISTS source_id TEXT;      -- kofia-terms-123-1709-015
ALTER TABLE parents ADD COLUMN IF NOT EXISTS source_doc_id TEXT;  -- kofia-terms-123-1709
CREATE INDEX IF NOT EXISTS parents_doc_idx ON parents (doc_type, doc_id, seq);
CREATE INDEX IF NOT EXISTS parents_source_id_idx ON parents (source_id);
"""

COLUMNS = ["parent_id", "doc_id", "doc_type", "source_file", "doc_title", "seq", "heading",
           "article_no", "article_title", "content", "page_start", "page_end", "page_breaks",
           "source_url", "effective_date", "version", "char_len", "source_id", "source_doc_id"]


def ensure_database(cfg: PgSettings | None = None) -> None:
    """RAG_POSTGRES_DB가 없으면 만든다(대문자 이름이라 따옴표로 감싼 그대로 만든다)."""
    cfg = cfg or pg_settings()
    with psycopg.connect(**cfg.conninfo("postgres"), autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (cfg.dbname,)).fetchone()
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(cfg.dbname)))
            print(f"created database {cfg.dbname}")


def connect(cfg: PgSettings | None = None) -> psycopg.Connection:
    cfg = cfg or pg_settings()
    return psycopg.connect(**cfg.conninfo())


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
    conn.commit()


def replace_parents(conn: psycopg.Connection, parents: list[Parent]) -> int:
    """parents에 들어 있는 문서(doc_id)의 기존 행을 지우고 새로 넣는다(다시 돌려도 같은 결과)."""
    doc_ids = sorted({p.doc_id for p in parents})
    insert = sql.SQL("INSERT INTO parents ({}) VALUES ({})").format(
        sql.SQL(", ").join(map(sql.Identifier, COLUMNS)),
        sql.SQL(", ").join(sql.Placeholder() * len(COLUMNS)),
    )
    rows = [(
        p.parent_id, p.doc_id, p.doc_type, p.source_file, p.doc_title, p.seq, p.heading,
        p.article_no, p.article_title, p.content, p.page_start, p.page_end,
        json.dumps(p.page_breaks), p.source_url, p.effective_date, p.version, len(p.content),
        p.source_id, p.source_doc_id,
    ) for p in parents]
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("DELETE FROM parents WHERE doc_id = ANY(%s)", (doc_ids,))
            cur.executemany(insert, rows)
    return len(rows)


def get_parents(conn: psycopg.Connection, parent_ids: list[str]) -> dict[str, dict]:
    """검색 때 child의 parent_id로 parent 원문을 가져온다."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM parents WHERE parent_id = ANY(%s)", (parent_ids,))
        return {r["parent_id"]: r for r in cur.fetchall()}
