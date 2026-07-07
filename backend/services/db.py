"""SQLite store for :class:`PaperCard` records.

A tiny stdlib ``sqlite3`` wrapper. A *fresh* connection is opened per call so the
module is safe to use from FastAPI's threaded request handlers. The ``authors``
list is serialized to JSON in a single ``TEXT`` column.

All paths come from :mod:`backend.config` (never hardcoded).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing

from backend.config import settings
from backend.models.schemas import PaperCard

# Columns of the ``papers`` table, in declaration order. ``authors``,
# ``key_terms`` and ``evidence`` are stored as JSON-encoded strings; every other
# field maps directly to a PaperCard attr.
_COLUMNS: tuple[str, ...] = (
    "paper_id",
    "title",
    "authors",
    "year",
    "source",
    "problem",
    "method",
    "dataset",
    "contribution",
    "limitation",
    "paper_type",
    "key_terms",
    "evidence",
)

# Columns added after the first release, with their DDL — applied via ALTER
# TABLE when opening an older database file.
_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("paper_type", "TEXT NOT NULL DEFAULT ''"),
    ("key_terms", "TEXT NOT NULL DEFAULT '[]'"),
    ("evidence", "TEXT NOT NULL DEFAULT '{}'"),
)


def _connect() -> sqlite3.Connection:
    """Open a fresh connection to the configured SQLite database.

    Row results are exposed as :class:`sqlite3.Row` so we can address columns by
    name. A new connection is created on every call (never cached) to keep the
    store thread-safe under FastAPI.
    """

    conn = sqlite3.connect(str(settings.sqlite_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Ensure data directories exist and create the ``papers`` table if missing."""

    settings.ensure_dirs()
    with closing(_connect()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                paper_id     TEXT PRIMARY KEY,
                title        TEXT NOT NULL DEFAULT '',
                authors      TEXT NOT NULL DEFAULT '[]',
                year         INTEGER,
                source       TEXT NOT NULL DEFAULT '',
                problem      TEXT NOT NULL DEFAULT '',
                method       TEXT NOT NULL DEFAULT '',
                dataset      TEXT NOT NULL DEFAULT '',
                contribution TEXT NOT NULL DEFAULT '',
                limitation   TEXT NOT NULL DEFAULT '',
                paper_type   TEXT NOT NULL DEFAULT '',
                key_terms    TEXT NOT NULL DEFAULT '[]',
                evidence     TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        # Migrate databases created before the nature-skills-inspired fields.
        existing = {
            row[1] for row in conn.execute("PRAGMA table_info(papers)").fetchall()
        }
        for column, ddl in _MIGRATIONS:
            if column not in existing:
                conn.execute(f"ALTER TABLE papers ADD COLUMN {column} {ddl}")
        conn.commit()


def _loads_json(raw: object, default: object) -> object:
    """Decode a JSON column value, falling back to ``default`` on any problem."""

    if not raw:
        return default
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default
    return value if isinstance(value, type(default)) else default


def _row_to_card(row: sqlite3.Row) -> PaperCard:
    """Convert a DB row into a :class:`PaperCard`, decoding the JSON columns."""

    authors = _loads_json(row["authors"], [])
    key_terms = _loads_json(row["key_terms"], [])
    evidence = _loads_json(row["evidence"], {})

    return PaperCard(
        paper_id=row["paper_id"],
        title=row["title"] or "",
        authors=[str(a) for a in authors],
        year=row["year"],
        source=row["source"] or "",
        problem=row["problem"] or "",
        method=row["method"] or "",
        dataset=row["dataset"] or "",
        contribution=row["contribution"] or "",
        limitation=row["limitation"] or "",
        paper_type=row["paper_type"] or "",
        key_terms=[str(t) for t in key_terms],
        evidence={str(k): str(v) for k, v in evidence.items()},
    )


def save_card(card: PaperCard) -> None:
    """Upsert a :class:`PaperCard` keyed by ``paper_id``.

    Existing rows with the same ``paper_id`` are fully overwritten.
    """

    init_db()
    values = (
        card.paper_id,
        card.title,
        json.dumps(list(card.authors), ensure_ascii=False),
        card.year,
        card.source,
        card.problem,
        card.method,
        card.dataset,
        card.contribution,
        card.limitation,
        card.paper_type,
        json.dumps(list(card.key_terms), ensure_ascii=False),
        json.dumps(dict(card.evidence), ensure_ascii=False),
    )
    placeholders = ", ".join(["?"] * len(_COLUMNS))
    columns = ", ".join(_COLUMNS)
    with closing(_connect()) as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO papers ({columns}) VALUES ({placeholders})",
            values,
        )
        conn.commit()


def get_card(paper_id: str) -> PaperCard | None:
    """Return the :class:`PaperCard` for ``paper_id`` or ``None`` if absent."""

    init_db()
    with closing(_connect()) as conn:
        cur = conn.execute(
            "SELECT * FROM papers WHERE paper_id = ?",
            (paper_id,),
        )
        row = cur.fetchone()
    return _row_to_card(row) if row is not None else None


def list_cards() -> list[PaperCard]:
    """Return all stored cards ordered by title (case-insensitive)."""

    init_db()
    with closing(_connect()) as conn:
        cur = conn.execute("SELECT * FROM papers ORDER BY title COLLATE NOCASE ASC")
        rows = cur.fetchall()
    return [_row_to_card(row) for row in rows]


def delete_card(paper_id: str) -> None:
    """Delete the card with the given ``paper_id`` (no-op if it does not exist)."""

    init_db()
    with closing(_connect()) as conn:
        conn.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,))
        conn.commit()


def paper_exists(paper_id: str) -> bool:
    """Return ``True`` if a card with ``paper_id`` is stored."""

    init_db()
    with closing(_connect()) as conn:
        cur = conn.execute(
            "SELECT 1 FROM papers WHERE paper_id = ? LIMIT 1",
            (paper_id,),
        )
        return cur.fetchone() is not None
