"""Ingestion pipeline.

Orchestrates the full path from a raw source (uploaded PDF or arXiv link) to a
persisted, searchable paper:

    parse -> chunk -> index (vector store) -> extract PaperCard -> persist (sqlite)

Public entry points return an :class:`IngestResponse` and never raise; any error
is captured and reported through the response's ``status``/``message`` fields so
the API layer can surface it cleanly.
"""
from __future__ import annotations

from uuid import uuid4

from backend.config import settings
from backend.models.schemas import IngestResponse, PaperCard
from backend.services import db, parser
from backend.services.llm import get_llm
from backend.services.vectorstore import get_store

# Cap the amount of text handed to the LLM for card extraction. The most
# information-dense parts of a paper (title, abstract, intro, method) live near
# the front, so a leading slice is a good, cheap proxy for the whole document.
_MAX_CARD_CHARS = 24000


def ingest_pdf(path: str, source_name: str) -> IngestResponse:
    """Ingest a local PDF file.

    Parses the PDF into chunks, indexes them in the vector store, extracts a
    structured :class:`PaperCard`, and persists the card. Returns an
    :class:`IngestResponse` describing the outcome; failures are reported via
    ``status="error"`` rather than raised.

    Args:
        path: Absolute or relative path to the PDF on disk.
        source_name: Human-facing name of the source (typically the original
            filename); used for the card title fallback and ``source`` tag.

    Returns:
        IngestResponse with ``status="ok"`` on success, otherwise ``"error"``.
    """
    paper_id = ""
    title = ""
    try:
        settings.ensure_dirs()
        db.init_db()

        paper_id = uuid4().hex[:12]
        full_text, chunks = parser.parse_pdf(path, paper_id)
        title = parser.guess_title(full_text) or source_name

        get_store().add_chunks(chunks)

        card = _extract_card(paper_id, title, f"upload:{source_name}", full_text)
        db.save_card(card)

        return IngestResponse(
            paper_id=paper_id,
            title=card.title,
            status="ok",
        )
    except Exception as e:  # noqa: BLE001 - surface any failure to the caller
        return IngestResponse(
            paper_id=paper_id or "",
            title=title or source_name,
            status="error",
            message=str(e),
        )


def ingest_arxiv(url: str) -> IngestResponse:
    """Ingest a paper from an arXiv URL or bare id.

    Downloads the PDF from arXiv, then runs the same parse/index/extract/persist
    pipeline as :func:`ingest_pdf`. The arXiv-provided title is preferred over a
    heuristic guess from the PDF text. Failures are reported via the response.

    Args:
        url: An arXiv URL (``/abs/``, ``/pdf/``) or a bare id, optionally
            versioned (e.g. ``"2401.01234v2"``).

    Returns:
        IngestResponse with ``status="ok"`` on success, otherwise ``"error"``.
    """
    paper_id = ""
    title = ""
    aid = ""
    try:
        settings.ensure_dirs()
        db.init_db()

        aid = parser.extract_arxiv_id(url)
        pdf_path, arxiv_title = parser.download_arxiv(url, settings.upload_dir)

        paper_id = uuid4().hex[:12]
        full_text, chunks = parser.parse_pdf(pdf_path, paper_id)
        # Prefer the authoritative arXiv title; fall back to a heuristic guess,
        # then finally to the arXiv id itself.
        title = arxiv_title or parser.guess_title(full_text) or aid

        get_store().add_chunks(chunks)

        card = _extract_card(paper_id, title, f"arxiv:{aid}", full_text)
        db.save_card(card)

        return IngestResponse(
            paper_id=paper_id,
            title=card.title,
            status="ok",
        )
    except Exception as e:  # noqa: BLE001 - surface any failure to the caller
        return IngestResponse(
            paper_id=paper_id or "",
            title=title or aid,
            status="error",
            message=str(e),
        )


def _extract_card(paper_id: str, title: str, source: str, full_text: str) -> PaperCard:
    """Extract a structured :class:`PaperCard` from a paper's full text via the LLM.

    Sends a truncated slice of the document to the LLM with a JSON schema and
    assembles the result into a PaperCard. The provided ``paper_id`` and
    ``source`` are always preserved; the given ``title`` is used as a fallback
    when the model omits one.

    Args:
        paper_id: Stable id assigned at ingestion time.
        title: Best-known title, used as a fallback if the model omits it.
        source: Provenance tag, e.g. ``"upload:<name>"`` or ``"arxiv:<id>"``.
        full_text: The paper's extracted text (will be truncated for the prompt).

    Returns:
        A populated PaperCard. On extraction failure, a minimal card carrying
        the known title/source is returned rather than raising.
    """
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "authors": {"type": "array", "items": {"type": "string"}},
            "year": {"type": "integer"},
            "problem": {"type": "string"},
            "method": {"type": "string"},
            "dataset": {"type": "string"},
            "contribution": {"type": "string"},
            "limitation": {"type": "string"},
        },
        "required": [
            "title",
            "authors",
            "problem",
            "method",
            "dataset",
            "contribution",
            "limitation",
        ],
    }

    excerpt = full_text[:_MAX_CARD_CHARS]

    system = (
        "You are a meticulous research assistant that reads academic papers "
        "(often on TEM microscopy or robotics) and extracts a concise, "
        "structured summary card. Return ONLY a JSON object matching the "
        "provided schema. Be faithful to the text: if a field cannot be "
        "determined, use an empty string (or an empty list for authors). "
        "Keep each field to one or two sentences."
    )
    user = (
        f"Known title (may be imperfect): {title}\n\n"
        "Extract the following fields from the paper text below:\n"
        "- title: the paper's title\n"
        "- authors: list of author names\n"
        "- year: publication year (integer)\n"
        "- problem: the problem or question the paper addresses\n"
        "- method: the core method or approach\n"
        "- dataset: datasets, materials, or experimental setup used\n"
        "- contribution: the main contribution(s)\n"
        "- limitation: stated or evident limitations\n\n"
        "=== PAPER TEXT (may be truncated) ===\n"
        f"{excerpt}"
    )

    try:
        data = get_llm().structured(system, user, schema)
    except Exception:  # noqa: BLE001 - never let extraction failure abort ingestion
        data = {}

    if not isinstance(data, dict):
        data = {}

    extracted_title = str(data.get("title") or "").strip()
    card_title = extracted_title or title

    authors_raw = data.get("authors") or []
    if isinstance(authors_raw, list):
        authors = [str(a).strip() for a in authors_raw if str(a).strip()]
    elif isinstance(authors_raw, str):
        # Tolerate a comma-separated string if the model ignores the array type.
        authors = [a.strip() for a in authors_raw.split(",") if a.strip()]
    else:
        authors = []

    year = _coerce_year(data.get("year"))

    return PaperCard(
        paper_id=paper_id,
        title=card_title,
        authors=authors,
        year=year,
        source=source,
        problem=str(data.get("problem") or "").strip(),
        method=str(data.get("method") or "").strip(),
        dataset=str(data.get("dataset") or "").strip(),
        contribution=str(data.get("contribution") or "").strip(),
        limitation=str(data.get("limitation") or "").strip(),
    )


def _coerce_year(value: object) -> int | None:
    """Best-effort coercion of a model-supplied year into an int (or None)."""
    if value is None:
        return None
    if isinstance(value, bool):  # bool is a subclass of int; reject it explicitly
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # Pull the first 4-digit run if the model returned something like "2024.".
        digits = ""
        for ch in s:
            if ch.isdigit():
                digits += ch
                if len(digits) == 4:
                    break
            elif digits:
                break
        try:
            return int(digits) if digits else int(s)
        except ValueError:
            return None
    return None
