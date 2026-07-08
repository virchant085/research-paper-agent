"""FastAPI route definitions for the research-paper-agent backend.

This module wires the HTTP surface to the service/core layer. Handlers are
declared ``async`` but delegate to synchronous service functions, which is fine
for a prototype (the calls are quick relative to network/LLM latency).

Endpoints
---------
- ``POST /papers/upload``   : ingest an uploaded PDF file.
- ``POST /papers/arxiv``    : ingest a paper by arXiv URL/id.
- ``GET  /papers``          : list all stored paper cards.
- ``GET  /papers/{id}``     : fetch a single paper card (404 if missing).
- ``DELETE /papers/{id}``   : remove a paper from the DB and the vector store.
- ``POST /query``           : run the agent over a question.
- ``POST /compare``         : produce a markdown comparison of papers (optional AI cross-analysis).
- ``POST /table``           : produce a markdown literature table.
- ``POST /review``          : synthesize a multi-paper literature review.
- ``POST /export``          : export cards/content to a file on disk.
- ``GET  /health``          : liveness + configured provider.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.config import settings
from backend.core import agent, ingestion, tools
from backend.models.schemas import (
    ArxivRequest,
    CompareRequest,
    ExportRequest,
    ExportResponse,
    IngestResponse,
    PaperCard,
    QueryRequest,
    QueryResponse,
    ReviewRequest,
    TableRequest,
)
from backend.services import db
from backend.services.vectorstore import get_store

router = APIRouter()


# --------------------------------------------------------------------------- #
# Paper ingestion
# --------------------------------------------------------------------------- #
@router.post("/papers/upload", response_model=IngestResponse)
async def upload_paper(file: UploadFile = File(...)) -> IngestResponse:
    """Persist an uploaded PDF to the upload dir, then ingest it.

    The raw bytes are written to ``settings.upload_dir/<filename>`` before being
    handed to the ingestion pipeline (parse -> chunk -> index -> extract card).
    """
    settings.ensure_dirs()

    filename = file.filename or "upload.pdf"
    # Guard against path traversal in the client-supplied filename.
    safe_name = Path(filename).name
    dest_path = Path(settings.upload_dir) / safe_name

    contents = await file.read()
    dest_path.write_bytes(contents)

    return ingestion.ingest_pdf(str(dest_path), safe_name)


@router.post("/papers/arxiv", response_model=IngestResponse)
async def ingest_arxiv(req: ArxivRequest) -> IngestResponse:
    """Ingest a paper identified by an arXiv URL or bare id."""
    return ingestion.ingest_arxiv(req.url)


# --------------------------------------------------------------------------- #
# Paper library CRUD
# --------------------------------------------------------------------------- #
@router.get("/papers", response_model=list[PaperCard])
async def list_papers() -> list[PaperCard]:
    """Return every stored paper card."""
    return db.list_cards()


@router.get("/papers/{paper_id}", response_model=PaperCard)
async def get_paper(paper_id: str) -> PaperCard:
    """Return a single paper card, or 404 if it does not exist."""
    card = db.get_card(paper_id)
    if card is None:
        raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")
    return card


@router.delete("/papers/{paper_id}")
async def delete_paper(paper_id: str) -> dict:
    """Delete a paper from both the card DB and the vector store."""
    db.delete_card(paper_id)
    get_store().delete_paper(paper_id)
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Agentic query & analysis
# --------------------------------------------------------------------------- #
@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """Run the tool-using agent over a natural-language question."""
    return agent.run_agent(req.question, req.paper_ids)


@router.post("/compare")
async def compare(req: CompareRequest) -> dict:
    """Return a markdown comparison across the requested papers."""
    return {
        "markdown": tools.compare_papers(
            req.paper_ids, req.dimensions, req.synthesize
        )
    }


@router.post("/table")
async def table(req: TableRequest) -> dict:
    """Return a markdown literature-review table."""
    return {"markdown": tools.generate_lit_table(req.paper_ids)}


@router.post("/review")
async def review(req: ReviewRequest) -> dict:
    """Return a synthesized multi-paper literature review (markdown)."""
    return {"markdown": tools.literature_review(req.paper_ids, req.focus)}


@router.post("/export", response_model=ExportResponse)
async def export(req: ExportRequest) -> ExportResponse:
    """Export cards or verbatim content to a file and return its path."""
    path = tools.export(req.format, req.paper_ids, req.content)
    return ExportResponse(path=path, format=req.format)


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@router.get("/health")
async def health() -> dict:
    """Liveness probe reporting the configured model and provider."""
    return {
        "status": "ok",
        "provider": settings.provider,
        "llm_model": settings.llm_model,
        "embed_model": settings.embed_model,
    }
