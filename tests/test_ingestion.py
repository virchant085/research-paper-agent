"""End-to-end ingestion tests (offline: fake LLM, real parse/index/persist)."""
from __future__ import annotations

import fitz

from backend.core.ingestion import ingest_pdf
from backend.services import db
from backend.services.vectorstore import get_store

_PARA = (
    "This paragraph describes the experimental procedure in great detail so that "
    "the extracted text fills the page and produces several distinct chunks. "
) * 8


def _build_pdf(path) -> str:
    pages = [
        "A Real Looking Paper Title\n\nAbstract\n\n" + _PARA,
        "Introduction\n\n" + _PARA + "\n\nMethod\n\n" + _PARA,
    ]
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        rect = fitz.Rect(50, 50, 545, 780)
        page.insert_textbox(rect, text, fontsize=9)
    doc.save(str(path))
    doc.close()
    return str(path)


def test_ingest_pdf_ok_and_persisted(fake_llm, tmp_path):
    pdf = _build_pdf(tmp_path / "paper.pdf")

    resp = ingest_pdf(pdf, "paper.pdf")
    assert resp.status == "ok"
    assert resp.paper_id
    # Title comes from the fake structured_result ("Fake Paper").
    assert resp.title == "Fake Paper"

    # Card persisted in the DB with the structured fields.
    card = db.get_card(resp.paper_id)
    assert card is not None
    assert card.title == "Fake Paper"
    assert card.authors == ["A. One", "B. Two"]
    assert card.year == 2024
    assert card.source == "upload:paper.pdf"
    assert card.problem == "the problem"
    assert card.method == "the method"


def test_ingest_pdf_chunks_searchable(fake_llm, tmp_path):
    pdf = _build_pdf(tmp_path / "paper.pdf")
    resp = ingest_pdf(pdf, "paper.pdf")
    assert resp.status == "ok"

    # The chunks were indexed and are retrievable for this paper.
    store = get_store()
    hits = store.search("experimental procedure", paper_id=resp.paper_id, k=5)
    assert hits
    assert all(h.paper_id == resp.paper_id for h in hits)

    # Detected sections should include the planted headers.
    sections = set(store.sections_for(resp.paper_id))
    assert "Abstract" in sections or "Introduction" in sections or "Method" in sections


def test_ingest_pdf_bad_path_reports_error(fake_llm):
    resp = ingest_pdf("does-not-exist.pdf", "missing.pdf")
    assert resp.status == "error"
    assert resp.message
