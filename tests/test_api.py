"""API-level tests using FastAPI's TestClient (offline)."""
from __future__ import annotations

import io

import fitz
from fastapi.testclient import TestClient

import backend.main
from backend.config import settings
from backend.models.schemas import PaperCard
from backend.services import db

client = TestClient(backend.main.app)


def _pdf_bytes() -> bytes:
    para = (
        "This paragraph fills the page so parsing yields multiple chunks that "
        "can be indexed and later searched by the vector store. "
    ) * 8
    doc = fitz.open()
    for text in ["A Title\n\nAbstract\n\n" + para, "Method\n\n" + para]:
        page = doc.new_page()
        rect = fitz.Rect(50, 50, 545, 780)
        page.insert_textbox(rect, text, fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm_model"] == settings.llm_model
    assert body["provider"] == settings.provider
    assert body["embed_model"] == settings.embed_model


def test_list_papers_empty():
    resp = client.get("/papers")
    assert resp.status_code == 200
    assert resp.json() == []


def test_table_returns_markdown_empty():
    resp = client.post("/table", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert "markdown" in body
    assert body["markdown"] == "No papers in the library yet."


def test_table_returns_markdown_with_papers():
    db.save_card(PaperCard(paper_id="p1", title="API Paper", year=2020))
    resp = client.post("/table", json={})
    assert resp.status_code == 200
    md = resp.json()["markdown"]
    assert "API Paper" in md
    assert "| Title | Year |" in md


def test_get_missing_paper_404():
    resp = client.get("/papers/nope")
    assert resp.status_code == 404


def test_get_existing_paper():
    db.save_card(PaperCard(paper_id="p1", title="API Paper", year=2020))
    resp = client.get("/papers/p1")
    assert resp.status_code == 200
    assert resp.json()["title"] == "API Paper"


def test_export_returns_path():
    db.save_card(PaperCard(paper_id="p1", title="API Paper", year=2020))
    resp = client.post("/export", json={"format": "markdown"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["format"] == "markdown"
    assert body["path"]
    # File should really exist on disk.
    from pathlib import Path

    assert Path(body["path"]).exists()


def test_compare_endpoint():
    db.save_card(PaperCard(paper_id="p1", title="Alpha", problem="prob1"))
    resp = client.post("/compare", json={"paper_ids": ["p1"]})
    assert resp.status_code == 200
    md = resp.json()["markdown"]
    assert "Alpha" in md
    assert "prob1" in md


def test_query_returns_answer(fake_llm):
    # fake_llm requested so the FakeLLM singleton is installed before the request.
    resp = client.post("/query", json={"question": "what is this?"})
    assert resp.status_code == 200
    body = resp.json()
    # Empty chat_script => FakeLLM returns "final answer".
    assert body["answer"] == "final answer"
    assert body["steps"] == []


def test_upload_pdf_ok(fake_llm):
    pdf = _pdf_bytes()
    resp = client.post(
        "/papers/upload",
        files={"file": ("p.pdf", pdf, "application/pdf")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["paper_id"]
    # Title from the fake structured extraction.
    assert body["title"] == "Fake Paper"

    # And the card is now listed.
    listed = client.get("/papers").json()
    assert any(c["paper_id"] == body["paper_id"] for c in listed)


def test_delete_paper():
    db.save_card(PaperCard(paper_id="p1", title="ToDelete"))
    resp = client.delete("/papers/p1")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert client.get("/papers/p1").status_code == 404
