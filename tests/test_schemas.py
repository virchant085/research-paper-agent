"""Tests for the Pydantic data contracts in backend.models.schemas."""
from __future__ import annotations

from backend.models.schemas import (
    ArxivRequest,
    Chunk,
    CompareRequest,
    ExportRequest,
    ExportResponse,
    IngestResponse,
    PaperCard,
    QueryRequest,
    QueryResponse,
    SearchHit,
    TableRequest,
    ToolStep,
)


def test_paper_card_defaults():
    card = PaperCard(paper_id="p1", title="A Title")
    assert card.paper_id == "p1"
    assert card.title == "A Title"
    # authors defaults to an (independent) empty list
    assert card.authors == []
    assert card.year is None
    assert card.source == ""
    assert card.problem == ""
    assert card.method == ""
    assert card.dataset == ""
    assert card.contribution == ""
    assert card.limitation == ""


def test_paper_card_default_list_is_not_shared():
    a = PaperCard(paper_id="a", title="A")
    b = PaperCard(paper_id="b", title="B")
    a.authors.append("Someone")
    # The default_factory must give each instance its own list.
    assert b.authors == []


def test_paper_card_roundtrip_authors():
    card = PaperCard(paper_id="p", title="t", authors=["X", "Y"], year=2020)
    assert card.authors == ["X", "Y"]
    assert card.year == 2020


def test_chunk_defaults():
    chunk = Chunk(chunk_id="p:0", paper_id="p", text="hello")
    assert chunk.chunk_id == "p:0"
    assert chunk.paper_id == "p"
    assert chunk.text == "hello"
    # section/page have sensible defaults
    assert chunk.section == "body"
    assert chunk.page == 0


def test_search_hit_fields():
    hit = SearchHit(text="t", paper_id="p", section="Method", page=3, score=0.9)
    assert hit.text == "t"
    assert hit.paper_id == "p"
    assert hit.section == "Method"
    assert hit.page == 3
    assert hit.score == 0.9


def test_query_request_defaults():
    req = QueryRequest(question="what is X?")
    assert req.question == "what is X?"
    assert req.paper_ids is None


def test_query_response_defaults():
    resp = QueryResponse(answer="the answer")
    assert resp.answer == "the answer"
    assert resp.steps == []


def test_ingest_response_defaults():
    resp = IngestResponse(paper_id="p1", title="t", status="ok")
    assert resp.status == "ok"
    # message defaults to empty string
    assert resp.message == ""


def test_export_request_defaults():
    req = ExportRequest()
    assert req.format == "markdown"
    assert req.paper_ids is None
    assert req.content is None


def test_compare_and_table_and_arxiv_requests():
    cmp = CompareRequest(paper_ids=["a", "b"])
    assert cmp.paper_ids == ["a", "b"]
    assert cmp.dimensions is None

    tbl = TableRequest()
    assert tbl.paper_ids is None

    arx = ArxivRequest(url="2401.01234")
    assert arx.url == "2401.01234"


def test_tool_step_and_export_response():
    step = ToolStep(tool="search_chunks", arguments={"query": "x"})
    assert step.tool == "search_chunks"
    assert step.arguments == {"query": "x"}
    assert step.result_preview == ""

    er = ExportResponse(path="/tmp/x.md", format="markdown")
    assert er.path == "/tmp/x.md"
    assert er.format == "markdown"
