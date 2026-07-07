"""Tests for the agent tools in backend.core.tools."""
from __future__ import annotations

import csv
from pathlib import Path

from backend.config import settings
from backend.core import tools
from backend.core.tools import (
    _escape_cell,
    compare_papers,
    export,
    generate_lit_table,
    search_chunks,
    summarize_section,
)
from backend.models.schemas import Chunk, PaperCard
from backend.services import db
from backend.services.vectorstore import get_store


def _card(paper_id="p1", **kw):
    base = dict(
        paper_id=paper_id,
        title=f"Title of {paper_id}",
        authors=["Alice", "Bob"],
        year=2022,
        source="upload:x.pdf",
        problem="prob-" + paper_id,
        method="meth-" + paper_id,
        dataset="data-" + paper_id,
        contribution="contrib-" + paper_id,
        limitation="limit-" + paper_id,
    )
    base.update(kw)
    return PaperCard(**base)


# --------------------------------------------------------------------------- #
# generate_lit_table
# --------------------------------------------------------------------------- #
def test_generate_lit_table_empty_message():
    msg = generate_lit_table()
    assert msg == "No papers in the library yet."


def test_generate_lit_table_contains_titles_and_header():
    db.save_card(_card("p1", title="First Paper"))
    db.save_card(_card("p2", title="Second Paper"))

    table = generate_lit_table()
    # Header row present.
    assert "| Title | Year | Problem | Method | Dataset | Contribution | Limitation |" in table
    # Divider row present.
    assert "| --- | --- | --- | --- | --- | --- | --- |" in table
    # Both titles present.
    assert "First Paper" in table
    assert "Second Paper" in table


def test_generate_lit_table_restricted_to_ids():
    db.save_card(_card("p1", title="First Paper"))
    db.save_card(_card("p2", title="Second Paper"))
    table = generate_lit_table(["p1"])
    assert "First Paper" in table
    assert "Second Paper" not in table


# --------------------------------------------------------------------------- #
# compare_papers
# --------------------------------------------------------------------------- #
def test_compare_papers_builds_table_with_row_per_dimension():
    db.save_card(_card("p1", title="Alpha"))
    db.save_card(_card("p2", title="Beta"))

    table = compare_papers(["p1", "p2"])
    lines = table.splitlines()

    # Header carries the paper labels (titles).
    assert "Alpha" in lines[0]
    assert "Beta" in lines[0]
    assert lines[0].startswith("| Dimension |")

    # One row per default dimension.
    for dim in ["problem", "method", "dataset", "contribution", "limitation"]:
        assert any(line.startswith(f"| {dim} |") for line in lines), dim

    # Cell values are populated from the cards.
    assert "prob-p1" in table
    assert "prob-p2" in table


def test_compare_papers_custom_dimensions():
    db.save_card(_card("p1", title="Alpha"))
    table = compare_papers(["p1"], dimensions=["method"])
    lines = table.splitlines()
    assert any(line.startswith("| method |") for line in lines)
    assert not any(line.startswith("| problem |") for line in lines)


def test_compare_papers_reports_missing_ids():
    db.save_card(_card("p1", title="Alpha"))
    table = compare_papers(["p1", "ghost"])
    assert "ghost" in table
    assert "no card found for" in table.lower()
    # Missing cells rendered as "_not found_".
    assert "_not found_" in table


def test_compare_papers_empty_ids():
    assert compare_papers([]) == "No paper ids provided to compare."


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def test_export_markdown_writes_file_under_export_dir():
    db.save_card(_card("p1", title="Exported Paper"))
    path = export("markdown")
    p = Path(path)
    assert p.exists()
    # Written under the configured export dir.
    assert p.parent == Path(settings.export_dir)
    assert p.suffix == ".md"
    text = p.read_text(encoding="utf-8")
    assert "Exported Paper" in text
    assert "| Title | Year |" in text


def test_export_csv_writes_file_with_contents():
    db.save_card(_card("p1", title="CSV Paper"))
    path = export("csv")
    p = Path(path)
    assert p.exists()
    assert p.parent == Path(settings.export_dir)
    assert p.suffix == ".csv"

    with p.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    row = rows[0]
    assert row["title"] == "CSV Paper"
    assert row["paper_id"] == "p1"
    # authors joined with "; "
    assert row["authors"] == "Alice; Bob"
    assert row["year"] == "2022"


def test_export_content_written_verbatim():
    content = "# My Answer\n\nSome **markdown** content here."
    path = export(content=content)
    p = Path(path)
    assert p.exists()
    assert p.parent == Path(settings.export_dir)
    assert p.read_text(encoding="utf-8") == content


# --------------------------------------------------------------------------- #
# _escape_cell
# --------------------------------------------------------------------------- #
def test_escape_cell_flattens_pipes_and_newlines():
    out = _escape_cell("a | b\nc\r\nd")
    # Pipes escaped, newlines flattened to spaces.
    assert "\n" not in out
    assert "\r" not in out
    assert "\\|" in out
    assert "|" not in out.replace("\\|", "")


def test_escape_cell_none_and_nonstring():
    assert _escape_cell(None) == ""
    assert _escape_cell(2024) == "2024"


# --------------------------------------------------------------------------- #
# search_chunks / summarize_section (need fake_llm for embeddings + chat)
# --------------------------------------------------------------------------- #
def test_search_chunks_formats_hits(fake_llm):
    store = get_store()
    store.add_chunks([
        Chunk(chunk_id="p1:0", paper_id="p1", section="Method",
              page=2, text="the special retrieval sentence"),
    ])
    out = search_chunks("the special retrieval sentence", paper_id="p1")
    assert "the special retrieval sentence" in out
    # Provenance tag present.
    assert "p1" in out
    assert "Method" in out
    assert "p.2" in out


def test_search_chunks_no_results(fake_llm):
    # Nothing indexed for this paper.
    out = search_chunks("anything", paper_id="does-not-exist")
    assert out == "No matching chunks found."


def test_summarize_section_uses_llm(fake_llm):
    store = get_store()
    store.add_chunks([
        Chunk(chunk_id="p1:0", paper_id="p1", section="Method",
              page=1, text="method details worth summarizing"),
    ])
    out = summarize_section("p1", "Method")
    # FakeLLM.chat returns "final answer" when the script is empty.
    assert out == "final answer"


def test_summarize_section_no_content(fake_llm):
    out = summarize_section("p1", "Nonexistent")
    assert "No content found" in out


def test_tool_registry_and_schemas_present():
    assert set(tools.TOOL_REGISTRY) == {
        "search_chunks",
        "summarize_section",
        "compare_papers",
        "generate_lit_table",
        "export",
    }
    names = {s["function"]["name"] for s in tools.TOOL_SCHEMAS}
    assert names == set(tools.TOOL_REGISTRY)
