"""Tests for the Chroma-backed vector store (uses the deterministic fake LLM).

The ``fake_llm`` fixture gives a stable per-text embedding, so searching for the
exact text of a chunk returns that chunk as the top hit.
"""
from __future__ import annotations

from backend.models.schemas import Chunk, SearchHit
from backend.services.vectorstore import get_store


def _chunks():
    return [
        Chunk(chunk_id="p1:0", paper_id="p1", section="Introduction",
              page=1, text="alpha introduction text about microscopy"),
        Chunk(chunk_id="p1:1", paper_id="p1", section="Method",
              page=2, text="beta method text describing the approach"),
        Chunk(chunk_id="p2:0", paper_id="p2", section="Introduction",
              page=1, text="gamma introduction for the second paper"),
        Chunk(chunk_id="p2:1", paper_id="p2", section="Results",
              page=3, text="delta results section of the second paper"),
    ]


def test_add_and_search_returns_hits(fake_llm):
    store = get_store()
    store.add_chunks(_chunks())

    hits = store.search("alpha introduction text about microscopy", k=4)
    assert hits
    assert all(isinstance(h, SearchHit) for h in hits)


def test_exact_text_is_top_hit(fake_llm):
    store = get_store()
    store.add_chunks(_chunks())

    target = "beta method text describing the approach"
    hits = store.search(target, k=4)
    assert hits
    # Identical text embeds identically -> distance 0 -> top result.
    assert hits[0].text == target
    assert hits[0].paper_id == "p1"
    assert hits[0].section == "Method"
    assert hits[0].page == 2


def test_paper_id_filter_narrows_results(fake_llm):
    store = get_store()
    store.add_chunks(_chunks())

    hits = store.search("introduction", paper_id="p2", k=10)
    assert hits
    assert all(h.paper_id == "p2" for h in hits)


def test_section_filter_narrows_results(fake_llm):
    store = get_store()
    store.add_chunks(_chunks())

    hits = store.search("something", section="Introduction", k=10)
    assert hits
    assert all(h.section == "Introduction" for h in hits)


def test_combined_paper_and_section_filter(fake_llm):
    store = get_store()
    store.add_chunks(_chunks())

    hits = store.search("anything", paper_id="p1", section="Method", k=10)
    assert hits
    assert all(h.paper_id == "p1" and h.section == "Method" for h in hits)


def test_sections_for_lists_distinct_sections(fake_llm):
    store = get_store()
    store.add_chunks(_chunks())

    p1_sections = store.sections_for("p1")
    assert set(p1_sections) == {"Introduction", "Method"}

    p2_sections = store.sections_for("p2")
    assert set(p2_sections) == {"Introduction", "Results"}


def test_delete_paper_removes_chunks(fake_llm):
    store = get_store()
    store.add_chunks(_chunks())

    store.delete_paper("p1")

    # p1 chunks are gone.
    assert store.sections_for("p1") == []
    hits = store.search("beta method text describing the approach", paper_id="p1", k=10)
    assert hits == []

    # p2 chunks remain.
    assert set(store.sections_for("p2")) == {"Introduction", "Results"}


def test_add_chunks_skips_empty_text(fake_llm):
    store = get_store()
    store.add_chunks([
        Chunk(chunk_id="p1:0", paper_id="p1", section="body", page=1, text="   "),
        Chunk(chunk_id="p1:1", paper_id="p1", section="body", page=1, text="real content here"),
    ])
    # Only the non-empty chunk is indexed.
    hits = store.search("real content here", paper_id="p1", k=10)
    assert len(hits) == 1
    assert hits[0].chunk_id if hasattr(hits[0], "chunk_id") else True
    assert hits[0].text == "real content here"
