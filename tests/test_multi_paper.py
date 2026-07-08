"""Tests for the multi-paper features: compare cross-analysis + literature_review."""
from __future__ import annotations

from fastapi.testclient import TestClient

import backend.main as main
from backend.core.tools import compare_papers, literature_review
from backend.models.schemas import PaperCard
from backend.services import db
from backend.services.llm import ChatResult


def _card(paper_id="p1", **kw):
    base = dict(
        paper_id=paper_id,
        title=f"Title of {paper_id}",
        problem="prob-" + paper_id,
        method="meth-" + paper_id,
        dataset="data-" + paper_id,
        contribution="contrib-" + paper_id,
        limitation="limit-" + paper_id,
    )
    base.update(kw)
    return PaperCard(**base)


# --------------------------------------------------------------------------- #
# literature_review
# --------------------------------------------------------------------------- #
def test_literature_review_empty_library(fake_llm):
    assert "No papers" in literature_review()


def test_literature_review_synthesizes_over_cards(fake_llm):
    db.save_card(_card("r1", title="Paper One"))
    db.save_card(_card("r2", title="Paper Two"))
    fake_llm.chat_script = [
        ChatResult(content="## Overview\nA synthesis across two papers.", tool_calls=[])
    ]
    out = literature_review()
    assert "synthesis across two papers" in out.lower()
    # The synthesis prompt carried both cards as grounded context.
    ctx = fake_llm.chat_calls[-1]["messages"][-1]["content"]
    assert "Paper One" in ctx and "Paper Two" in ctx


def test_literature_review_single_paper_note(fake_llm):
    db.save_card(_card("r1", title="Only One"))
    out = literature_review()
    assert "at least two" in out.lower()


def test_literature_review_focus_steers_prompt(fake_llm):
    db.save_card(_card("r1"))
    db.save_card(_card("r2"))
    literature_review(focus="how they handle limited data")
    ctx = fake_llm.chat_calls[-1]["messages"][-1]["content"]
    assert "how they handle limited data" in ctx


def test_literature_review_restricted_to_ids(fake_llm):
    db.save_card(_card("r1", title="Included"))
    db.save_card(_card("r2", title="Excluded"))
    literature_review(paper_ids=["r1"])
    ctx = fake_llm.chat_calls[-1]["messages"][-1]["content"]
    assert "Included" in ctx and "Excluded" not in ctx


# --------------------------------------------------------------------------- #
# compare_papers cross-analysis (synthesize)
# --------------------------------------------------------------------------- #
def test_compare_without_synthesize_makes_no_llm_call(fake_llm):
    db.save_card(_card("c1"))
    db.save_card(_card("c2"))
    out = compare_papers(["c1", "c2"])  # synthesize defaults False
    assert "Cross-analysis" not in out
    assert fake_llm.chat_calls == []  # purely mechanical, no model call


def test_compare_with_synthesize_appends_analysis(fake_llm):
    db.save_card(_card("c1", title="Alpha"))
    db.save_card(_card("c2", title="Beta"))
    fake_llm.chat_script = [
        ChatResult(content="Alpha and Beta differ in their method.", tool_calls=[])
    ]
    out = compare_papers(["c1", "c2"], synthesize=True)
    assert "### Cross-analysis" in out
    assert "differ in their method" in out
    # Still contains the mechanical table above the analysis.
    assert "| Dimension |" in out
    assert fake_llm.chat_calls


def test_compare_synthesize_skipped_with_fewer_than_two_present(fake_llm):
    db.save_card(_card("c1"))
    out = compare_papers(["c1", "ghost"], synthesize=True)  # only one present
    assert "Cross-analysis" not in out
    assert fake_llm.chat_calls == []


# --------------------------------------------------------------------------- #
# API endpoints
# --------------------------------------------------------------------------- #
def test_review_endpoint_returns_markdown(fake_llm):
    db.save_card(_card("r1"))
    db.save_card(_card("r2"))
    client = TestClient(main.app)
    resp = client.post("/review", json={"paper_ids": None, "focus": None})
    assert resp.status_code == 200
    assert "markdown" in resp.json()


def test_compare_endpoint_accepts_synthesize(fake_llm):
    db.save_card(_card("c1"))
    db.save_card(_card("c2"))
    client = TestClient(main.app)
    resp = client.post(
        "/compare", json={"paper_ids": ["c1", "c2"], "synthesize": True}
    )
    assert resp.status_code == 200
    assert "### Cross-analysis" in resp.json()["markdown"]
