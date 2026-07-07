"""Tests for the nature-skills-inspired features.

Covers the paper-type taxonomy, the terminology ledger (key_terms), verified
evidence quotes (source grounding), the six-dimension ``score_papers`` tool,
and the SQLite schema migration for databases created before these fields.
"""
from __future__ import annotations

import sqlite3

import fitz

from backend.config import settings
from backend.core import ingestion
from backend.core.tools import TOOL_REGISTRY, compare_papers, score_papers
from backend.models.schemas import PAPER_TYPES, PaperCard
from backend.services import db


def _card(paper_id="p1", **kw):
    base = dict(
        paper_id=paper_id,
        title=f"Title of {paper_id}",
        problem="prob",
        method="meth",
        dataset="data",
        contribution="contrib",
        limitation="limit",
    )
    base.update(kw)
    return PaperCard(**base)


def _make_pdf(tmp_path, text: str) -> str:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(72, 72, 540, 760), text, fontsize=9)
    path = str(tmp_path / "nature_test.pdf")
    doc.save(path)
    doc.close()
    return path


# --------------------------------------------------------------------------- #
# PaperCard new fields: db round-trip
# --------------------------------------------------------------------------- #
def test_db_roundtrip_paper_type_terms_evidence():
    card = _card(
        "np1",
        paper_type="algorithmic",
        key_terms=["HRTEM — high-resolution transmission electron microscopy"],
        evidence={"method": "we propose a transformer"},
    )
    db.save_card(card)
    got = db.get_card("np1")
    assert got is not None
    assert got.paper_type == "algorithmic"
    assert got.key_terms == ["HRTEM — high-resolution transmission electron microscopy"]
    assert got.evidence == {"method": "we propose a transformer"}


def test_db_defaults_for_new_fields():
    db.save_card(_card("np2"))
    got = db.get_card("np2")
    assert got.paper_type == ""
    assert got.key_terms == []
    assert got.evidence == {}


def test_db_migrates_pre_taxonomy_schema():
    """A database created before the new columns is upgraded in place."""
    # Build an old-schema DB file directly.
    conn = sqlite3.connect(str(settings.sqlite_path))
    conn.execute("DROP TABLE IF EXISTS papers")
    conn.execute(
        """
        CREATE TABLE papers (
            paper_id     TEXT PRIMARY KEY,
            title        TEXT NOT NULL DEFAULT '',
            authors      TEXT NOT NULL DEFAULT '[]',
            year         INTEGER,
            source       TEXT NOT NULL DEFAULT '',
            problem      TEXT NOT NULL DEFAULT '',
            method       TEXT NOT NULL DEFAULT '',
            dataset      TEXT NOT NULL DEFAULT '',
            contribution TEXT NOT NULL DEFAULT '',
            limitation   TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.execute(
        "INSERT INTO papers (paper_id, title) VALUES ('old1', 'Legacy Paper')"
    )
    conn.commit()
    conn.close()

    db.init_db()  # must ALTER TABLE, not fail

    got = db.get_card("old1")
    assert got is not None
    assert got.title == "Legacy Paper"
    assert got.paper_type == ""
    assert got.key_terms == []
    assert got.evidence == {}

    # And new-field saves now work against the migrated table.
    db.save_card(_card("new1", paper_type="review"))
    assert db.get_card("new1").paper_type == "review"


# --------------------------------------------------------------------------- #
# Ingestion: taxonomy, ledger, and verified evidence
# --------------------------------------------------------------------------- #
def test_ingestion_populates_taxonomy_and_verified_evidence(tmp_path, fake_llm):
    pdf = _make_pdf(
        tmp_path,
        "A Study of Copper Oxidation\n\n"
        "Abstract\nWe observe oxide growth in situ using HRTEM imaging. "
        "The oxide island nucleates at step edges. " * 5,
    )
    fake_llm.structured_result = {
        "title": "A Study of Copper Oxidation",
        "authors": ["C. Author"],
        "year": 2025,
        "problem": "How copper oxidizes",
        "method": "In-situ HRTEM",
        "dataset": "Cu thin films",
        "contribution": "Step-edge nucleation",
        "limitation": "Vacuum only",
        "paper_type": "research",
        "key_terms": ["HRTEM — high-resolution TEM imaging"],
        "evidence": {
            # Real quote: present in the PDF text (verbatim).
            "method": "oxide growth in situ using HRTEM imaging",
            # Fabricated quote: NOT in the PDF — must be dropped.
            "limitation": "experiments were limited to 300 K only",
        },
    }

    resp = ingestion.ingest_pdf(pdf, "oxide.pdf")
    assert resp.status == "ok", resp.message

    card = db.get_card(resp.paper_id)
    assert card.paper_type == "research"
    assert card.key_terms == ["HRTEM — high-resolution TEM imaging"]
    # The verified quote survives; the fabricated one is gone.
    assert "method" in card.evidence
    assert "limitation" not in card.evidence


def test_ingestion_rejects_unknown_paper_type(tmp_path, fake_llm):
    pdf = _make_pdf(tmp_path, "Some Paper\n\nAbstract\nContent here. " * 20)
    fake_llm.structured_result = {
        "title": "Some Paper",
        "authors": [],
        "problem": "p",
        "method": "m",
        "dataset": "d",
        "contribution": "c",
        "limitation": "l",
        "paper_type": "masterpiece",  # not in the taxonomy
    }
    resp = ingestion.ingest_pdf(pdf, "x.pdf")
    assert resp.status == "ok"
    assert db.get_card(resp.paper_id).paper_type == ""


def test_paper_types_constant_is_the_canonical_five():
    assert PAPER_TYPES == ("research", "methods", "hypothesis", "algorithmic", "review")


# --------------------------------------------------------------------------- #
# compare_papers can compare on paper_type
# --------------------------------------------------------------------------- #
def test_compare_papers_on_paper_type_dimension():
    db.save_card(_card("c1", title="Alpha", paper_type="methods"))
    db.save_card(_card("c2", title="Beta", paper_type="review"))
    table = compare_papers(["c1", "c2"], dimensions=["paper_type"])
    assert "| paper_type |" in table
    assert "methods" in table
    assert "review" in table


# --------------------------------------------------------------------------- #
# score_papers: six-dimension rubric enforced in code
# --------------------------------------------------------------------------- #
def test_score_papers_registered_as_tool():
    assert "score_papers" in TOOL_REGISTRY


def test_score_papers_requires_focus_and_papers(fake_llm):
    assert "No research focus" in score_papers("")
    assert "No papers" in score_papers("oxide growth")


def test_score_papers_caps_scores_and_applies_topic_gate(fake_llm):
    db.save_card(_card("s1", title="On Topic"))
    db.save_card(_card("s2", title="Off Topic"))

    fake_llm.structured_result = {
        "scores": [
            {
                "paper_id": "s1",
                # topic over the 35 cap -> must be clamped; total recalculated.
                "topic": 99,
                "method": 20,
                "venue": 15,
                "network": 10,
                "applied": 10,
                "archival": 10,
                "rationale": "highly relevant",
            },
            {
                "paper_id": "s2",
                # below the topic gate (10) -> rejected outright.
                "topic": 3,
                "method": 20,
                "venue": 15,
                "network": 10,
                "applied": 10,
                "archival": 10,
                "rationale": "prestigious but off-topic",
            },
        ]
    }

    out = score_papers("in-situ TEM oxide growth")
    # Capped: 35+20+15+10+10+10 = 100, not 99+... = 164.
    assert "| 1 | On Topic | 100 | 35 |" in out
    # Gate: the off-topic paper never appears as a ranked row...
    assert not any(
        line.startswith("|") and "Off Topic" in line for line in out.splitlines()
    )
    # ...and is reported in the rejection note instead.
    assert "Rejected by topic gate" in out
    assert "Off Topic (topic=3)" in out


def test_score_papers_handles_bad_model_output(fake_llm):
    db.save_card(_card("s1"))
    fake_llm.structured_result = {"nonsense": True}
    out = score_papers("anything")
    assert "Scoring failed" in out