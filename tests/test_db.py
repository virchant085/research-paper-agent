"""Round-trip tests for the SQLite card store (backend.services.db)."""
from __future__ import annotations

from backend.models.schemas import PaperCard
from backend.services import db


def _card(paper_id="p1", **kw):
    base = dict(
        paper_id=paper_id,
        title="Deep Learning for TEM",
        authors=["Alice Zhang", "Bob Li"],
        year=2023,
        source="upload:paper.pdf",
        problem="the problem",
        method="the method",
        dataset="the dataset",
        contribution="the contribution",
        limitation="the limitation",
    )
    base.update(kw)
    return PaperCard(**base)


def test_save_and_get_roundtrip():
    card = _card()
    db.save_card(card)

    got = db.get_card("p1")
    assert got is not None
    assert got.paper_id == "p1"
    assert got.title == "Deep Learning for TEM"
    # authors list order and contents preserved through the JSON column
    assert got.authors == ["Alice Zhang", "Bob Li"]
    assert got.year == 2023
    assert got.source == "upload:paper.pdf"
    assert got.problem == "the problem"
    assert got.method == "the method"
    assert got.dataset == "the dataset"
    assert got.contribution == "the contribution"
    assert got.limitation == "the limitation"


def test_get_missing_returns_none():
    assert db.get_card("does-not-exist") is None


def test_paper_exists():
    assert db.paper_exists("p1") is False
    db.save_card(_card("p1"))
    assert db.paper_exists("p1") is True


def test_list_cards_ordered_by_title():
    db.save_card(_card("z", title="Zebra study"))
    db.save_card(_card("a", title="apple analysis"))
    db.save_card(_card("m", title="Middle paper"))

    cards = db.list_cards()
    assert len(cards) == 3
    titles = [c.title for c in cards]
    # ORDER BY title COLLATE NOCASE ASC => case-insensitive alphabetical
    assert titles == ["apple analysis", "Middle paper", "Zebra study"]


def test_delete_card():
    db.save_card(_card("p1"))
    assert db.paper_exists("p1") is True
    db.delete_card("p1")
    assert db.paper_exists("p1") is False
    assert db.get_card("p1") is None


def test_delete_missing_is_noop():
    # Deleting an absent id must not raise.
    db.delete_card("nope")
    assert db.list_cards() == []


def test_save_upserts_by_paper_id():
    db.save_card(_card("p1", title="Original"))
    db.save_card(_card("p1", title="Updated", authors=["Only One"]))

    got = db.get_card("p1")
    assert got is not None
    assert got.title == "Updated"
    assert got.authors == ["Only One"]
    # still only one row
    assert len(db.list_cards()) == 1


def test_empty_authors_preserved():
    db.save_card(_card("p1", authors=[]))
    got = db.get_card("p1")
    assert got is not None
    assert got.authors == []


def test_year_can_be_none():
    db.save_card(_card("p1", year=None))
    got = db.get_card("p1")
    assert got is not None
    assert got.year is None
