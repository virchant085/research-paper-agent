"""Tests for backend.services.parser (offline: no network / no download_arxiv)."""
from __future__ import annotations

import fitz
import pytest

from backend.services.parser import extract_arxiv_id, guess_title, parse_pdf

# A paragraph long enough that a few of them fill a page and force the
# character-based splitter (chunk_size ~1200) to emit several chunks.
_PARA = (
    "This paragraph describes the experimental procedure in great detail so that "
    "the extracted text fills the page and produces several distinct chunks when "
    "the character-based splitter runs over the concatenated document text. "
) * 8


def _build_pdf(path) -> str:
    """Write a small multi-page PDF whose pages actually fill with text.

    ``insert_textbox`` with a large rect + small fontsize is used so the text
    wraps and fills the page (``insert_text`` alone would clip past the margin).
    """
    pages = [
        "Neural Methods for Electron Microscopy\n\nAbstract\n\n" + _PARA,
        "Introduction\n\n" + _PARA + "\n\nMethod\n\n" + _PARA,
        "Results\n\n" + _PARA + "\n\nConclusion\n\n" + _PARA,
    ]
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        rect = fitz.Rect(50, 50, 545, 780)
        page.insert_textbox(rect, text, fontsize=9)
    doc.save(str(path))
    doc.close()
    return str(path)


def test_parse_pdf_multiple_chunks_and_provenance(tmp_path):
    pdf = _build_pdf(tmp_path / "paper.pdf")
    full_text, chunks = parse_pdf(pdf, "pid123")

    # Multiple chunks produced from a multi-page paper.
    assert len(chunks) >= 3

    # chunk_id uses the "<paper_id>:<i>" prefix and is sequential from 0.
    assert chunks[0].chunk_id == "pid123:0"
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"pid123:{i}"
        assert chunk.paper_id == "pid123"
        # Pages are 1-indexed.
        assert chunk.page >= 1
        assert chunk.text.strip()

    # Chunks span more than a single page of the multi-page PDF.
    pages = {c.page for c in chunks}
    assert len(pages) >= 2
    assert min(pages) >= 1

    assert isinstance(full_text, str) and full_text


def test_parse_pdf_detects_section_headers(tmp_path):
    pdf = _build_pdf(tmp_path / "paper.pdf")
    _full_text, chunks = parse_pdf(pdf, "pid123")

    sections = {c.section for c in chunks}
    # The canonical headers we planted must be detected and normalized (title-cased).
    assert "Abstract" in sections
    assert "Introduction" in sections
    assert "Method" in sections


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2401.01234", "2401.01234"),                              # bare id
        ("https://arxiv.org/abs/2401.01234", "2401.01234"),         # /abs/
        ("https://arxiv.org/pdf/2401.01234", "2401.01234"),         # /pdf/
        ("2401.01234v2", "2401.01234"),                             # versioned
        ("https://arxiv.org/pdf/2401.01234v3.pdf", "2401.01234"),   # .pdf suffix + version
        ("arxiv.org/abs/2401.01234v1", "2401.01234"),               # versioned /abs/
    ],
)
def test_extract_arxiv_id(value, expected):
    assert extract_arxiv_id(value) == expected


def test_extract_arxiv_id_old_style():
    assert extract_arxiv_id("https://arxiv.org/abs/hep-th/9901001") == "hep-th/9901001"


def test_guess_title_nonempty(tmp_path):
    pdf = _build_pdf(tmp_path / "paper.pdf")
    full_text, _chunks = parse_pdf(pdf, "pid123")
    title = guess_title(full_text)
    assert title
    assert "Neural Methods for Electron Microscopy" in title


def test_guess_title_empty_input():
    assert guess_title("") == ""
