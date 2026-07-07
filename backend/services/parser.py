"""PDF and arXiv parsing plus character-based chunking.

Uses PyMuPDF (``import fitz``) for text extraction and ``requests`` +
``xml.etree`` for arXiv metadata. All chunking knobs come from
``backend.config.settings`` (never hardcoded).

Public surface:
- ``parse_pdf(path, paper_id) -> (full_text, chunks)``
- ``extract_arxiv_id(url_or_id) -> str``
- ``download_arxiv(url_or_id, dest_dir) -> (local_pdf_path, title)``
- ``guess_title(full_text) -> str``
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple
from xml.etree import ElementTree as ET

from backend.config import settings
from backend.models.schemas import Chunk

# --------------------------------------------------------------------------- #
# Section-header detection
# --------------------------------------------------------------------------- #
# Canonical section keywords (matched case-insensitively). Order does not
# matter for detection; each is treated as a potential standalone heading.
_SECTION_KEYWORDS = [
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "methodology",
    "approach",
    "model",
    "experiment",
    "experiments",
    "experimental setup",
    "results",
    "evaluation",
    "analysis",
    "discussion",
    "conclusion",
    "conclusions",
    "future work",
    "references",
    "acknowledgments",
    "acknowledgements",
    "appendix",
]

# A heading line is short-ish and either matches a known keyword, or looks like
# a numbered heading such as "3 Method", "3.1 Data", "II. Related Work".
_KEYWORD_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+|[IVXLC]+\.?\s+)?(" + "|".join(
        re.escape(k) for k in sorted(_SECTION_KEYWORDS, key=len, reverse=True)
    ) + r")\b\s*[:.]?\s*$",
    re.IGNORECASE,
)

# A generic numbered heading: "3 Something", "3.1 Something Else". The title
# portion is validated separately (see ``_looks_like_heading``) so that body
# sentences beginning with a number are not mistaken for headings.
_NUMBERED_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+(.+?)\s*$")

# Lowercase function words that betray prose rather than a section title.
_PROSE_WORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "that",
    "which", "with", "we", "our", "this", "is", "are", "was", "were", "by",
    "as", "at", "from", "it", "these", "consists", "based",
}


def _looks_like_heading(title: str) -> bool:
    """Heuristic: does ``title`` read like a section heading, not prose?

    A heading is short (few words), contains no obvious prose function words,
    and is dominated by capitalized/title-cased words.
    """
    words = title.split()
    if not words or len(words) > 7:
        return False
    lower = [w.lower().strip(".,;:") for w in words]
    if any(w in _PROSE_WORDS for w in lower):
        return False
    # Require most alphabetic words to start with an uppercase letter (or be an
    # acronym), which distinguishes "Data Preprocessing" from "our pipeline".
    alpha = [w for w in words if w[:1].isalpha()]
    if not alpha:
        return False
    capitalized = sum(1 for w in alpha if w[:1].isupper())
    return capitalized >= max(1, (len(alpha) + 1) // 2)


def _detect_section(line: str) -> str | None:
    """Return a normalized section label if ``line`` looks like a header.

    Returns ``None`` when the line is ordinary body text.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None

    m = _KEYWORD_RE.match(stripped)
    if m:
        # Normalize to a clean title-cased label from the matched keyword.
        return m.group(1).strip().title()

    # Numbered heading heuristic. Reject lines ending in sentence punctuation
    # (almost certainly prose) and validate the title looks heading-like.
    if stripped.endswith((".", ",", ";", ":")) and not stripped.endswith("..."):
        return None
    m = _NUMBERED_RE.match(stripped)
    if m:
        title = m.group(2).strip()
        if title and _looks_like_heading(title):
            return f"{m.group(1)} {title}"

    return None


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def _split_text(text: str, size: int, overlap: int) -> List[Tuple[int, int]]:
    """Compute (start, end) character spans for ``text``.

    Splits are aligned to paragraph or sentence boundaries where possible, so a
    chunk rarely ends mid-sentence. Falls back to a hard character cut when no
    boundary is found within the window. Returns index spans into ``text``.
    """
    spans: List[Tuple[int, int]] = []
    n = len(text)
    if n == 0:
        return spans

    size = max(1, size)
    overlap = max(0, min(overlap, size - 1))

    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Prefer to break on a paragraph boundary, else sentence, else
            # whitespace, searching within the latter part of the window so
            # chunks stay reasonably full. ``search_floor`` bounds how far back
            # a boundary may pull the cut (never past half the window).
            window = text[start:end]
            search_floor = max(1, len(window) // 2)
            cut = -1

            para = window.rfind("\n\n")
            if para >= search_floor:
                cut = para + 2
            if cut < 0:
                # Last sentence boundary (". ", ".\n", "! ", "? ") past floor.
                last = None
                for m in re.finditer(r"[.!?][\s]", window):
                    if m.end() >= search_floor:
                        last = m.end()
                if last is not None:
                    cut = last
            if cut < 0:
                ws = window.rfind(" ")
                if ws >= search_floor:
                    cut = ws + 1

            if cut > 0:
                end = start + cut

        spans.append((start, end))
        if end >= n:
            break
        # Advance from the actual end, applying overlap. Clamp so we always make
        # forward progress (never revisit or stall) while preserving overlap
        # relative to where this chunk truly ended.
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return spans


def parse_pdf(path: str, paper_id: str) -> Tuple[str, List[Chunk]]:
    """Extract text from a PDF and split it into provenance-tagged chunks.

    Args:
        path: Filesystem path to the PDF.
        paper_id: Identifier assigned to this paper; used for chunk ids.

    Returns:
        ``(full_text, chunks)`` where ``full_text`` is the concatenated page
        text (page-break separated) and ``chunks`` is a list of
        :class:`~backend.models.schemas.Chunk`. Each chunk's ``section`` is the
        nearest preceding detected header (default ``"body"``) and ``page`` is
        the 1-indexed page on which the chunk starts.
    """
    import fitz  # PyMuPDF; imported lazily so the module imports without it.

    # Extract per-page text and remember where each page begins in full_text so
    # we can map chunk start offsets back to a page number.
    page_texts: List[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            page_texts.append(page.get_text("text") or "")

    # Build full_text, tracking (char_offset_of_page_start, page_number).
    full_parts: List[str] = []
    page_starts: List[Tuple[int, int]] = []  # (offset, page_number 1-indexed)
    offset = 0
    for idx, ptext in enumerate(page_texts):
        page_starts.append((offset, idx + 1))
        full_parts.append(ptext)
        offset += len(ptext)
        # Page separator (form feed keeps page boundaries visible in full_text).
        sep = "\n\f\n"
        full_parts.append(sep)
        offset += len(sep)

    full_text = "".join(full_parts)

    # Precompute a per-character section map by scanning lines and remembering
    # the most recent detected header up to each character offset.
    # We build a list of (offset, section) transitions.
    section_transitions: List[Tuple[int, str]] = [(0, "body")]
    pos = 0
    for raw_line in full_text.splitlines(keepends=True):
        label = _detect_section(raw_line)
        if label is not None:
            section_transitions.append((pos, label))
        pos += len(raw_line)

    def _section_at(char_offset: int) -> str:
        section = "body"
        for t_off, t_label in section_transitions:
            if t_off <= char_offset:
                section = t_label
            else:
                break
        return section

    def _page_at(char_offset: int) -> int:
        page = page_starts[0][1] if page_starts else 1
        for p_off, p_num in page_starts:
            if p_off <= char_offset:
                page = p_num
            else:
                break
        return page

    spans = _split_text(full_text, settings.chunk_size, settings.chunk_overlap)

    chunks: List[Chunk] = []
    i = 0
    for start, end in spans:
        text = full_text[start:end].strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{paper_id}:{i}",
                paper_id=paper_id,
                section=_section_at(start),
                page=_page_at(start),
                text=text,
            )
        )
        i += 1

    return full_text, chunks


# --------------------------------------------------------------------------- #
# arXiv helpers
# --------------------------------------------------------------------------- #
# New-style ids: 2401.01234 or 2401.01234v2. Old-style: math.GT/0309136 or
# hep-th/9901001 (optionally with version).
_ARXIV_NEW_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)
_ARXIV_OLD_RE = re.compile(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.IGNORECASE)


def extract_arxiv_id(url_or_id: str) -> str:
    """Extract a bare arXiv id from a URL or id string.

    Handles bare ids, ``/abs/`` and ``/pdf/`` URLs, a trailing ``.pdf``, and
    version suffixes like ``v2`` (which are stripped). Returns the bare id, or
    the trimmed input if nothing matched.
    """
    s = (url_or_id or "").strip()
    # Strip a trailing .pdf extension for /pdf/<id>.pdf style URLs.
    if s.lower().endswith(".pdf"):
        s = s[:-4]

    m = _ARXIV_NEW_RE.search(s)
    if m:
        return m.group(1)

    m = _ARXIV_OLD_RE.search(s)
    if m:
        return m.group(1)

    return s


def download_arxiv(url_or_id: str, dest_dir: str) -> Tuple[str, str]:
    """Download an arXiv PDF and fetch its title.

    Downloads ``https://arxiv.org/pdf/<id>.pdf`` and writes it to
    ``dest_dir/<id>.pdf`` (slashes in old-style ids are replaced so the
    filename is filesystem-safe). The title is fetched from the arXiv Atom API;
    it is ``""`` if unavailable.

    Returns ``(local_pdf_path, title)``.
    """
    import requests

    aid = extract_arxiv_id(url_or_id)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    safe_name = aid.replace("/", "_")
    pdf_path = dest / f"{safe_name}.pdf"

    headers = {"User-Agent": "research-paper-agent/0.1 (+https://arxiv.org)"}

    pdf_url = f"https://arxiv.org/pdf/{aid}.pdf"
    resp = requests.get(pdf_url, headers=headers, timeout=60)
    resp.raise_for_status()
    pdf_path.write_bytes(resp.content)

    title = _fetch_arxiv_title(aid, headers)

    return str(pdf_path), title


def _fetch_arxiv_title(aid: str, headers: dict) -> str:
    """Best-effort fetch of a paper title from the arXiv Atom API.

    Returns ``""`` on any failure so ingestion can fall back to a heuristic
    title.
    """
    import requests

    try:
        api_url = f"http://export.arxiv.org/api/query?id_list={aid}"
        resp = requests.get(api_url, headers=headers, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return ""
        title_el = entry.find("atom:title", ns)
        if title_el is None or not title_el.text:
            return ""
        # Collapse internal whitespace/newlines the API inserts for wrapping.
        return re.sub(r"\s+", " ", title_el.text).strip()
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Title heuristic
# --------------------------------------------------------------------------- #
def guess_title(full_text: str) -> str:
    """Best-effort title guess from the opening lines of a paper.

    Scans the first non-empty lines of page 1, skipping obvious non-title
    boilerplate (arXiv stamps, page numbers, emails/URLs), and returns the
    first plausible line. Returns ``""`` when nothing suitable is found.
    """
    if not full_text:
        return ""

    # Restrict to the first page (before the form-feed separator we inserted).
    first_page = full_text.split("\f", 1)[0]

    skip_re = re.compile(
        r"(arxiv|doi|@|http|www\.|preprint|\bvol\.|copyright|\bpage\b|^\d+$)",
        re.IGNORECASE,
    )

    candidates: List[str] = []
    for raw in first_page.splitlines():
        line = raw.strip()
        if not line:
            if candidates:
                # A blank line after we already collected something marks the
                # end of the (possibly multi-line) title block.
                break
            continue
        if skip_re.search(line):
            if candidates:
                break
            continue
        # Reject lines that are clearly not titles.
        if len(line) < 3 or len(line) > 250:
            continue
        # A line that is all lowercase and long is likely body text.
        candidates.append(line)
        # Titles are usually one or two lines; stop after a reasonable amount.
        if len(candidates) >= 2:
            break

    if not candidates:
        return ""

    title = " ".join(candidates).strip()
    title = re.sub(r"\s+", " ", title)
    return title
