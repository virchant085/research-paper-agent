"""The agent tools, their OpenAI-format schemas, and a name->function registry.

These functions are the concrete actions the agent can take. Each returns a
plain string (which is fed back to the LLM as a tool result) except ``export``,
which writes a file and returns the file path.

The public surface used by the rest of the codebase is:

* the six tool functions (``search_chunks``, ``summarize_section``,
  ``compare_papers``, ``generate_lit_table``, ``score_papers``, ``export``),
* ``TOOL_SCHEMAS`` -- OpenAI function-calling schemas for all tools,
* ``TOOL_REGISTRY`` -- maps a tool name to its callable.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

from backend.config import settings
from backend.models.schemas import PaperCard, SearchHit
from backend.services import db
from backend.services.llm import get_llm
from backend.services.vectorstore import get_store

# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #
_DEFAULT_DIMENSIONS = ["problem", "method", "dataset", "contribution", "limitation"]


def _escape_cell(value: str) -> str:
    """Make a value safe to drop into a single markdown table cell.

    Pipes would break the column layout and newlines would break the row, so we
    escape/flatten both.
    """
    text = "" if value is None else str(value)
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = text.replace("|", "\\|")
    return text.strip()


def _format_hit(hit: SearchHit) -> str:
    """Render one search hit as a single provenance-tagged line."""
    return (
        f"[{hit.paper_id} | {hit.section} p.{hit.page} | score={hit.score:.2f}] "
        f"{hit.text.strip()}"
    )


def _card_label(card: PaperCard) -> str:
    """Human-friendly column/section label for a paper card."""
    title = (card.title or "").strip()
    return title if title else card.paper_id


# --------------------------------------------------------------------------- #
# Tool 1: semantic retrieval
# --------------------------------------------------------------------------- #
def search_chunks(
    query: str,
    paper_id: str | None = None,
    section: str | None = None,
    k: int | None = None,
) -> str:
    """Semantic search over indexed paper chunks.

    Returns the matching chunks formatted as provenance-tagged lines so the
    agent can cite ``paper_id`` / ``section`` / page in its answer.
    """
    hits = get_store().search(query=query, paper_id=paper_id, section=section, k=k)
    if not hits:
        return "No matching chunks found."
    return "\n\n".join(_format_hit(h) for h in hits)


# --------------------------------------------------------------------------- #
# Tool 2: section summary
# --------------------------------------------------------------------------- #
def summarize_section(paper_id: str, section: str) -> str:
    """Summarize a single section of one paper.

    Retrieves the chunks belonging to ``section`` of ``paper_id`` and asks the
    LLM for a concise summary grounded only in that retrieved text.
    """
    hits = get_store().search(query=section, paper_id=paper_id, section=section, k=12)
    if not hits:
        return f"No content found for section '{section}' in paper {paper_id}."

    context = "\n\n".join(h.text.strip() for h in hits)
    system = (
        "You are a meticulous research assistant. Summarize the provided excerpts "
        "from a single section of an academic paper. Be concise and faithful: only "
        "use information present in the excerpts, and do not invent citations or "
        "results."
    )
    user = (
        f"Paper id: {paper_id}\n"
        f"Section: {section}\n\n"
        "Excerpts:\n"
        f"{context}\n\n"
        "Write a concise summary (3-6 sentences) of this section."
    )
    result = get_llm().chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    return (result.content or "").strip() or "No summary produced."


# --------------------------------------------------------------------------- #
# Tool 3: multi-paper comparison
# --------------------------------------------------------------------------- #
def compare_papers(paper_ids: list[str], dimensions: list[str] | None = None) -> str:
    """Compare several papers across the given dimensions.

    Produces a markdown table with one row per dimension and one column per
    paper. Missing papers are reported inline rather than raising.
    """
    if not paper_ids:
        return "No paper ids provided to compare."

    dims = dimensions or list(_DEFAULT_DIMENSIONS)

    cards: list[PaperCard | None] = [db.get_card(pid) for pid in paper_ids]

    missing = [pid for pid, card in zip(paper_ids, cards) if card is None]

    # Header row: the dimension label plus one column per paper.
    headers = ["Dimension"] + [
        _escape_cell(_card_label(card)) if card is not None else _escape_cell(pid)
        for pid, card in zip(paper_ids, cards)
    ]

    lines: list[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for dim in dims:
        row_cells = [_escape_cell(dim)]
        for card in cards:
            if card is None:
                row_cells.append("_not found_")
            else:
                value = getattr(card, dim, "")
                row_cells.append(_escape_cell(value) or "-")
        lines.append("| " + " | ".join(row_cells) + " |")

    table = "\n".join(lines)
    if missing:
        table += "\n\n_Note: no card found for: " + ", ".join(missing) + "._"
    return table


# --------------------------------------------------------------------------- #
# Tool 4: literature table
# --------------------------------------------------------------------------- #
_LIT_TABLE_HEADER = (
    "| Title | Year | Type | Problem | Method | Dataset | Contribution | Limitation |"
)
_LIT_TABLE_DIVIDER = "| --- | --- | --- | --- | --- | --- | --- | --- |"


def generate_lit_table(paper_ids: list[str] | None = None) -> str:
    """Build a markdown literature-review table from stored paper cards.

    With no ``paper_ids`` every stored card is included; otherwise only the
    requested (and existing) cards are used.
    """
    if paper_ids:
        cards = [c for c in (db.get_card(pid) for pid in paper_ids) if c is not None]
    else:
        cards = db.list_cards()

    if not cards:
        return "No papers in the library yet."

    lines = [_LIT_TABLE_HEADER, _LIT_TABLE_DIVIDER]
    for card in cards:
        year = "" if card.year is None else str(card.year)
        row = [
            _escape_cell(card.title),
            _escape_cell(year),
            _escape_cell(card.paper_type) or "-",
            _escape_cell(card.problem),
            _escape_cell(card.method),
            _escape_cell(card.dataset),
            _escape_cell(card.contribution),
            _escape_cell(card.limitation),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Tool 5: six-dimension relevance scoring
# --------------------------------------------------------------------------- #
# Weights follow the nature-literature-pipeline rubric: topic match dominates,
# and it is also a gate — a paper that misses the topic is rejected outright no
# matter how prestigious or well-made it is.
_SCORE_DIMENSIONS: tuple[tuple[str, int], ...] = (
    ("topic", 35),      # alignment with the stated research focus
    ("method", 20),     # methodological quality and applicability
    ("venue", 15),      # source/venue quality and credibility
    ("network", 10),    # relevance to tracked authors/institutions
    ("applied", 10),    # practical utility: protocols, datasets, benchmarks
    ("archival", 10),   # long-term reference value
)
_TOPIC_GATE = 10  # papers scoring below this on topic are auto-rejected


def score_papers(research_focus: str, paper_ids: list[str] | None = None) -> str:
    """Score stored papers against a research focus on six weighted dimensions.

    Asks the LLM to score each paper card, then enforces the rubric in code:
    every dimension is capped at its weight, the total is recalculated (never
    trusted), and papers under the topic gate are rejected regardless of their
    other scores. Returns a ranked markdown table.
    """
    if not research_focus or not research_focus.strip():
        return "No research focus provided to score against."

    if paper_ids:
        cards = [c for c in (db.get_card(pid) for pid in paper_ids) if c is not None]
    else:
        cards = db.list_cards()
    if not cards:
        return "No papers in the library yet."

    dims = ", ".join(f"{name} (max {cap})" for name, cap in _SCORE_DIMENSIONS)
    card_lines = []
    for c in cards:
        card_lines.append(
            f"- paper_id={c.paper_id} | title={c.title} | type={c.paper_type or '?'} | "
            f"problem={c.problem} | method={c.method} | dataset={c.dataset} | "
            f"contribution={c.contribution} | limitation={c.limitation}"
        )

    schema = {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string"},
                        **{
                            name: {"type": "integer"}
                            for name, _ in _SCORE_DIMENSIONS
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": ["paper_id", "topic"],
                },
            }
        },
        "required": ["scores"],
    }
    system = (
        "You are a rigorous literature triage assistant. Score each paper "
        "against the user's research focus on six dimensions. Be conservative: "
        "a famous paper that is off-topic must score low on topic. Base scores "
        "only on the provided card fields; do not reward prestige over fit."
    )
    user = (
        f"Research focus: {research_focus}\n\n"
        f"Dimensions and caps: {dims}\n\n"
        "Papers:\n" + "\n".join(card_lines) + "\n\n"
        "For each paper return its paper_id, an integer score per dimension "
        "(never exceeding that dimension's cap), and a one-sentence rationale."
    )

    try:
        data = get_llm().structured(system, user, schema)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the agent loop
        return f"Scoring failed: {exc}"

    raw_scores = data.get("scores") if isinstance(data, dict) else None
    if not isinstance(raw_scores, list):
        return "Scoring failed: model returned no scores."

    by_id = {c.paper_id: c for c in cards}
    ranked: list[tuple[int, dict, PaperCard]] = []
    rejected: list[tuple[PaperCard, int]] = []
    for entry in raw_scores:
        if not isinstance(entry, dict):
            continue
        card = by_id.get(str(entry.get("paper_id", "")))
        if card is None:
            continue
        # Enforce the rubric in code: cap each dimension, recalculate the total.
        capped = {
            name: max(0, min(_as_int(entry.get(name)), cap))
            for name, cap in _SCORE_DIMENSIONS
        }
        total = sum(capped.values())
        if capped["topic"] < _TOPIC_GATE:
            rejected.append((card, capped["topic"]))
            continue
        entry = {**entry, **capped, "total": total}
        ranked.append((total, entry, card))

    ranked.sort(key=lambda item: item[0], reverse=True)

    lines = [
        f"Ranking against research focus: {research_focus}",
        "",
        "| Rank | Title | Total | Topic | Method | Venue | Network | Applied | Archival | Rationale |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, (total, entry, card) in enumerate(ranked, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    _escape_cell(_card_label(card)),
                    str(total),
                    *[str(entry[name]) for name, _ in _SCORE_DIMENSIONS],
                    _escape_cell(str(entry.get("rationale", ""))),
                ]
            )
            + " |"
        )
    if rejected:
        lines.append("")
        lines.append(
            "_Rejected by topic gate (topic < "
            f"{_TOPIC_GATE}): "
            + ", ".join(f"{_card_label(c)} (topic={t})" for c, t in rejected)
            + "._"
        )
    if not ranked and not rejected:
        return "Scoring failed: no scored papers matched the library."
    return "\n".join(lines)


def _as_int(value: object) -> int:
    """Best-effort integer coercion for model-supplied scores (bad input -> 0)."""
    try:
        if isinstance(value, bool):
            return 0
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# Tool 6: export
# --------------------------------------------------------------------------- #
def _next_export_index(export_dir: Path, prefix: str, ext: str) -> int:
    """Pick a stable index by counting existing matching export files.

    We deliberately avoid time/date/random: the index is derived from the files
    already present in ``export_dir`` so filenames are reproducible.
    """
    existing = list(export_dir.glob(f"{prefix}*.{ext}"))
    return len(existing) + 1


def export(
    format: str = "markdown",
    paper_ids: list[str] | None = None,
    content: str | None = None,
) -> str:
    """Export content or the literature library to a file; return its path.

    * If ``content`` is provided, it is written verbatim as markdown.
    * Otherwise, ``format == "csv"`` writes a CSV of the (selected) paper cards
      and any other format writes the markdown literature table.
    """
    settings.ensure_dirs()
    export_dir = Path(settings.export_dir)

    if content is not None:
        ext = "md"
        prefix = "export_markdown_"
        idx = _next_export_index(export_dir, prefix, ext)
        path = export_dir / f"{prefix}{idx}.{ext}"
        path.write_text(content, encoding="utf-8")
        return str(path)

    if format == "csv":
        if paper_ids:
            cards = [
                c for c in (db.get_card(pid) for pid in paper_ids) if c is not None
            ]
        else:
            cards = db.list_cards()

        ext = "csv"
        prefix = "export_csv_"
        idx = _next_export_index(export_dir, prefix, ext)
        path = export_dir / f"{prefix}{idx}.{ext}"

        fieldnames = [
            "paper_id",
            "title",
            "authors",
            "year",
            "source",
            "paper_type",
            "problem",
            "method",
            "dataset",
            "contribution",
            "limitation",
            "key_terms",
        ]
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for card in cards:
                writer.writerow(
                    {
                        "paper_id": card.paper_id,
                        "title": card.title,
                        "authors": "; ".join(card.authors),
                        "year": "" if card.year is None else card.year,
                        "source": card.source,
                        "paper_type": card.paper_type,
                        "problem": card.problem,
                        "method": card.method,
                        "dataset": card.dataset,
                        "contribution": card.contribution,
                        "limitation": card.limitation,
                        "key_terms": "; ".join(card.key_terms),
                    }
                )
        return str(path)

    # Default: markdown literature table.
    table = generate_lit_table(paper_ids)
    ext = "md"
    prefix = "export_markdown_"
    idx = _next_export_index(export_dir, prefix, ext)
    path = export_dir / f"{prefix}{idx}.{ext}"
    path.write_text(table, encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------- #
# OpenAI function-calling schemas
# --------------------------------------------------------------------------- #
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_chunks",
            "description": (
                "Semantic search over indexed paper chunks. Returns the most "
                "relevant excerpts with provenance (paper_id, section, page, "
                "score). Use this to gather evidence before answering."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query.",
                    },
                    "paper_id": {
                        "type": "string",
                        "description": "Optional: restrict search to a single paper id.",
                    },
                    "section": {
                        "type": "string",
                        "description": (
                            "Optional: restrict search to a section name "
                            "(e.g. 'Method', 'Results')."
                        ),
                    },
                    "k": {
                        "type": "integer",
                        "description": "Optional: number of chunks to return.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_section",
            "description": (
                "Summarize one section of a single paper, grounded only in the "
                "retrieved text of that section."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "The paper id to summarize.",
                    },
                    "section": {
                        "type": "string",
                        "description": (
                            "Section name to summarize (e.g. 'Introduction', "
                            "'Method', 'Conclusion')."
                        ),
                    },
                },
                "required": ["paper_id", "section"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_papers",
            "description": (
                "Compare multiple papers across dimensions and return a markdown "
                "comparison table (dimensions as rows, papers as columns)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The paper ids to compare.",
                    },
                    "dimensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional dimensions to compare. Defaults to "
                            "problem, method, dataset, contribution, limitation."
                        ),
                    },
                },
                "required": ["paper_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_lit_table",
            "description": (
                "Generate a markdown literature-review table from stored paper "
                "cards. Omit paper_ids to include the whole library."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional: restrict the table to these paper ids. "
                            "Omit for all papers."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_papers",
            "description": (
                "Score and rank stored papers against a research focus on six "
                "weighted dimensions (topic match 35, methodological value 20, "
                "venue quality 15, network relevance 10, applied value 10, "
                "archival value 10). Papers that miss the topic are rejected "
                "outright. Use when the user asks which papers are most "
                "relevant/important for a topic, or to triage the library."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "research_focus": {
                        "type": "string",
                        "description": (
                            "The research question or focus to score papers "
                            "against — any field, e.g. 'sample-efficient "
                            "reinforcement learning' or 'CRISPR delivery "
                            "methods'."
                        ),
                    },
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional: restrict scoring to these paper ids. "
                            "Omit for the whole library."
                        ),
                    },
                },
                "required": ["research_focus"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "export",
            "description": (
                "Export content or the literature library to a file and return "
                "the file path. Provide 'content' to write markdown verbatim, or "
                "choose format 'csv'/'markdown' to export the paper cards."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "description": "Export format: 'markdown' or 'csv'.",
                    },
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional: restrict export to these paper ids. Omit "
                            "for all papers."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Optional: exact markdown content to export verbatim "
                            "(e.g. a chat answer)."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
]


# --------------------------------------------------------------------------- #
# Name -> callable registry (used by the agent to dispatch tool calls)
# --------------------------------------------------------------------------- #
TOOL_REGISTRY: dict[str, Callable] = {
    "search_chunks": search_chunks,
    "summarize_section": summarize_section,
    "compare_papers": compare_papers,
    "generate_lit_table": generate_lit_table,
    "score_papers": score_papers,
    "export": export,
}
