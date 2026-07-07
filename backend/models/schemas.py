"""Pydantic models shared across the whole backend.

These are the authoritative data contracts. Services and the API both depend on
them, so keep field names stable.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Core domain models
# --------------------------------------------------------------------------- #
# The canonical five-type paper taxonomy (from the nature-skills shared core:
# research / methods / hypothesis / algorithmic / review).
PAPER_TYPES: tuple = ("research", "methods", "hypothesis", "algorithmic", "review")


class PaperCard(BaseModel):
    """The structured summary extracted once at ingestion time.

    Beyond the five analysis elements (problem/method/dataset/contribution/
    limitation), the card carries:

    - ``paper_type`` — one of the five canonical taxonomy types, so downstream
      comparison and review know the paper's argument structure.
    - ``key_terms`` — a small terminology ledger ("term — definition" entries)
      so every later output uses one name for one thing.
    - ``evidence`` — source-grounding anchors: short verbatim quotes from the
      paper backing each of the five elements. Quotes are verified against the
      source text at extraction time; unverifiable quotes are dropped.
    """

    paper_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    source: str = ""  # "upload:<filename>" or "arxiv:<id>"

    problem: str = ""
    method: str = ""
    dataset: str = ""
    contribution: str = ""
    limitation: str = ""

    paper_type: str = ""  # one of PAPER_TYPES, or "" if undetermined
    key_terms: List[str] = Field(default_factory=list)
    evidence: dict = Field(default_factory=dict)  # element -> verbatim quote


class Chunk(BaseModel):
    """A retrievable slice of a paper with provenance metadata."""

    chunk_id: str
    paper_id: str
    section: str = "body"
    page: int = 0
    text: str


class SearchHit(BaseModel):
    text: str
    paper_id: str
    section: str
    page: int
    score: float


# --------------------------------------------------------------------------- #
# API request / response models
# --------------------------------------------------------------------------- #
class ArxivRequest(BaseModel):
    url: str  # arXiv URL or bare id, e.g. "2401.01234" or "https://arxiv.org/abs/2401.01234"


class IngestResponse(BaseModel):
    paper_id: str
    title: str
    status: str  # "ok" | "error"
    message: str = ""


class QueryRequest(BaseModel):
    question: str
    paper_ids: Optional[List[str]] = None  # restrict retrieval to these papers


class ToolStep(BaseModel):
    tool: str
    arguments: dict
    result_preview: str = ""


class QueryResponse(BaseModel):
    answer: str
    steps: List[ToolStep] = Field(default_factory=list)


class CompareRequest(BaseModel):
    paper_ids: List[str]
    dimensions: Optional[List[str]] = None  # defaults to the five card fields


class TableRequest(BaseModel):
    paper_ids: Optional[List[str]] = None  # None => all papers


class ExportRequest(BaseModel):
    format: str = "markdown"  # "markdown" | "csv"
    paper_ids: Optional[List[str]] = None
    content: Optional[str] = None  # if provided, exported verbatim (e.g. a chat answer)


class ExportResponse(BaseModel):
    path: str
    format: str
