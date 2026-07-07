"""Chroma-backed vector store for paper chunks.

Persists chunk embeddings in a local Chroma collection and exposes similarity
search with optional filtering by paper and section. Embeddings are produced by
the configured LLM provider (:func:`backend.services.llm.get_llm`) so that the
query and document embedding spaces stay consistent.

The Chroma client is constructed lazily (on first :class:`VectorStore`
instantiation via :func:`get_store`), so importing this module has no side
effects and requires no API keys.
"""
from __future__ import annotations

from typing import Optional

import chromadb

from backend.config import settings
from backend.models.schemas import Chunk, SearchHit
from backend.services.llm import get_llm

_COLLECTION_NAME = "papers"


class VectorStore:
    """Thin wrapper around a persistent Chroma collection.

    Documents are chunk texts; metadata carries the ``paper_id``, ``section``
    and ``page`` provenance used for filtering and for building
    :class:`~backend.models.schemas.SearchHit` results.
    """

    def __init__(self) -> None:
        settings.ensure_dirs()
        self._client = chromadb.PersistentClient(path=settings.chroma_dir)
        self._collection = self._client.get_or_create_collection(
            _COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and upsert ``chunks`` into the collection.

        Chunks whose text is empty or whitespace-only are skipped. Embeddings
        are computed in a single batch call to the provider for efficiency.
        """
        # Filter out empties up front so ids/embeddings/metadatas stay aligned.
        valid = [c for c in chunks if c.text and c.text.strip()]
        if not valid:
            return

        embeddings = get_llm().embed([c.text for c in valid])

        ids = [c.chunk_id for c in valid]
        metadatas = [
            {"paper_id": c.paper_id, "section": c.section, "page": c.page}
            for c in valid
        ]
        documents = [c.text for c in valid]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        paper_id: Optional[str] = None,
        section: Optional[str] = None,
        k: Optional[int] = None,
    ) -> list[SearchHit]:
        """Return the ``k`` most similar chunks to ``query``.

        Optional ``paper_id`` / ``section`` narrow the search via a Chroma
        ``where`` filter. ``score`` is ``1 - distance`` (cosine similarity).
        """
        n_results = k or settings.top_k
        where = self._build_where(paper_id, section)

        query_embedding = get_llm().embed([query])[0]

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )

        return self._to_hits(result)

    def delete_paper(self, paper_id: str) -> None:
        """Remove every chunk belonging to ``paper_id``."""
        self._collection.delete(where={"paper_id": paper_id})

    def sections_for(self, paper_id: str) -> list[str]:
        """Return the distinct section names present for ``paper_id``.

        Order of first appearance is preserved.
        """
        result = self._collection.get(where={"paper_id": paper_id})
        metadatas = result.get("metadatas") or []

        seen: list[str] = []
        for meta in metadatas:
            section = (meta or {}).get("section")
            if section and section not in seen:
                seen.append(section)
        return seen

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_where(
        paper_id: Optional[str], section: Optional[str]
    ) -> Optional[dict]:
        """Assemble a Chroma ``where`` clause from optional filters.

        Chroma requires a single ``{key: value}`` map for one condition and an
        explicit ``$and`` when combining several. Returns ``None`` when there is
        nothing to filter on.
        """
        clauses: list[dict] = []
        if paper_id is not None:
            clauses.append({"paper_id": paper_id})
        if section is not None:
            clauses.append({"section": section})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _to_hits(result: dict) -> list[SearchHit]:
        """Convert a Chroma ``query`` result into :class:`SearchHit` objects."""
        # Query results are nested one level deep (one list per query).
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        hits: list[SearchHit] = []
        for text, meta, distance in zip(docs, metas, dists):
            meta = meta or {}
            hits.append(
                SearchHit(
                    text=text or "",
                    paper_id=meta.get("paper_id", ""),
                    section=meta.get("section", "body"),
                    page=int(meta.get("page", 0) or 0),
                    score=1.0 - float(distance),
                )
            )
        return hits


# --------------------------------------------------------------------------- #
# Singleton accessor
# --------------------------------------------------------------------------- #
_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    """Return the process-wide :class:`VectorStore`, building it on first use."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
