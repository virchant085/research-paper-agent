# Research Paper Agent

> Personal Research Assistant Agent for **TEM / Robotics** papers.

Upload a paper PDF or paste an arXiv link, and the agent parses it, auto-extracts a
structured **problem / method / dataset / contribution / limitation** card, indexes it
for RAG retrieval, and answers questions by calling tools — search, summarize, compare,
build a literature table, and export to Markdown / CSV.

**Status:** ✅ Core implemented and verified. The full backend (FastAPI) and frontend
(Streamlit) are in place; the import graph, app boot, SQLite store, PDF/arXiv parsing,
literature table and export are covered by smoke tests. LLM-dependent paths (embedding,
card extraction, the agent loop) run once you add a `GEMINI_API_KEY` or `OPENAI_API_KEY`.

---

## Architecture

```
Streamlit UI  ──HTTP──▶  FastAPI  ──▶  Agent (function-calling loop)
                                          ├─ Ingestion pipeline (parse → chunk → extract → index)
                                          └─ Tools: search / summarize / compare / lit-table / export
                                                       │
                              Chroma (vectors) ◀───────┼───────▶ SQLite (PaperCards)
                                                       │
                                          LLM provider abstraction (Gemini / OpenAI)
```

Two data flows:

- **Ingestion** — `PDF / arXiv → parse → section-aware chunks → embed into Chroma`, and
  in one shot the LLM extracts a structured `PaperCard` (the five elements) stored in SQLite.
- **Query** — a bounded function-calling loop hands the LLM the tools, executes whichever
  it calls, feeds results back, and returns a cited answer plus the tool trace.

The key design choice: `compare_papers` and `generate_lit_table` read the pre-extracted
**PaperCards**, not fresh RAG — so comparison and table-building are cheap, deterministic
structured operations rather than repeated LLM retrieval.

| Layer | Choice | Why |
|-------|--------|-----|
| API | FastAPI | async, typed, tiny |
| LLM | Gemini (default) or OpenAI, behind one abstraction | swap providers with one env var |
| Vectors | Chroma (persistent, metadata-filtered) | filter by `paper_id` / `section` for free |
| Cards | SQLite | structured five-element store |
| Parsing | PyMuPDF + arXiv fetch | fast, no heavy deps |
| Frontend | Streamlit | minimal UI surface |
| Agent | hand-written function-calling loop (~120 LoC) | transparent, no framework overhead |

## Tools the agent can call

| Tool | What it does |
|------|--------------|
| `search_chunks` | Vector search over paper chunks, filterable by paper / section |
| `summarize_section` | Summarize one section of a paper, grounded in its retrieved text |
| `compare_papers` | Markdown comparison table across the five card dimensions |
| `generate_lit_table` | Literature-review table over selected papers |
| `export` | Write Markdown or CSV to disk, return the path |

## REST API

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/papers/upload` | Upload a PDF (multipart), ingest it |
| `POST` | `/papers/arxiv` | Ingest by arXiv URL or id |
| `GET` | `/papers` | List all extracted paper cards |
| `GET` | `/papers/{id}` | One paper card (404 if absent) |
| `DELETE` | `/papers/{id}` | Remove a paper (cards + vectors) |
| `POST` | `/query` | Ask a question → agent answer + tool trace |
| `POST` | `/compare` | Compare papers → markdown |
| `POST` | `/table` | Literature table → markdown |
| `POST` | `/export` | Export markdown/CSV → file path |
| `GET` | `/health` | Liveness + active provider |

## Quick start

```bash
python -m venv .venv && . .venv/Scripts/activate      # Windows
pip install -r requirements.txt
cp .env.example .env          # then add GEMINI_API_KEY or OPENAI_API_KEY

# Terminal 1 — API
uvicorn backend.main:app --reload

# Terminal 2 — UI
streamlit run frontend/app.py
```

Switch providers by setting `LLM_PROVIDER=gemini` or `LLM_PROVIDER=openai` in `.env`.

## Layout

```
backend/
  main.py            FastAPI entry (CORS, startup: ensure dirs + init db)
  config.py          settings (env-driven, pydantic-settings)
  api/routes.py      REST endpoints
  core/
    agent.py         bounded function-calling loop
    tools.py         the 5 tools + OpenAI-format schemas + registry
    ingestion.py     parse → chunk → extract card → index
  services/
    llm.py           Gemini / OpenAI abstraction (chat / structured / embed)
    parser.py        PDF (PyMuPDF) + arXiv fetch, section-aware chunking
    vectorstore.py   Chroma wrapper (metadata-filtered search)
    db.py            SQLite PaperCard store
  models/schemas.py  Pydantic contracts (PaperCard, Chunk, request/response)
frontend/app.py      Streamlit UI (Papers / Chat / Library tabs)
```

## Notes

- LLM provider and vector store are each behind a single seam — swapping Gemini↔OpenAI is
  one env var, and Chroma could be replaced without touching the tools or agent.
- Client/SDK construction is lazy, so every module imports with **no** API keys set
  (which is what makes the non-LLM test suite possible).

## License

MIT — see [LICENSE](LICENSE).
