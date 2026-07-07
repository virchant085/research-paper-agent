# Research Paper Agent

[![CI](https://github.com/virchant085/research-paper-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/virchant085/research-paper-agent/actions/workflows/ci.yml)

> Personal Research Assistant Agent for **TEM / Robotics** papers.

Upload a paper PDF or paste an arXiv link, and the agent parses it, auto-extracts a
structured **problem / method / dataset / contribution / limitation** card, indexes it
for RAG retrieval, and answers questions by calling tools — search, summarize, compare,
build a literature table, and export to Markdown / CSV.

**Model-agnostic:** works with any mainstream LLM (OpenAI, Anthropic Claude, Google
Gemini, Mistral, Groq, DeepSeek, Cohere, local Ollama, …) — switch by changing one
`LLM_MODEL` string.

---

## Architecture

```
Streamlit UI  ──HTTP──▶  FastAPI  ──▶  Agent (function-calling loop)
                                          ├─ Ingestion pipeline (parse → chunk → extract → index)
                                          └─ Tools: search / summarize / compare / lit-table / export
                                                       │
                              Chroma (vectors) ◀───────┼───────▶ SQLite (PaperCards)
                                                       │
                                    Universal LLM layer (LiteLLM → any provider)
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
| LLM | **LiteLLM** — one interface to 100+ providers | swap models with one env var; no per-provider code |
| Vectors | Chroma (persistent, metadata-filtered) | filter by `paper_id` / `section` for free |
| Cards | SQLite | structured five-element store |
| Parsing | PyMuPDF + arXiv fetch | fast, no heavy deps |
| Frontend | Streamlit | minimal UI surface |
| Agent | hand-written function-calling loop (~120 LoC) | transparent, no framework overhead |

## Any model, one switch

Model selection is just a [LiteLLM](https://docs.litellm.ai/) model string. Set it in `.env`
and provide that provider's key under its standard variable name:

| `LLM_MODEL` | Needs |
|-------------|-------|
| `openai/gpt-4o-mini` | `OPENAI_API_KEY` |
| `gemini/gemini-2.0-flash` | `GEMINI_API_KEY` |
| `anthropic/claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` |
| `mistral/mistral-large-latest` | `MISTRAL_API_KEY` |
| `groq/llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| `deepseek/deepseek-chat` | `DEEPSEEK_API_KEY` |
| `ollama/llama3` | — (local) |

`EMBED_MODEL` works the same way (e.g. `openai/text-embedding-3-small`,
`gemini/text-embedding-004`). No code changes — the whole app is provider-agnostic.

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
| `GET` | `/health` | Liveness + active model |

## Quick start

```bash
python -m venv .venv && . .venv/Scripts/activate      # Windows (use bin/activate on *nix)
pip install -r requirements.txt
cp .env.example .env          # set LLM_MODEL / EMBED_MODEL + the matching API key

# Terminal 1 — API
uvicorn backend.main:app --reload

# Terminal 2 — UI
streamlit run frontend/app.py
```

### Docker

Run the API and UI together:

```bash
cp .env.example .env          # add your key(s)
docker compose up --build
# API → http://localhost:8000    UI → http://localhost:8501
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest                        # 87 tests, fully offline (LLM + network are mocked)
```

The suite runs with no API keys: the LLM is replaced by a deterministic fake, while Chroma
and SQLite run for real against a temp dir — so the vector-store and ingestion pipelines get
genuine end-to-end coverage. CI runs `ruff` + `pytest` on Python 3.11 and 3.12.

## Layout

```
backend/
  main.py            FastAPI entry (CORS, startup: ensure dirs + init db)
  config.py          settings (env-driven; model = LiteLLM string)
  api/routes.py      REST endpoints
  core/
    agent.py         bounded function-calling loop
    tools.py         the 5 tools + OpenAI-format schemas + registry
    ingestion.py     parse → chunk → extract card → index
  services/
    llm.py           universal LLM client (LiteLLM: chat / structured / embed)
    parser.py        PDF (PyMuPDF) + arXiv fetch, section-aware chunking
    vectorstore.py   Chroma wrapper (metadata-filtered search)
    db.py            SQLite PaperCard store
  models/schemas.py  Pydantic contracts (PaperCard, Chunk, request/response)
frontend/app.py      Streamlit UI (Papers / Chat / Library tabs)
tests/               offline pytest suite (87 tests)
Dockerfile           docker-compose.yml   .github/workflows/ci.yml
```

## Notes

- The LLM provider and vector store are each behind a single seam — swapping models is one
  env var, and Chroma could be replaced without touching the tools or agent.
- Client/SDK construction is lazy, so every module imports with **no** API keys set (which is
  what makes the fully-offline test suite possible).

## License

MIT — see [LICENSE](LICENSE).
