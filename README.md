# Research Paper Agent

[![CI](https://github.com/virchant085/research-paper-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/virchant085/research-paper-agent/actions/workflows/ci.yml)

> A general-purpose AI research assistant that reads your papers — **in any field** —
> and answers grounded, source-cited questions about them, with any LLM you choose.

Drop in a paper (PDF upload or arXiv link) and the agent parses it, then auto-extracts a
structured **problem / method / dataset / contribution / limitation** card — classified by a
five-type paper taxonomy, carrying a terminology ledger and **source-verified evidence
quotes** (every quote is machine-checked against the paper; unverifiable ones are dropped).
Papers are chunked and embedded for RAG, and you query the library through a tool-using agent
that searches, summarizes, compares, ranks by relevance, builds literature tables, and
exports to Markdown / CSV — always citing the paper and section it drew from.

Three things set it apart:

- **Model-agnostic** — works with any mainstream LLM (OpenAI, Anthropic Claude, Google
  Gemini, Mistral, Groq, DeepSeek, Cohere, local Ollama, …). Switch by changing one
  `LLM_MODEL` string; no per-provider code.
- **Grounded by design** — both extraction and answers must cite the source or say it isn't
  stated, never guess. Evidence quotes are verified against the original text.
- **Production-shaped** — typed FastAPI backend, 108 offline tests, and CI on Python 3.11 / 3.12;
  one-command Docker for the API + UI.

---

## Architecture

```
Streamlit UI  ──HTTP──▶  FastAPI  ──▶  Agent (function-calling loop)
                                          ├─ Ingestion pipeline (parse → chunk → extract → index)
                                          └─ Tools: search / summarize / compare / review / score / lit-table / export
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
| `compare_papers` | Side-by-side table across card dimensions, plus an optional AI cross-analysis (similarities / differences / what each is best for) |
| `generate_lit_table` | Literature-review table over selected papers |
| `literature_review` | Synthesize themes, methods, consensus, disagreements & gaps across many papers |
| `score_papers` | Rank papers against a research focus on six weighted dimensions |
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
| `POST` | `/compare` | Compare papers → markdown (optional AI cross-analysis) |
| `POST` | `/table` | Literature table → markdown |
| `POST` | `/review` | Multi-paper literature review (synthesis) → markdown |
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
pytest                        # 108 tests, fully offline (LLM + network are mocked)
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
    tools.py         the 7 tools + OpenAI-format schemas + registry
    ingestion.py     parse → chunk → extract card → index
  services/
    llm.py           universal LLM client (LiteLLM: chat / structured / embed)
    parser.py        PDF (PyMuPDF) + arXiv fetch, section-aware chunking
    vectorstore.py   Chroma wrapper (metadata-filtered search)
    db.py            SQLite PaperCard store
  models/schemas.py  Pydantic contracts (PaperCard, Chunk, request/response)
frontend/app.py      Streamlit UI (Papers / Chat / Library tabs)
tests/               offline pytest suite (108 tests)
Dockerfile           docker-compose.yml   .github/workflows/ci.yml
```

## Design influences: nature-skills

The analysis methodology is adapted from
[nature-skills](https://github.com/Yuan1z0825/nature-skills) (Apache-2.0), a research-skill
library for AI agents. Four of its ideas are built into this codebase:

1. **Source grounding** (from `nature-reader`'s grounding rules) — extraction asks the model
   for short *verbatim* evidence quotes backing each card field, then **machine-verifies**
   every quote against the source text and drops any that don't match: grounding that cannot
   be verified is not grounding. The agent's system prompt enforces the same discipline at
   query time — cite provenance inline, and say plainly when the source doesn't state
   something instead of guessing.
2. **Five-type paper taxonomy** (from the shared core) — every paper is classified as
   `research / methods / hypothesis / algorithmic / review`, which tells downstream
   comparison and triage what the paper's argument structure is.
3. **Terminology ledger** — extraction records the paper's own canonical terms
   ("one name for one thing"), and the agent is instructed never to coin synonyms for the
   authors' concepts.
4. **Six-dimension scoring** (from `nature-literature-pipeline`) — `score_papers` ranks the
   library against a research focus with weighted dimensions (topic 35, method 20, venue 15,
   network 10, applied 10, archival 10), enforcing the rubric *in code*: per-dimension caps,
   recalculated totals, and a topic gate that rejects off-topic papers outright.

The extraction prompt is also organized around the reader's question sequence (relevance →
novelty → trust → reuse → boundaries), with an explicit instruction not to skip limitations —
the most commonly skipped question.

## Notes

- The LLM provider and vector store are each behind a single seam — swapping models is one
  env var, and Chroma could be replaced without touching the tools or agent.
- Client/SDK construction is lazy, so every module imports with **no** API keys set (which is
  what makes the fully-offline test suite possible).

## License

MIT — see [LICENSE](LICENSE).
