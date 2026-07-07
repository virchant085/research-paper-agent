# Research Paper Agent

> Personal Research Assistant Agent for **TEM / Robotics** papers.

Upload a paper PDF or paste an arXiv link, and the agent parses it, auto-extracts a
structured **problem / method / dataset / contribution / limitation** card, indexes it
for RAG retrieval, and answers questions by calling tools — search, summarize, compare,
build a literature table, and export to Markdown / CSV.

**Status:** 🚧 scaffolding — foundations landed, core modules in progress.

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

| Layer | Choice |
|-------|--------|
| API | FastAPI |
| LLM | Gemini (default) or OpenAI, behind one abstraction |
| Vectors | Chroma (persistent, metadata-filtered) |
| Cards | SQLite |
| Parsing | PyMuPDF + arXiv fetch |
| Frontend | Streamlit |

## Tools the agent can call

| Tool | What it does |
|------|--------------|
| `search_chunks` | Vector search over paper chunks, filterable by paper / section |
| `summarize_section` | Summarize one section of a paper |
| `compare_papers` | Side-by-side comparison across the five card dimensions |
| `generate_lit_table` | Literature-review table over selected papers |
| `export` | Write Markdown or CSV to disk |

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # then add your GEMINI_API_KEY or OPENAI_API_KEY

# Terminal 1 — API
uvicorn backend.main:app --reload

# Terminal 2 — UI
streamlit run frontend/app.py
```

## Layout

```
backend/
  main.py            FastAPI entry
  config.py          settings (env-driven)
  api/routes.py      REST endpoints
  core/
    agent.py         function-calling loop
    tools.py         the 5 tools + registry
    ingestion.py     parse → chunk → extract → index
  services/
    llm.py           Gemini / OpenAI abstraction
    parser.py        PDF + arXiv parsing
    vectorstore.py   Chroma wrapper
    db.py            SQLite PaperCard store
  models/schemas.py  Pydantic contracts
frontend/app.py      Streamlit UI
```

## License

MIT
