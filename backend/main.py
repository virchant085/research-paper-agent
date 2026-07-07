"""FastAPI application entry point for the Research Paper Agent.

Wires up the API router, permissive CORS, and startup initialization
(directory creation and SQLite schema setup). Run directly with
``python -m backend.main`` (or the ``__main__`` guard below) to launch
a local development server via uvicorn.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.config import settings
from backend.services import db

app = FastAPI(title="Research Paper Agent", version="0.1.0")

# Permissive CORS so the standalone Streamlit frontend (any origin) can call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def on_startup() -> None:
    """Ensure data directories exist and the SQLite schema is initialized."""
    settings.ensure_dirs()
    db.init_db()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
