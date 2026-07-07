"""Shared pytest fixtures.

The whole suite runs **offline** — no provider API keys, no network. The LLM is
replaced by a deterministic :class:`FakeLLM`, and all storage is redirected to a
per-test temp directory. Chroma and SQLite run for real against those temp dirs,
so the vector-store and ingestion paths get genuine end-to-end coverage.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

# Make the project root importable even without the pyproject pythonpath setting.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.services.llm as llm_mod  # noqa: E402
import backend.services.vectorstore as vs_mod  # noqa: E402
from backend.config import settings  # noqa: E402
from backend.services.llm import ChatResult  # noqa: E402


class FakeLLM:
    """Deterministic, offline stand-in for the universal LLM client.

    * ``chat`` pops pre-scripted :class:`ChatResult` objects from ``chat_script``
      (falling back to a plain "final answer" when the script is empty).
    * ``structured`` returns ``structured_result`` (a paper-card-shaped dict).
    * ``embed`` returns a stable 16-dim vector per text, so identical text embeds
      identically — which lets similarity search be asserted deterministically.
    """

    def __init__(self) -> None:
        self.chat_script: list[ChatResult] = []
        self.structured_result: dict = {
            "title": "Fake Paper",
            "authors": ["A. One", "B. Two"],
            "year": 2024,
            "problem": "the problem",
            "method": "the method",
            "dataset": "the dataset",
            "contribution": "the contribution",
            "limitation": "the limitation",
        }
        self.chat_calls: list[dict] = []
        self.embed_calls: list[list[str]] = []

    def chat(self, messages, tools=None, temperature=0.2) -> ChatResult:
        self.chat_calls.append({"messages": messages, "tools": tools})
        if self.chat_script:
            return self.chat_script.pop(0)
        return ChatResult(content="final answer", tool_calls=[])

    def structured(self, system, user, schema) -> dict:
        return dict(self.structured_result)

    def embed(self, texts) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [self._vec(t) for t in texts]

    @staticmethod
    def _vec(text: str) -> list[float]:
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:16]]


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect all storage to a temp dir and reset process-wide singletons."""
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "chroma_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "sqlite_path", str(tmp_path / "papers.db"))
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "export_dir", str(tmp_path / "exports"))
    settings.ensure_dirs()

    # Rebuild the vector-store/LLM singletons against the temp dirs / fake client.
    monkeypatch.setattr(llm_mod, "_CLIENT", None, raising=False)
    monkeypatch.setattr(vs_mod, "_store", None, raising=False)

    from backend.services import db

    db.init_db()
    yield


@pytest.fixture
def fake_llm(monkeypatch):
    """Install a :class:`FakeLLM` as the process-wide LLM singleton."""
    fake = FakeLLM()
    monkeypatch.setattr(llm_mod, "_CLIENT", fake, raising=False)
    return fake
