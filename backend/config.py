"""Central configuration. All settings come from environment / .env file.

Import `settings` anywhere; it is a singleton.

Model selection is provider-agnostic: `llm_model` / `embed_model` are LiteLLM
model strings (e.g. ``openai/gpt-4o-mini``, ``gemini/gemini-2.0-flash``,
``anthropic/claude-3-5-sonnet-20241022``, ``groq/llama-3.3-70b-versatile``,
``ollama/llama3``). Whichever providers you use, put their API keys in the
environment / .env under the standard variable names (``OPENAI_API_KEY``,
``GEMINI_API_KEY``, ``ANTHROPIC_API_KEY``, ``MISTRAL_API_KEY``, ...); LiteLLM
reads them from there.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Populate os.environ from a local .env so LiteLLM (and any provider SDK) can see
# provider API keys by their standard names. Done before Settings is built.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Model selection (LiteLLM model strings; any mainstream provider) ---
    llm_model: str = "gemini/gemini-2.0-flash"
    embed_model: str = "gemini/text-embedding-004"

    # --- Storage locations ---
    data_dir: str = "./data"
    chroma_dir: str = "./data/chroma"
    sqlite_path: str = "./data/papers.db"
    upload_dir: str = "./data/uploads"
    export_dir: str = "./data/exports"

    # --- RAG / chunking knobs ---
    chunk_size: int = 1200       # characters per chunk (approx)
    chunk_overlap: int = 200
    top_k: int = 6               # default chunks returned by search
    agent_max_steps: int = 6     # max tool-calling iterations

    @property
    def provider(self) -> str:
        """The provider prefix of ``llm_model`` (e.g. 'openai', 'gemini'), best-effort."""
        return self.llm_model.split("/", 1)[0] if "/" in self.llm_model else self.llm_model

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.chroma_dir, self.upload_dir, self.export_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()
