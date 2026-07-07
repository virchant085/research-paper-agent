"""Central configuration. All settings come from environment / .env file.

Import `settings` anywhere; it is a singleton.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider selection ---
    llm_provider: str = "gemini"  # "gemini" | "openai"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_embed_model: str = "text-embedding-004"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

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

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.chroma_dir, self.upload_dir, self.export_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()
