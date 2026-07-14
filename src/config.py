from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider selection -------------------------------------------------
    llm_provider: Literal["anthropic", "openai", "gemini", "mock"] = "anthropic"
    anthropic_model: str = "claude-sonnet-4-6"
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-2.5-flash"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None

    # --- Embedding provider selection -------------------------------------------
    embedding_provider: Literal["sentence-transformers", "openai", "local-tfidf"] = "local-tfidf"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Retrieval / graph behavior ---------------------------------------------
    top_k_per_subtask: int = 4
    max_writer_revisions: int = 2          
    max_retrieval_expansions: int = 1      
    min_citation_coverage: float = 0.8      

    # --- Paths --------------------------------------------------------------
    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    data_processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    faiss_index_path: Path = PROJECT_ROOT / "data" / "processed" / "index.faiss"
    chunks_path: Path = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
    prompts_dir: Path = PROJECT_ROOT / "src" / "prompts"

    # --- Chunking -----------------------------------------------------------
    chunk_size_chars: int = 1400
    chunk_overlap_chars: int = 200

    # --- MLOps ----------------------------------------------------------------
    mlflow_tracking_uri: str = f"file:{PROJECT_ROOT / 'mlruns'}"
    mlflow_experiment_name: str = "multi-agent-research-assistant"


settings = Settings()
