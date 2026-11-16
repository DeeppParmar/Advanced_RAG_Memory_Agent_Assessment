import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment or .env file."""
    openai_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None

    qdrant_url: str = ":memory:"
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "advanced_rag_documents"

    child_chunk_size: int = 200
    parent_chunk_size: int = 1000
    chunk_overlap: int = 50

    hybrid_top_k: int = 20
    rerank_top_n: int = 5
    rerank_score_threshold: float = 0.3
    rrf_k: int = 60
    cohere_rerank_model: str = "rerank-english-v3.0"

    memory_storage_path: str = "data/user_memories.json"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
