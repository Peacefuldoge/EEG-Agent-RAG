"""Runtime configuration, loaded once from environment/.env."""
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    milvus_collection: str = "eeg_rag_v1"
    sqlite_path: Path = Path("data/rag.sqlite3")
    upload_max_mb: int = Field(default=20, ge=1, le=200)
    chunk_size: int = Field(default=1000, ge=128, le=3000)
    chunk_overlap: int = Field(default=150, ge=0)
    batch_size: int = Field(default=16, ge=1, le=128)

    embedding_provider: Literal["bge", "hashing"] = "bge"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = Field(default=1024, ge=8)
    embedding_device: str = "auto"
    rerank_provider: Literal["none", "cross-encoder"] = "none"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = "/home/zzz/models/Qwen3-8B"
    llm_max_tokens: int = Field(default=768, ge=64, le=8192)
    llm_timeout: float = Field(default=120, ge=5, le=600)
    candidate_k: int = Field(default=30, ge=5, le=200)
    api_key: str = ""

    @model_validator(mode="after")
    def validate_config(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.embedding_provider == "bge" and self.embedding_dimension != 1024:
            raise ValueError("BAAI/bge-m3 requires embedding_dimension=1024")
        return self
