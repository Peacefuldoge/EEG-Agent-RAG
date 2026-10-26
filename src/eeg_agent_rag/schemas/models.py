from typing import Literal

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    document_id: str
    title: str
    source: str = "upload"
    doi: str = ""
    year: int = 0
    license: str = "unknown"
    status: str
    chunk_count: int = 0
    error: str = ""


class RetrievalHit(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    title: str = ""
    source: str = ""
    doi: str = ""
    license: str = "unknown"
    year: int = 0
    page: int = 0
    section: str = ""
    score: float = 0.0
    rerank_score: float | None = None
    citation: str = ""


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=2, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=30)
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    year_from: int | None = Field(default=None, ge=1800, le=2200)
    year_to: int | None = Field(default=None, ge=1800, le=2200)
    source: str | None = Field(default=None, max_length=200)
    rerank: bool = True


class RetrievalResponse(BaseModel):
    query: str
    results: list[RetrievalHit]


class AnswerRequest(RetrievalRequest):
    pass


class AnswerResponse(BaseModel):
    answer: str
    sources: list[RetrievalHit]
    cited_source_ids: list[str]
    citation_warning: str | None = None


class HealthResponse(BaseModel):
    status: str
    milvus: str
    collection: str
