from __future__ import annotations

from dataclasses import dataclass

from eeg_agent_rag.embeddings.providers import Embedder
from eeg_agent_rag.retrieval.rerank import Reranker
from eeg_agent_rag.schemas.models import RetrievalHit, RetrievalRequest


@dataclass
class RetrievalService:
    vectors: object
    embedder: Embedder
    reranker: Reranker
    candidate_k: int = 30

    def retrieve(self, request: RetrievalRequest) -> list[RetrievalHit]:
        if (
            request.year_from is not None and request.year_to is not None
            and request.year_from > request.year_to
        ):
            raise ValueError("year_from cannot exceed year_to")
        vector = self.embedder.embed([request.query])[0] if request.mode != "sparse" else None
        hits = self.vectors.search(
            query=request.query, vector=vector, mode=request.mode,
            limit=max(request.top_k, self.candidate_k if request.rerank else request.top_k),
            year_from=request.year_from, year_to=request.year_to, source=request.source,
        )
        if request.rerank:
            hits = self.reranker.rerank(request.query, hits)
        return hits[:request.top_k]
