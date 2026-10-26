from __future__ import annotations

import threading
from typing import Protocol

from eeg_agent_rag.schemas.models import RetrievalHit


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]: ...


class NoopReranker:
    def rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        return hits


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import CrossEncoder
                    except ImportError as exc:
                        raise RuntimeError("Install models: uv sync --extra models") from exc
                    kwargs = {} if self.device == "auto" else {"device": self.device}
                    self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model

    def rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        if not hits:
            return []
        scores = self._get_model().predict([(query, hit.content) for hit in hits])
        scored = [
            hit.model_copy(update={"rerank_score": float(score)})
            for hit, score in zip(hits, scores, strict=True)
        ]
        return sorted(scored, key=lambda hit: hit.rerank_score or 0.0, reverse=True)
