"""Lazy model loading keeps package imports cheap and tests completely offline."""
from __future__ import annotations

import hashlib
import math
import re
import threading
from typing import Protocol


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic TEST/DEMO embedder. Not semantically meaningful; never benchmark as BGE."""

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        output = []
        for text in texts:
            tokens = re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]", text.lower())
            vector = [0.0] * self.dimension
            for token in tokens:
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                index = value % self.dimension
                vector[index] += -1.0 if (value >> 20) & 1 else 1.0
            norm = math.sqrt(sum(x * x for x in vector)) or 1.0
            output.append([x / norm for x in vector])
        return output


class BGEEmbedder:
    """Dense representation of BAAI/bge-m3 using SentenceTransformers."""

    dimension = 1024

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
                        from sentence_transformers import SentenceTransformer
                    except ImportError as exc:
                        raise RuntimeError("Install models: uv sync --extra models") from exc
                    kwargs = {} if self.device == "auto" else {"device": self.device}
                    self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._get_model().encode(
            texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()
