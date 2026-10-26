"""Pure functions for reproducible IR evaluation."""
from __future__ import annotations

import math


def recall_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(predicted[:k]) & relevant) / len(relevant)


def reciprocal_rank_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    for rank, chunk_id in enumerate(predicted[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(predicted[:k]) if item in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0
