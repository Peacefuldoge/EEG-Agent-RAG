"""Grounded retrieval evaluation from manually curated JSONL qrels.

Format per line: {"query": "...", "relevant_chunk_ids": ["docid:000001"]}
Usage: uv run python scripts/evaluate_retrieval.py --dataset data/sample/eeg_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from eeg_agent_rag.core.config import Settings
from eeg_agent_rag.core.container import build_services
from eeg_agent_rag.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank_at_k
from eeg_agent_rag.schemas.models import RetrievalRequest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=["dense", "sparse", "hybrid"], default="hybrid")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/eval_results.json"))
    args = parser.parse_args()
    if args.k < 1 or args.k > 30:
        parser.error("--k must be 1..30")
    records = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        parser.error("Evaluation set is empty")
    services = build_services(Settings())
    details = []
    for item in records:
        if not item.get("relevant_chunk_ids"):
            parser.error("Each question must have at least one manually validated relevant chunk")
        request = RetrievalRequest(
            query=item["query"], top_k=args.k, mode=args.mode, rerank=args.rerank
        )
        started = time.perf_counter()
        hits = services.retrieval.retrieve(request)
        latency = (time.perf_counter() - started) * 1000
        retrieved = [h.chunk_id for h in hits]
        relevant = set(item["relevant_chunk_ids"])
        details.append({
            "query": item["query"], "retrieved": retrieved,
            "relevant": sorted(relevant), "latency_ms": latency,
            "recall_at_k": recall_at_k(retrieved, relevant, args.k),
            "mrr_at_k": reciprocal_rank_at_k(retrieved, relevant, args.k),
            "ndcg_at_k": ndcg_at_k(retrieved, relevant, args.k),
        })
    latencies = sorted(x["latency_ms"] for x in details)
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
    summary = {
        "dataset": str(args.dataset), "queries": len(details),
        "mode": args.mode, "rerank": args.rerank, "k": args.k,
        "recall_at_k": statistics.mean(x["recall_at_k"] for x in details),
        "mrr_at_k": statistics.mean(x["mrr_at_k"] for x in details),
        "ndcg_at_k": statistics.mean(x["ndcg_at_k"] for x in details),
        "latency_p50_ms": statistics.median(latencies), "latency_p95_ms": p95,
        "embedding_model": Settings().embedding_model,
        "reranker_model": Settings().rerank_model if args.rerank else "none",
        "collection": Settings().milvus_collection,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Details written to {args.output}")


if __name__ == "__main__":
    main()
