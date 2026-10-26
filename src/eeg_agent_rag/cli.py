"""Lightweight CLI for WSL, CI and scripting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_agent_rag.core.config import Settings
from eeg_agent_rag.core.container import build_services
from eeg_agent_rag.schemas.models import RetrievalRequest


def main():
    parser = argparse.ArgumentParser(prog="eeg-rag")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create and load Milvus collection")

    ingest = sub.add_parser("ingest", help="Index a local PDF, TXT or Markdown file")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--source", default="local")
    ingest.add_argument("--doi", default="")
    ingest.add_argument("--year", type=int, default=0)
    ingest.add_argument("--license", default="unknown")
    ingest.add_argument("--force", action="store_true")

    query = sub.add_parser("query", help="Search indexed evidence")
    query.add_argument("text")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid")

    args = parser.parse_args()
    services = build_services(Settings())
    if args.command == "init":
        print("Milvus collection ready")
    elif args.command == "ingest":
        content = args.path.read_bytes()
        item = services.indexing.index_bytes(
            args.path.name, content, args.source, args.doi, args.year, args.license,
            args.force,
        )
        print(item.model_dump_json(indent=2))
    else:
        result = services.retrieval.retrieve(
            RetrievalRequest(query=args.text, top_k=args.top_k, mode=args.mode)
        )
        print(json.dumps([h.model_dump() for h in result], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
