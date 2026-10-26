"""LangGraph-agnostic adapter. Expose lookup_evidence as an Agent tool in EEG-Agent-Core."""
from __future__ import annotations

import os

import httpx


class RAGToolClient:
    def __init__(self, base_url: str = "http://localhost:8010", api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("RAG_API_KEY", "")

    def retrieve(self, query: str, top_k: int = 5) -> dict:
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        with httpx.Client(timeout=90) as client:
            resp = client.post(
                f"{self.base_url}/v1/retrieve",
                headers=headers,
                json={"query": query, "top_k": top_k, "mode": "hybrid"},
            )
            resp.raise_for_status()
            return resp.json()


def lookup_eeg_evidence(question: str) -> dict:
    """Tool signature for an Agent's dynamic tool registry."""
    return RAGToolClient().retrieve(question)


if __name__ == "__main__":
    print(lookup_eeg_evidence("Which EEG preprocessing parameters should be reported?"))
