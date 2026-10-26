from __future__ import annotations

from dataclasses import dataclass

import pytest

from eeg_agent_rag.embeddings.providers import HashingEmbedder
from eeg_agent_rag.pipelines.answer import AnswerService
from eeg_agent_rag.pipelines.indexing import IndexingService
from eeg_agent_rag.pipelines.retrieval import RetrievalService
from eeg_agent_rag.retrieval.rerank import NoopReranker
from eeg_agent_rag.schemas.models import RetrievalHit
from eeg_agent_rag.storage.metadata import DocumentRegistry


class FakeMilvus:
    def __init__(self):
        self.rows = {}
        self.collection = "eeg_rag_test"

    def insert(self, rows):
        for row in rows:
            self.rows[row["id"]] = row

    def delete_document(self, document_id):
        self.rows = {k: row for k, row in self.rows.items() if row["document_id"] != document_id}

    def ping(self):
        return True

    def search(self, query, vector, mode, limit, year_from=None, year_to=None, source=None):
        hits = []
        for row in self.rows.values():
            if (year_from is not None and row["year"] < year_from) or (
                year_to is not None and row["year"] > year_to
            ):
                continue
            if source and row["source"] != source:
                continue
            score = float(len(set(query.lower().split()) & set(row["content"].lower().split())))
            if mode == "dense":
                score += 0.01
            hits.append(RetrievalHit(
                chunk_id=row["id"], document_id=row["document_id"],
                content=row["content"], title=row["title"], source=row["source"],
                doi=row["doi"], year=row["year"], page=row["page"], score=score,
            ))
        return sorted(hits, key=lambda h: h.score, reverse=True)[:limit]


class FakeGenerator:
    async def generate(self, system, user):
        assert "Untrusted reference excerpts:" in user
        return "The evidence is in the retrieved document [S1]."

    async def stream(self, system, user):
        for part in ["The evidence ", "is here [S1]."]:
            yield part


@dataclass
class FakeServices:
    vectors: object
    registry: object
    indexing: object
    retrieval: object
    answer: object


@pytest.fixture
def components(tmp_path):
    vectors = FakeMilvus()
    embedder = HashingEmbedder(64)
    registry = DocumentRegistry(tmp_path / "docs.sqlite3")
    indexing = IndexingService(vectors, registry, embedder, chunk_size=140, chunk_overlap=20)
    retrieval = RetrievalService(vectors, embedder, NoopReranker(), candidate_k=10)
    answer = AnswerService(retrieval, FakeGenerator())
    return FakeServices(vectors, registry, indexing, retrieval, answer)
