"""Composition root: inject mocked providers here during testing."""
from __future__ import annotations

from dataclasses import dataclass

from eeg_agent_rag.core.config import Settings
from eeg_agent_rag.embeddings.providers import BGEEmbedder, HashingEmbedder
from eeg_agent_rag.generation.llm import VLLMGenerator
from eeg_agent_rag.pipelines.answer import AnswerService
from eeg_agent_rag.pipelines.indexing import IndexingService
from eeg_agent_rag.pipelines.retrieval import RetrievalService
from eeg_agent_rag.retrieval.rerank import CrossEncoderReranker, NoopReranker
from eeg_agent_rag.storage.metadata import DocumentRegistry
from eeg_agent_rag.storage.milvus import MilvusVectorStore


@dataclass
class Services:
    vectors: MilvusVectorStore
    registry: DocumentRegistry
    indexing: IndexingService
    retrieval: RetrievalService
    answer: AnswerService


def build_services(config: Settings) -> Services:
    embedder = (
        HashingEmbedder(config.embedding_dimension)
        if config.embedding_provider == "hashing"
        else BGEEmbedder(config.embedding_model, config.embedding_device)
    )
    reranker = (
        NoopReranker()
        if config.rerank_provider == "none"
        else CrossEncoderReranker(config.rerank_model, config.embedding_device)
    )
    vectors = MilvusVectorStore(
        config.milvus_uri, config.milvus_collection,
        embedder.dimension, config.milvus_token,
    )
    vectors.ensure_collection()
    registry = DocumentRegistry(config.sqlite_path)
    indexing = IndexingService(
        vectors=vectors, registry=registry, embedder=embedder,
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap,
        batch_size=config.batch_size,
    )
    retrieval = RetrievalService(
        vectors=vectors, embedder=embedder,
        reranker=reranker, candidate_k=config.candidate_k,
    )
    generator = VLLMGenerator(
        base_url=config.llm_base_url, api_key=config.llm_api_key,
        model=config.llm_model, timeout=config.llm_timeout,
        max_tokens=config.llm_max_tokens,
    )
    return Services(
        vectors=vectors, registry=registry, indexing=indexing,
        retrieval=retrieval, answer=AnswerService(retrieval, generator),
    )
