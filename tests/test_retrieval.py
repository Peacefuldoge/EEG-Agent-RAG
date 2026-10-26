import pytest

from eeg_agent_rag.embeddings.providers import HashingEmbedder
from eeg_agent_rag.generation.citations import check_citations, source_context
from eeg_agent_rag.schemas.models import RetrievalHit, RetrievalRequest


def test_hashing_embedder_reproducible_and_normalized():
    model = HashingEmbedder(128)
    a, b = model.embed(["EEG data", "EEG data"])
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-8


def test_retrieval_with_filter(components):
    components.indexing.index_bytes("old.txt", b"EEG signal noise", source="A", year=2021)
    components.indexing.index_bytes("new.txt", b"EEG signal preprocessing", source="B", year=2025)
    results = components.retrieval.retrieve(RetrievalRequest(
        query="EEG signal", top_k=5, mode="hybrid", year_from=2024, source="B",
    ))
    assert len(results) == 1
    assert results[0].source == "B"


def test_invalid_year_interval(components):
    with pytest.raises(ValueError, match="year_from"):
        components.retrieval.retrieve(
            RetrievalRequest(query="EEG", year_from=2026, year_to=2020)
        )


def test_citation_validation():
    cited, warning = check_citations("Evidence [S2] and invalid [S9]", 2)
    assert cited == ["S2"]
    assert warning and "Unknown" in warning
    assert check_citations("No citations", 3)[1]


def test_context_assigns_sources():
    hits = [RetrievalHit(chunk_id="c1", document_id="d", title="Research",
                         content="Text", page=2)]
    text, new_hits = source_context(hits)
    assert "[S1]" in text and new_hits[0].citation == "S1"
