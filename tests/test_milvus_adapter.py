"""Verifies PyMilvus call contracts without requiring a live container."""
import sys
import types

from eeg_agent_rag.storage.milvus import MilvusVectorStore


class SearchRequest:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class Ranker:
    def __init__(self, k=60):
        self.k = k


class MockClient:
    def __init__(self, **kwargs):
        self.calls = []

    def hybrid_search(self, **kwargs):
        self.calls.append(("hybrid", kwargs))
        return [[{"id": "a:000001", "distance": 0.45,
                  "entity": {"document_id": "a", "content": "EEG", "year": 2025}}]]

    def search(self, **kwargs):
        self.calls.append(("single", kwargs))
        return [[]]

    def delete(self, **kwargs):
        self.calls.append(("delete", kwargs))


def test_bm25_query_is_raw_text_and_filter_is_applied(monkeypatch):
    fake = types.ModuleType("pymilvus")
    fake.MilvusClient = MockClient
    fake.AnnSearchRequest = SearchRequest
    fake.RRFRanker = Ranker
    monkeypatch.setitem(sys.modules, "pymilvus", fake)
    store = MilvusVectorStore("http://localhost:19530", "papers", 4)
    hits = store.search("depression EEG", [0.1] * 4, "hybrid", 4, year_from=2020)
    assert len(hits) == 1 and hits[0].document_id == "a"
    name, params = store.client.calls[-1]
    assert name == "hybrid"
    dense, sparse = params["reqs"]
    assert dense.kwargs["anns_field"] == "dense"
    assert sparse.kwargs["data"] == ["depression EEG"]
    assert sparse.kwargs["filter"] == "year >= 2020"
    assert params["ranker"].k == 60
    store.search("EEG", None, "sparse", 3, source='evil" or year > 0')
    assert store.client.calls[-1][1]["filter"] == 'source == "evil\\\" or year > 0"'
