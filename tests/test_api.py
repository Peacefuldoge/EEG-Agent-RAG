from fastapi.testclient import TestClient

from eeg_agent_rag.api.app import create_app
from eeg_agent_rag.core.config import Settings


def test_health_and_upload_retrieve_answer_stream(components):
    config = Settings(
        _env_file=None, embedding_provider="hashing", embedding_dimension=64,
        api_key="dev-secret", milvus_collection="eeg_rag_test",
    )
    client = TestClient(create_app(config=config, services=components))
    assert client.get("/health").status_code == 200
    assert client.get("/v1/documents").status_code == 401
    headers = {"X-API-Key": "dev-secret"}
    uploaded = client.post(
        "/v1/documents", headers=headers,
        files={"file": ("example.md", b"EEG spectral analysis and depression." * 6)},
        data={"source": "pubmed", "year": "2025", "doi": "10.demo/only"},
    )
    assert uploaded.status_code == 200, uploaded.text
    doc_id = uploaded.json()["document_id"]
    assert client.get(f"/v1/documents/{doc_id}", headers=headers).status_code == 200
    payload = {"query": "EEG spectral analysis", "top_k": 3}
    search = client.post("/v1/retrieve", json=payload, headers=headers)
    assert search.status_code == 200 and search.json()["results"]
    answered = client.post("/v1/answer", json=payload, headers=headers)
    assert answered.status_code == 200
    assert answered.json()["cited_source_ids"] == ["S1"]
    sse = client.post("/v1/answer/stream", json=payload, headers=headers)
    assert "event: sources" in sse.text
    assert "event: token" in sse.text
    assert "event: done" in sse.text
    assert client.delete(f"/v1/documents/{doc_id}", headers=headers).status_code == 200


def test_upload_rejects_unsupported_and_oversized(components):
    config = Settings(
        _env_file=None, embedding_provider="hashing", embedding_dimension=64,
        upload_max_mb=1, milvus_collection="eeg_rag_test",
    )
    client = TestClient(create_app(config=config, services=components))
    bad = client.post("/v1/documents", files={"file": ("x.exe", b"abcd")})
    assert bad.status_code == 415
    huge = client.post("/v1/documents", files={"file": ("x.txt", b"a" * (1024 * 1024 + 1))})
    assert huge.status_code == 413
