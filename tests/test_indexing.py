import pytest


def test_index_idempotent_and_force(components):
    service = components.indexing
    content = b"EEG preprocessing includes a bandpass filter. " * 20
    first = service.index_bytes("a.md", content, source="lab", year=2025)
    assert first.status == "completed"
    assert first.chunk_count > 1
    before = len(components.vectors.rows)
    again = service.index_bytes("b.md", content)
    assert again.document_id == first.document_id
    assert len(components.vectors.rows) == before
    forced = service.index_bytes("a.md", content, source="lab", year=2026, force=True)
    assert forced.year == 2026
    assert len(components.vectors.rows) == before
    assert all(row["year"] == 2026 for row in components.vectors.rows.values())


def test_invalid_document_marks_failed(components):
    with pytest.raises(ValueError):
        components.indexing.index_bytes("bad.txt", b"\xff")
    docs = components.registry.list()
    assert len(docs) == 1
    assert docs[0].status == "failed"


def test_delete_document_cleans_index(components):
    info = components.indexing.index_bytes("a.txt", b"EEG studies compare bands")
    assert components.indexing.delete(info.document_id)
    assert not components.vectors.rows
    assert not components.registry.get(info.document_id)
    assert not components.indexing.delete(info.document_id)


def test_partial_index_failure_retries(components):
    orig_insert = components.vectors.insert
    calls = 0

    def failing_insert(rows):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("temporary failure")
        orig_insert(rows)

    components.vectors.insert = failing_insert
    text = b"EEG sleep seizure depression signal analysis. " * 120
    components.indexing.batch_size = 2
    with pytest.raises(RuntimeError):
        components.indexing.index_bytes("failure.md", text)
    assert components.registry.list()[0].status == "failed"
    components.vectors.insert = orig_insert
    result = components.indexing.index_bytes("failure.md", text)
    assert result.status == "completed"
    assert len(components.vectors.rows) == result.chunk_count
