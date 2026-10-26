from eeg_agent_rag.ingestion.chunker import chunk_pages
from eeg_agent_rag.ingestion.loaders import Page, parse_document


def test_chunk_content_retains_page():
    pages = [Page(3, "EEG preprocessing " * 40), Page(5, "Milvus retrieval " * 30)]
    chunks = chunk_pages(pages, 140, 20)
    assert len(chunks) > 2
    assert {chunk.page for chunk in chunks} == {3, 5}
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert all(len(chunk.text) <= 140 for chunk in chunks)


def test_chunker_rejects_invalid_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_pages([Page(1, "abc")], 100, 100)


def test_parse_document_rejects_non_utf8():
    import pytest

    with pytest.raises(UnicodeDecodeError):
        parse_document("test.txt", b"\xff")


def test_scanned_pdf_rejected():
    import pytest
    import pymupdf

    pdf = pymupdf.open()
    pdf.new_page()
    content = pdf.tobytes()
    pdf.close()
    with pytest.raises(ValueError, match="No extractable text"):
        parse_document("paper.pdf", content)


def test_pdf_page_numbers():
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((40, 40), "Functional connectivity is discussed here.")
    data = doc.tobytes()
    doc.close()
    parsed = parse_document("paper.pdf", data)
    assert parsed[0].number == 1
    assert "Functional connectivity" in parsed[0].text
