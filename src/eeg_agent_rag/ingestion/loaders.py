"""Local-file loaders; intentionally no URL fetching (prevents SSRF)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


@dataclass(frozen=True)
class Page:
    number: int
    text: str


def parse_document(filename: str, content: bytes, max_pages: int = 300) -> list[Page]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file extension: {suffix}")
    if not content:
        raise ValueError("Empty document")
    if suffix == ".pdf":
        import pymupdf

        if not content.startswith(b"%PDF"):
            raise ValueError("Invalid PDF signature")
        with pymupdf.open(stream=content, filetype="pdf") as pdf:
            if pdf.is_encrypted:
                raise ValueError("Encrypted PDF not supported")
            if pdf.page_count > max_pages:
                raise ValueError(f"PDF exceeds {max_pages} pages")
            pages = [Page(i + 1, page.get_text(sort=True)) for i, page in enumerate(pdf)]
    else:
        pages = [Page(1, content.decode("utf-8-sig", errors="strict"))]
    if not any(p.text.strip() for p in pages):
        raise ValueError("No extractable text; OCR/scanned PDFs are not supported")
    return pages
