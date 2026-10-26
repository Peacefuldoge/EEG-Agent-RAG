"""Stable character-based chunks, preserving source page numbers."""
from __future__ import annotations

import re
from dataclasses import dataclass

from eeg_agent_rag.ingestion.loaders import Page


@dataclass(frozen=True)
class Chunk:
    page: int
    index: int
    text: str
    section: str = ""


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n")
    return re.sub(r"[ \t]+", " ", text).strip()


def chunk_pages(pages: list[Page], chunk_size: int, overlap: int) -> list[Chunk]:
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    chunks: list[Chunk] = []
    for page in pages:
        text = normalize_text(page.text)
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                # Prefer natural sentence or paragraph boundaries without tiny fragments.
                candidates = [text.rfind(ch, start + chunk_size // 2, end) for ch in ("\n", ". ", "。", "！", "？")]
                split = max(candidates)
                if split > start:
                    end = split + 1
            part = text[start:end].strip()
            if part:
                chunks.append(Chunk(page=page.number, index=len(chunks), text=part))
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks
