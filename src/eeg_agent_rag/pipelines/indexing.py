"""Resumable at document level: content-hash IDs, explicit failures and safe reindex."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from eeg_agent_rag.embeddings.providers import Embedder
from eeg_agent_rag.ingestion.chunker import chunk_pages
from eeg_agent_rag.ingestion.loaders import parse_document
from eeg_agent_rag.schemas.models import DocumentInfo
from eeg_agent_rag.storage.metadata import DocumentRegistry


@dataclass
class IndexingService:
    vectors: object
    registry: DocumentRegistry
    embedder: Embedder
    chunk_size: int = 1000
    chunk_overlap: int = 150
    batch_size: int = 16

    def index_bytes(
        self, filename: str, content: bytes, source: str = "upload", doi: str = "",
        year: int = 0, license: str = "unknown", force: bool = False,
        title: str | None = None,
    ) -> DocumentInfo:
        doc_id = hashlib.sha256(content).hexdigest()[:32]
        previous = self.registry.get(doc_id)
        if previous and previous.status == "completed" and not force:
            return previous
        if previous and previous.status == "processing":
            raise ValueError("Document is already being indexed")
        # Avoid unescaped or overlong VARCHAR metadata.
        title = (title or Path(filename).name)[:250]
        item = DocumentInfo(
            document_id=doc_id, title=title, source=source[:200], doi=doi[:200],
            year=year, license=license[:100], status="processing",
        )
        self.registry.upsert(item)
        try:
            pages = parse_document(filename, content)
            chunks = chunk_pages(pages, self.chunk_size, self.chunk_overlap)
            if not chunks:
                raise ValueError("No valid text chunks")
            # Rebuild indexes for retries and explicit force=true. If a batch fails,
            # status is 'failed', and the next attempt deletes partial chunks first.
            self.vectors.delete_document(doc_id)
            for start in range(0, len(chunks), self.batch_size):
                batch = chunks[start:start + self.batch_size]
                vectors = self.embedder.embed([chunk.text for chunk in batch])
                if len(vectors) != len(batch):
                    raise RuntimeError("Embedding provider returned the wrong batch length")
                rows = []
                for chunk, vector in zip(batch, vectors, strict=True):
                    if len(vector) != self.embedder.dimension:
                        raise RuntimeError("Embedding provider returned wrong dimension")
                    rows.append({
                        "id": f"{doc_id}:{chunk.index:06d}",
                        "document_id": doc_id,
                        "content": chunk.text,
                        "title": item.title,
                        "source": item.source,
                        "doi": item.doi,
                        "license": item.license,
                        "section": chunk.section,
                        "year": item.year,
                        "page": chunk.page,
                        "dense": vector,
                    })
                self.vectors.insert(rows)
            result = item.model_copy(update={"status": "completed", "chunk_count": len(chunks)})
            self.registry.upsert(result)
            return result
        except Exception as exc:
            self.registry.upsert(item.model_copy(update={"status": "failed", "error": str(exc)[:500]}))
            raise

    def delete(self, document_id: str) -> bool:
        if not self.registry.get(document_id):
            return False
        self.vectors.delete_document(document_id)
        self.registry.delete(document_id)
        return True
