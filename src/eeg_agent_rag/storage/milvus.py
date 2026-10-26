"""Milvus 2.6 adapter: native BM25 + Dense + RRF hybrid retrieval."""
from __future__ import annotations

import json
from typing import Any

from eeg_agent_rag.schemas.models import RetrievalHit

_OUTPUT_FIELDS = [
    "document_id", "content", "title", "source", "doi", "license", "year", "page", "section"
]


class MilvusVectorStore:
    def __init__(self, uri: str, collection: str, dimension: int, token: str = ""):
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise RuntimeError("Install runtime dependencies with: uv sync") from exc
        self.client = MilvusClient(uri=uri, **({"token": token} if token else {}))
        self.collection = collection
        self.dimension = dimension

    def ensure_collection(self) -> None:
        from pymilvus import DataType, Function, FunctionType

        if self.client.has_collection(self.collection):
            description = self.client.describe_collection(self.collection)
            fields = {field["name"]: field for field in description.get("fields", [])}
            dense = fields.get("dense")
            if dense is None:
                raise RuntimeError("Existing collection has incompatible schema (missing dense field)")
            existing_dim = int(dense.get("params", {}).get("dim", self.dimension))
            if existing_dim != self.dimension:
                raise RuntimeError(
                    f"Collection dimension {existing_dim} != embedding dimension {self.dimension}. "
                    "Create a new versioned collection to change embeddings."
                )
            self.client.load_collection(self.collection)
            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(
            field_name="content", datatype=DataType.VARCHAR, max_length=16384,
            enable_analyzer=True, analyzer_params={"type": "standard"},
        )
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="doi", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="license", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="year", datatype=DataType.INT64)
        schema.add_field(field_name="page", datatype=DataType.INT64)
        schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(Function(
            name="content_bm25", function_type=FunctionType.BM25,
            input_field_names=["content"], output_field_names=["sparse"],
        ))
        indexes = self.client.prepare_index_params()
        indexes.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
        indexes.add_index(
            field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25"
        )
        self.client.create_collection(
            collection_name=self.collection, schema=schema, index_params=indexes
        )
        self.client.load_collection(self.collection)

    def insert(self, rows: list[dict[str, Any]]) -> None:
        if rows:
            self.client.insert(collection_name=self.collection, data=rows)

    def delete_document(self, document_id: str) -> None:
        # document_id is a server-derived sha256 prefix; JSON encoding is defensive.
        expr = "document_id == " + json.dumps(document_id)
        self.client.delete(collection_name=self.collection, filter=expr)

    def ping(self) -> bool:
        return self.client.has_collection(self.collection)

    @staticmethod
    def _expr(year_from: int | None, year_to: int | None, source: str | None) -> str:
        conditions = []
        if year_from is not None:
            conditions.append(f"year >= {int(year_from)}")
        if year_to is not None:
            conditions.append(f"year <= {int(year_to)}")
        if source:
            conditions.append("source == " + json.dumps(source))
        return " and ".join(conditions)

    def search(
        self, query: str, vector: list[float] | None, mode: str, limit: int,
        year_from: int | None = None, year_to: int | None = None,
        source: str | None = None,
    ) -> list[RetrievalHit]:
        from pymilvus import AnnSearchRequest, RRFRanker

        expr = self._expr(year_from, year_to, source)
        if mode == "dense" and vector is None:
            raise ValueError("Dense search needs a query embedding")
        if mode == "hybrid" and vector is None:
            raise ValueError("Hybrid search needs a query embedding")
        kwargs = {"filter": expr} if expr else {}
        if mode == "hybrid":
            dense = AnnSearchRequest(
                data=[vector], anns_field="dense", param={"metric_type": "COSINE"},
                limit=limit, **kwargs,
            )
            sparse = AnnSearchRequest(
                data=[query], anns_field="sparse", param={"metric_type": "BM25"},
                limit=limit, **kwargs,
            )
            raw = self.client.hybrid_search(
                collection_name=self.collection, reqs=[dense, sparse],
                ranker=RRFRanker(60), limit=limit, output_fields=_OUTPUT_FIELDS,
                consistency_level="Strong",
            )
        else:
            raw = self.client.search(
                collection_name=self.collection,
                data=[query] if mode == "sparse" else [vector],
                anns_field="sparse" if mode == "sparse" else "dense",
                limit=limit, output_fields=_OUTPUT_FIELDS, consistency_level="Strong", **kwargs,
            )
        hits: list[RetrievalHit] = []
        for result in (raw[0] if raw else []):
            entity = result.get("entity", {})
            hits.append(RetrievalHit(
                chunk_id=str(result["id"]),
                document_id=str(entity.get("document_id", "")),
                content=str(entity.get("content", "")),
                title=str(entity.get("title", "")),
                source=str(entity.get("source", "")),
                doi=str(entity.get("doi", "")),
                license=str(entity.get("license", "unknown")),
                year=int(entity.get("year") or 0),
                page=int(entity.get("page") or 0),
                section=str(entity.get("section", "")),
                score=float(result.get("distance", 0.0)),
            ))
        return hits
