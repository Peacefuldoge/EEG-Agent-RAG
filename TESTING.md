# Verification report (2026-10-26)

- Source code successfully parsed/compiled with Python 3.13.5.
- `PYTHONPATH=src pytest -q`: **18 passed** (offline mock-backed tests).
- A complete smoke workflow was exercised using a fake vector store: PDF/text parsing,
  deduplicating indexer, retry after failed batch, metadata filtering, REST API auth,
  citation output and SSE streaming.
- A pymilvus call-shape test checks hybrid requests (Dense, raw-text BM25, filters, RRF).
- Live Milvus Standalone, actual BGE model download/inference, Qwen3 vLLM completion,
  external PubMed HTTP calls, and Docker image building **were not run** because this
  packaging environment has no outbound PyPI/network access or Milvus service.
- `uv.lock` is intentionally absent. Generate it in a networked WSL2 environment with
  `uv sync --extra dev --extra models`, inspect, then commit if desired.

Known limitations and production hardening items are documented in `docs/architecture.md`.
