# EEG-Agent-RAG

**独立、可供 EEG Agent 复用的 Milvus 混合检索 RAG 服务**  
`FastAPI + Milvus 2.6 + BGE-M3 + BM25 + RRF + (optional) CrossEncoder + Qwen3/vLLM`

> Status: engineering-oriented starter release v0.1.0. It includes offline automated tests,
> but this release has **not** been live-verified against Docker Milvus or a GPU model server
> in its packaging environment. It is not a clinical diagnosis system.

## 1. What is implemented?

- Local PDF / Markdown / UTF-8 TXT ingest; page-aware chunks and source metadata.
- Content-hash deduplication, retry and file-level reindex; SQLite document registry.
- Milvus 2.6 schema with **dense 1024-d embeddings** and **built-in BM25** sparse vectors.
- Dense-only, BM25-only, Hybrid (RRF) retrieval; optional BGE CrossEncoder reranking.
- Controlled metadata filters (`year_from`, `year_to`, `source`) rather than raw filters.
- OpenAI-compatible Qwen3/vLLM answer generation, citation IDs `[S1]` and SSE token streaming.
- REST API / Swagger, API key option, standalone CLI and EEG-Agent-Core tool adapter.
- Offline mock-backed tests; retrieval metrics (Recall@K, MRR@K, nDCG@K, p50/p95 latency).
- Optional PubMed abstract import (search terms, PMID, DOI, year; **no full-text scraping**).

Not implemented yet: OCR, scientific table extraction, distributed task queue, PostgreSQL,
full tenant ACL/RBAC, GraphRAG, atomic Milvus/SQLite transactions, automated citation
entailment verification, or clinical-quality validation. See `docs/architecture.md`.

## 2. Quick start (Windows 11 + WSL2 / Ubuntu)

Prerequisites: Docker Desktop with WSL integration, [uv](https://docs.astral.sh/uv/),
Python 3.11–3.13. Milvus Standalone runs in Docker; API/Embedding runs natively in WSL.

```bash
# Extract project, enter the directory
cd EEG-Agent-RAG
cp .env.example .env

# Download the official, pinned Milvus 2.6 Standalone Compose file and start it.
# The script downloads from Milvus official release and starts Milvus + etcd + MinIO.
bash infra/start-milvus.sh

docker compose -f infra/milvus-standalone.yml ps

# CPU/GPU embedding support: first install Python deps and optional model deps.
uv sync --extra models --extra dev

# Ensure Milvus collection exists; first indexing downloads BAAI/bge-m3 (~GBs).
uv run eeg-rag init
uv run eeg-rag ingest data/sample/eeg_methods.md --source demo --license synthetic
uv run eeg-rag query "How do we check EEG bad channels?" --mode hybrid

# HTTP API and Swagger (a second terminal)
uv run uvicorn eeg_agent_rag.api.app:app --host 127.0.0.1 --port 8010
```

- API docs: `http://127.0.0.1:8010/docs`
- Milvus console: `http://127.0.0.1:9091/webui/`
- Real LLM answers require an independent running vLLM server (see section 4).
- The **first model download is network-dependent**; any live indexing/query requires Milvus.
- Because API keys are blank by default, bind the API to localhost and do not expose it
  to an untrusted network. For shared use, set `RAG_API_KEY` and configure TLS/proxy.

Test retrieval via HTTP:

```bash
curl -s -X POST http://127.0.0.1:8010/v1/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"Which quality-control checks are useful for EEG?","mode":"hybrid","top_k":3}'
```

Upload another paper:

```bash
curl -X POST http://127.0.0.1:8010/v1/documents \
  -F 'file=@/path/to/paper.pdf' \
  -F 'source=local-paper' -F 'year=2025' \
  -F 'doi=10.xxxx/example' -F 'license=check-source'
```

If `RAG_API_KEY` is set, add `-H 'X-API-Key: your-secret'` to each request.
**HTTP indexing waits until all chunks are committed** (no background queue in v0.1).

## 3. Test without BGE weights (quick interface smoke test)

Use a **new collection name** so a 64-dimensional demo vector collection never collides
with your BGE-M3 1024-dimensional collection:

```dotenv
RAG_EMBEDDING_PROVIDER=hashing
RAG_EMBEDDING_DIMENSION=64
RAG_MILVUS_COLLECTION=eeg_rag_hash_demo_v1
RAG_RERANK_PROVIDER=none
```

Now `uv sync --extra dev` is sufficient (no `--extra models`). Run `init`, `ingest`
and `query` as above. Hashing vectors are deterministic keyword sketches **not actual
semantic embeddings**, and must not be used for scientific benchmarking.

To revert, restore `bge`, `1024`, and use `eeg_rag_v1` collection name. Existing
collections should be versioned and reindexed when embedding/analyzer settings change.

## 4. Connect existing Qwen3-8B / vLLM

Run your **existing vLLM uv environment in a separate terminal**. For example,
replace the path with your own model folder:

```bash
vllm serve /home/zzz/models/Qwen3-8B \
    --served-model-name qwen3-8b \
    --host 127.0.0.1 --port 8000 \
    --max-model-len 8192 --gpu-memory-utilization 0.70
```

Change `.env` in the RAG repo to:

```dotenv
RAG_LLM_BASE_URL=http://127.0.0.1:8000/v1
RAG_LLM_API_KEY=EMPTY
RAG_LLM_MODEL=qwen3-8b
```

Example:

```bash
curl -X POST http://127.0.0.1:8010/v1/answer \
  -H 'Content-Type: application/json' \
  -d '{"query":"What does the document say about preprocessing?","top_k":3}'

# Streaming SSE (sources -> token* -> done)
curl -N -X POST http://127.0.0.1:8010/v1/answer/stream \
  -H 'Content-Type: application/json' \
  -d '{"query":"How should EEG quality control be documented?"}'
```

For 32 GB GPU use, consider leaving vLLM in its own env/process while running
BGE/CrossEncoder on CPU or an isolated GPU budget. If GPU memory is tight, set
`RAG_EMBEDDING_DEVICE=cpu`; this may increase latency substantially.

To enable actual reranking:

```dotenv
RAG_RERANK_PROVIDER=cross-encoder
RAG_RERANK_MODEL=BAAI/bge-reranker-v2-m3
```

This requires `uv sync --extra models` and downloads the additional model.

## 5. Main directory layout

```text
src/eeg_agent_rag/
  api/app.py                 FastAPI, upload, retrieve, answer, streaming SSE
  core/config.py             Settings and environment validation
  core/container.py          dependency construction/injection
  ingestion/loaders.py       PDF / TXT / MD parser
  ingestion/chunker.py       page-aware text splitting
  embeddings/providers.py    BGE-M3 / deterministic demo embedder
  storage/milvus.py          Milvus 2.6 schema / Dense / BM25 / Hybrid
  storage/metadata.py        SQLite document status/metadata
  retrieval/rerank.py        CrossEncoder / noop
  pipelines/indexing.py      indexing, retry, cleanup, dedup
  pipelines/retrieval.py     query orchestration
  pipelines/answer.py        evidence-anchored generation
  generation/citations.py    citation formatting and ID checks
  generation/llm.py          OpenAI-compatible vLLM client
  evaluation/metrics.py      retrieval metric formulas
scripts/import_pubmed.py     optional, rate-limited E-utilities abstract importer
scripts/evaluate_retrieval.py recall@k / MRR@k / nDCG@k report
examples/agent_tool_adapter.py   standalone Agent tool integration
infra/start-milvus.sh       official version-pinned Docker Compose bootstrap
tests/                      fully offline unit / mocked-API tests
```

The repository deliberately does **not** import LangGraph: call `/v1/retrieve` as
an Agent Tool from EEG-Agent-Core. The RAG service is also independently usable.

## 6. PubMed import (abstracts only)

Set a contact email (required by this helper) and use a modest number of records:

```bash
uv run python scripts/import_pubmed.py \
  --query '(electroencephalography[MeSH Terms]) AND (depression)' \
  --email your-research-email@example.com --limit 20
```

PubMed abstract and full-text reuse terms are not universal. Do not assume a PubMed
index entry means a paper is open-access or redistributable. For paper PDFs, prefer
PMC Open Access works with verified per-article license and provenance.

## 7. Retrieval evaluation and ablations

Generate/check your own annotated `query -> relevant_chunk_ids` set. Sample file
`data/sample/eeg_eval.jsonl` is **synthetic and very small**, suitable for a pipeline
smoke test only, not for model-quality claims. It assumes the sample file has been indexed.

```bash
uv run python scripts/evaluate_retrieval.py --dataset data/sample/eeg_eval.jsonl \
  --mode sparse --k 5 --output data/sparse.json
uv run python scripts/evaluate_retrieval.py --dataset data/sample/eeg_eval.jsonl \
  --mode dense --k 5 --output data/dense.json
uv run python scripts/evaluate_retrieval.py --dataset data/sample/eeg_eval.jsonl \
  --mode hybrid --k 5 --output data/hybrid.json
uv run python scripts/evaluate_retrieval.py --dataset data/sample/eeg_eval.jsonl \
  --mode hybrid --rerank --k 5 --output data/hybrid_rerank.json
```

Record embedding model, corpus snapshot, chunking strategy, index revision, search
configuration and hardware in any published benchmark. RRF score, BM25 score and
COSINE score have **different scales**; compare ranking metrics, not their raw scores.

## 8. Run offline unit/API tests

```bash
uv sync --extra dev
uv run pytest -q
# Optionally:
uv run ruff check src tests
```

The project contains mock-backed tests for import, failure recovery, duplicate ingest,
file deletion, filter behavior, citation IDs, SSE, API key and ranking metrics.
**These tests do not establish that Milvus/vLLM was successfully launched on your machine.**

## 9. WSL troubleshooting

- `Connection refused on :19530`: check Docker Desktop WSL integration and
  `docker compose -f infra/milvus-standalone.yml ps`.
- Model OOM: lower vLLM's GPU memory allocation or run Embedding/Reranker on CPU.
- Wrong embedding dimension: use a new collection name and reindex; never reuse
  a 64-d demo collection for BGE-M3's 1024-d vectors.
- Scanned PDFs: v0.1 rejects them rather than silently returning empty chunks;
  add a vetted OCR stage for this corpus type.
- Chinese BM25 behavior: the `standard` analyzer is not optimized for Chinese.
  Benchmark an appropriate Chinese tokenizer/analyzer before using bilingual corpora.
- `502 LLM/RAG backend error`: verify `RAG_LLM_BASE_URL` and served model ID in vLLM.

## 10. Roadmap and documented constraints

See [architecture](docs/architecture.md) and [API reference](docs/api.md). Recommended
next steps: async durable ingestion queue + PostgreSQL registry, improved PDF/table
parsing, collection schema migrations, tenant isolation, prompt injection tests, logging,
actual Milvus integration tests, grounded answer verification, and a knowledge-base UI.

**Medical usage disclaimer:** information retrieval and citation support only; not
medical diagnosis, clinical decision-making, or autonomous patient management.
