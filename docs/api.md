# REST API

API docs: `http://127.0.0.1:8010/docs` or `/openapi.json`.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Milvus collection probe |
| POST | `/v1/documents` | multipart PDF/TXT/MD ingest (blocking until complete) |
| GET | `/v1/documents` | list indexed files and statuses |
| GET | `/v1/documents/{document_id}` | get status |
| DELETE | `/v1/documents/{document_id}` | delete document and vectors |
| POST | `/v1/retrieve` | dense, sparse or hybrid search |
| POST | `/v1/answer` | retrieved evidence + vLLM answer |
| POST | `/v1/answer/stream` | SSE events `sources`, `token`, `done`, `error` |

When `RAG_API_KEY` is nonempty, add `X-API-Key` to all `/v1` endpoints.

Example retrieval:

```json
{
  "query": "What QC signals should be checked before EEG classification?",
  "mode": "hybrid",
  "top_k": 5,
  "rerank": true,
  "year_from": 2020,
  "year_to": 2026
}
```

SSE uses event names and JSON `data:` payloads, not OpenAI ChatCompletion chunks.
`/v1/answer` needs a running OpenAI-compatible backend, but `/v1/retrieve` doesn't.
Scores: the raw `score` from Milvus has different scales in BM25/COSINE/RRF and
must not be compared between different retrieval modes. CrossEncoder `rerank_score`
is a raw model score, not an absolute probability.
