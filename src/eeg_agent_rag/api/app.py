"""HTTP endpoints. Model/GPU work is offloaded to a worker thread for async safety."""
from __future__ import annotations

import asyncio
import json
import secrets
import threading
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from eeg_agent_rag.core.config import Settings
from eeg_agent_rag.core.container import Services, build_services
from eeg_agent_rag.ingestion.loaders import SUPPORTED_SUFFIXES
from eeg_agent_rag.schemas.models import (
    AnswerRequest, AnswerResponse, DocumentInfo, HealthResponse, RetrievalRequest,
    RetrievalResponse,
)


def create_app(config: Settings | None = None, services: Services | None = None) -> FastAPI:
    app = FastAPI(
        title="EEG-Agent-RAG", version="0.1.0",
        description="Standalone, citation-aware Milvus 2.6 hybrid-RAG API",
    )
    app.state.config = config or Settings()
    app.state.services = services
    app.state.init_lock = threading.Lock()

    def get_services(request: Request) -> Services:
        if request.app.state.services is None:
            with request.app.state.init_lock:
                if request.app.state.services is None:
                    try:
                        request.app.state.services = build_services(request.app.state.config)
                    except Exception as exc:
                        raise HTTPException(status_code=503, detail=f"Service unavailable: {exc}") from exc
        return request.app.state.services

    def check_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
        expected = request.app.state.config.api_key
        if expected and (not x_api_key or not secrets.compare_digest(x_api_key, expected)):
            raise HTTPException(status_code=401, detail="Invalid API key")

    @app.get("/health", response_model=HealthResponse)
    def health(services: Services = Depends(get_services)):
        try:
            ready = services.vectors.ping()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Milvus health check failed") from exc
        if not ready:
            raise HTTPException(status_code=503, detail="Collection not ready")
        return HealthResponse(status="ok", milvus="connected", collection=app.state.config.milvus_collection)

    @app.post("/v1/documents", response_model=DocumentInfo, dependencies=[Depends(check_key)])
    async def upload_document(
        file: UploadFile = File(...),
        source: str = Form("upload"),
        doi: str = Form(""),
        year: int = Form(0),
        license: str = Form("unknown"),
        force: bool = Form(False),
        services: Services = Depends(get_services),
    ):
        from pathlib import Path

        filename = file.filename or ""
        if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise HTTPException(status_code=415, detail="Only PDF, TXT, Markdown files are supported")
        if year < 0 or year > 2200:
            raise HTTPException(status_code=422, detail="Invalid year")
        limit = app.state.config.upload_max_mb * 1024 * 1024
        try:
            data = await file.read(limit + 1)
        finally:
            await file.close()
        if len(data) > limit:
            raise HTTPException(status_code=413, detail="File exceeds configured size limit")
        try:
            # This initial implementation completes indexing within the request.
            return await asyncio.to_thread(
                services.indexing.index_bytes, filename, data, source, doi, year, license, force,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Document indexing failed; consult server logs") from exc

    @app.get("/v1/documents", response_model=list[DocumentInfo], dependencies=[Depends(check_key)])
    async def list_documents(limit: int = 100, services: Services = Depends(get_services)):
        if not 1 <= limit <= 1000:
            raise HTTPException(status_code=422, detail="limit must be 1..1000")
        return await asyncio.to_thread(services.registry.list, limit)

    @app.get("/v1/documents/{document_id}", response_model=DocumentInfo,
             dependencies=[Depends(check_key)])
    async def document_detail(document_id: str, services: Services = Depends(get_services)):
        result = await asyncio.to_thread(services.registry.get, document_id)
        if not result:
            raise HTTPException(status_code=404, detail="Document not found")
        return result

    @app.delete("/v1/documents/{document_id}", dependencies=[Depends(check_key)])
    async def delete_document(document_id: str, services: Services = Depends(get_services)):
        if not await asyncio.to_thread(services.indexing.delete, document_id):
            raise HTTPException(status_code=404, detail="Document not found")
        return {"deleted": document_id}

    @app.post("/v1/retrieve", response_model=RetrievalResponse,
              dependencies=[Depends(check_key)])
    async def retrieve(payload: RetrievalRequest, services: Services = Depends(get_services)):
        try:
            hits = await asyncio.to_thread(services.retrieval.retrieve, payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return RetrievalResponse(query=payload.query, results=hits)

    @app.post("/v1/answer", response_model=AnswerResponse,
              dependencies=[Depends(check_key)])
    async def answer(payload: AnswerRequest, services: Services = Depends(get_services)):
        try:
            return await services.answer.answer(payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail="LLM/RAG backend error") from exc

    @app.post("/v1/answer/stream", dependencies=[Depends(check_key)])
    async def answer_stream(payload: AnswerRequest, services: Services = Depends(get_services)):
        try:
            sources, prompt = await asyncio.to_thread(services.answer.prepare, payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        def event(kind: str, data: Any) -> str:
            return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        async def iterator():
            yield event("sources", [x.model_dump() for x in sources])
            if not sources:
                yield event("done", {"cited_source_ids": [], "warning": "No evidence retrieved"})
                return
            from eeg_agent_rag.generation.citations import check_citations

            parts = []
            try:
                async for token in services.answer.stream_prepared(prompt):
                    parts.append(token)
                    yield event("token", {"text": token})
                cited, warning = check_citations("".join(parts), len(sources))
                yield event("done", {"cited_source_ids": cited, "warning": warning})
            except Exception:
                yield event("error", {"message": "LLM stream failed"})

        return StreamingResponse(
            iterator(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()
