from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from eeg_agent_rag.generation.citations import SYSTEM_PROMPT, check_citations, source_context
from eeg_agent_rag.schemas.models import AnswerRequest, AnswerResponse


@dataclass
class AnswerService:
    retriever: object
    generator: object

    def prepare(self, request: AnswerRequest):
        hits = self.retriever.retrieve(request)
        context, sources = source_context(hits)
        prompt = f"Question:\n{request.query}\n\nUntrusted reference excerpts:\n{context}"
        return sources, prompt

    async def answer(self, request: AnswerRequest) -> AnswerResponse:
        import asyncio

        sources, prompt = await asyncio.to_thread(self.prepare, request)
        if not sources:
            return AnswerResponse(
                answer="No matching evidence was found in the indexed documents.",
                sources=[], cited_source_ids=[], citation_warning="No evidence retrieved",
            )
        answer = await self.generator.generate(SYSTEM_PROMPT, prompt)
        cited, warning = check_citations(answer, len(sources))
        return AnswerResponse(
            answer=answer, sources=sources,
            cited_source_ids=cited, citation_warning=warning,
        )

    async def stream_prepared(self, prompt: str) -> AsyncIterator[str]:
        async for text in self.generator.stream(SYSTEM_PROMPT, prompt):
            yield text
