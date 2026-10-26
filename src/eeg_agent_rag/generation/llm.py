"""OpenAI-compatible client for vLLM; does not manage the model server lifecycle."""
from __future__ import annotations

from collections.abc import AsyncIterator


class VLLMGenerator:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float, max_tokens: int):
        from openai import AsyncOpenAI

        self.model = model
        self.max_tokens = max_tokens
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    async def generate(self, system: str, user: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.1,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    async def stream(self, system: str, user: str) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.1,
            max_tokens=self.max_tokens,
            stream=True,
        )
        async for piece in stream:
            delta = piece.choices[0].delta.content if piece.choices else None
            if delta:
                yield delta
