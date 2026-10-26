# Optional API container. Requires reachable Milvus/vLLM and model cache/GPU access.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
ARG INSTALL_MODELS=false
RUN if [ "$INSTALL_MODELS" = "true" ]; then uv sync --no-dev --extra models; \
    else uv sync --no-dev; fi
EXPOSE 8010
CMD ["uv", "run", "--no-sync", "uvicorn", "eeg_agent_rag.api.app:app", "--host", "0.0.0.0", "--port", "8010"]
