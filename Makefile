.PHONY: install install-models milvus api test lint init demo
install:
	uv sync --extra dev
install-models:
	uv sync --extra models --extra dev
milvus:
	bash infra/start-milvus.sh
init:
	uv run eeg-rag init
demo:
	uv run eeg-rag ingest data/sample/eeg_methods.md --source demo --license synthetic
	uv run eeg-rag query "EEG quality control" --mode hybrid
api:
	uv run uvicorn eeg_agent_rag.api.app:app --host 127.0.0.1 --port 8010
test:
	uv run pytest -q
lint:
	uv run ruff check src tests
