"""Run: uv run python examples/python_client.py"""
import httpx

BASE = "http://127.0.0.1:8010"

with httpx.Client(timeout=120) as client:
    with open("data/sample/eeg_methods.md", "rb") as source:
        response = client.post(
            f"{BASE}/v1/documents",
            files={"file": ("eeg_methods.md", source, "text/markdown")},
            data={"source": "demo", "license": "synthetic"},
        )
    response.raise_for_status()
    print("Indexed:", response.json())
    response = client.post(
        f"{BASE}/v1/retrieve",
        json={"query": "How can EEG quality control detect bad channels?", "top_k": 3},
    )
    response.raise_for_status()
    print("Results:", response.json())
