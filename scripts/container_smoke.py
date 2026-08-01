"""Exercise the Compose API and persistent Qdrant path."""

from __future__ import annotations

import json
import os
import urllib.request

BASE_URL = os.getenv("RAG_SMOKE_URL", "http://127.0.0.1:8000")
AUTH_TOKEN = os.getenv("RAG_SMOKE_TOKEN", "local-development-key")


def request(path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"x-api-key": AUTH_TOKEN}
    if data is not None:
        headers["content-type"] = "application/json"
    call = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers=headers)
    with urllib.request.urlopen(call, timeout=10) as response:
        return json.loads(response.read())


def main() -> None:
    ready = request("/ready")
    if ready["backend"] != "qdrant":
        raise RuntimeError(f"expected qdrant backend, got {ready}")
    ingested = request(
        "/ingest",
        {
            "documents": [
                {
                    "id": "container-runbook",
                    "text": "Persistent readiness checks verify Qdrant before serving traffic.",
                    "metadata": {"source": "container-smoke"},
                }
            ],
            "chunk_size": 20,
            "chunk_overlap": 0,
        },
    )
    if ingested["chunks"] != 1:
        raise RuntimeError(f"expected one persisted chunk, got {ingested}")
    result = request(
        "/query",
        {
            "query": "What does persistent readiness verify?",
            "top_k": 1,
            "generator": "extractive",
            "judge": "lexical",
        },
    )
    if result["contexts"][0]["metadata"]["source"] != "container-smoke":
        raise RuntimeError(f"Qdrant payload did not round-trip: {result}")
    if not result["evaluation"]["passed"]:
        raise RuntimeError(f"offline evaluation failed: {result}")
    print(
        json.dumps(
            {
                "backend": ready["backend"],
                "chunks": ingested["chunks"],
                "context_id": result["contexts"][0]["id"],
                "judge": result["evaluation"]["backend"],
                "passed": result["evaluation"]["passed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
