"""One-command offline ingestion and query example."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from rag_eval_service.app import Settings, create_app
from rag_eval_service.store import InMemoryVectorStore

LOCAL_AUTH_TOKEN = "local-example"


def main() -> None:
    client = TestClient(
        create_app(Settings(api_key=LOCAL_AUTH_TOKEN), InMemoryVectorStore())
    )
    headers = {"x-api-key": LOCAL_AUTH_TOKEN}
    ingest = client.post(
        "/ingest",
        headers=headers,
        json={
            "documents": [
                {
                    "id": "runbook",
                    "text": (
                        "Readiness checks verify the vector store before traffic is served. "
                        "Health checks only verify that the API process is alive."
                    ),
                    "metadata": {"source": "operations-runbook"},
                }
            ],
            "chunk_size": 30,
            "chunk_overlap": 0,
        },
    )
    ingest.raise_for_status()
    query = client.post(
        "/query",
        headers=headers,
        json={"query": "What does the readiness check verify?", "top_k": 1},
    )
    query.raise_for_status()
    print(json.dumps({"ingest": ingest.json(), "query": query.json()}, indent=2))


if __name__ == "__main__":
    main()
