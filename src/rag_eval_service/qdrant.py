"""Small Qdrant REST adapter with deterministic local embeddings."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rag_eval_service.embeddings import DEFAULT_DIMENSIONS, hashing_embedding
from rag_eval_service.store import SearchResult

Transport = Callable[[str, str, dict[str, Any] | None], dict[str, Any]]


def _default_transport(method: str, url: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qdrant {method} {url} failed: {exc.code} {detail}") from exc
    return json.loads(raw) if raw else {}


@dataclass
class QdrantVectorStore:
    """Persistent vector storage through Qdrant's stable REST API."""

    url: str = "http://localhost:6333"
    collection: str = "rag_documents"
    dimensions: int = DEFAULT_DIMENSIONS
    transport: Transport = field(default=_default_transport, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.transport(method, f"{self.url.rstrip('/')}{path}", payload)

    def ensure_collection(self) -> None:
        if self._initialized:
            return
        try:
            self._request("GET", f"/collections/{self.collection}")
        except RuntimeError:
            self._request(
                "PUT",
                f"/collections/{self.collection}",
                {"vectors": {"size": self.dimensions, "distance": "Cosine"}},
            )
        self._initialized = True

    def upsert(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.ensure_collection()
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag-eval-service:{doc_id}"))
        self._request(
            "PUT",
            f"/collections/{self.collection}/points?wait=true",
            {
                "points": [
                    {
                        "id": point_id,
                        "vector": hashing_embedding(text, self.dimensions),
                        "payload": {
                            "doc_id": doc_id,
                            "text": text,
                            "metadata": metadata or {},
                        },
                    }
                ]
            },
        )

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        self.ensure_collection()
        response = self._request(
            "POST",
            f"/collections/{self.collection}/points/search",
            {
                "vector": hashing_embedding(query, self.dimensions),
                "limit": k,
                "with_payload": True,
            },
        )
        results: list[SearchResult] = []
        for item in response.get("result", []):
            payload = item.get("payload") or {}
            results.append(
                SearchResult(
                    doc_id=str(payload.get("doc_id", item["id"])),
                    text=str(payload.get("text", "")),
                    score=float(item.get("score", 0.0)),
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
        return results

    def snapshot(self) -> dict[str, str]:
        """Return stored IDs and text for baseline fingerprinting."""
        self.ensure_collection()
        response = self._request(
            "POST",
            f"/collections/{self.collection}/points/scroll",
            {"limit": 10_000, "with_payload": True, "with_vector": False},
        )
        return {
            str(item["payload"]["doc_id"]): str(item["payload"]["text"])
            for item in response.get("result", {}).get("points", [])
        }

    def count(self) -> int:
        self.ensure_collection()
        response = self._request(
            "POST",
            f"/collections/{self.collection}/points/count",
            {"exact": True},
        )
        return int(response.get("result", {}).get("count", 0))

    def ready(self) -> bool:
        try:
            self.ensure_collection()
        except (OSError, RuntimeError):
            return False
        return True
