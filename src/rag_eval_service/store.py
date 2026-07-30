"""Vector-store contracts and the deterministic in-memory implementation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _tf(tokens: list[str]) -> dict[str, float]:
    counts: dict[str, float] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0.0) + 1.0
    n = float(len(tokens)) or 1.0
    return {k: v / n for k, v in counts.items()}


def _dot(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def _norm(v: dict[str, float]) -> float:
    return math.sqrt(sum(x * x for x in v.values())) or 1.0


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    return _dot(a, b) / (_norm(a) * _norm(b))


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Backend contract shared by deterministic CI and persistent stores."""

    def upsert(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def search(self, query: str, k: int = 5) -> list[SearchResult]: ...

    def snapshot(self) -> dict[str, str]: ...

    def count(self) -> int: ...

    def ready(self) -> bool: ...


@dataclass
class InMemoryVectorStore:
    """Sparse vector store used by the offline path and ordinary CI."""

    docs: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    _vectors: dict[str, dict[str, float]] = field(default_factory=dict)

    def upsert(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.docs[doc_id] = text
        self.metadata[doc_id] = metadata or {}
        self._vectors[doc_id] = _tf(tokenize(text))

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        q = _tf(tokenize(query))
        scored = [(doc_id, cosine(q, vec)) for doc_id, vec in self._vectors.items()]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            SearchResult(
                doc_id=doc_id,
                text=self.docs[doc_id],
                score=score,
                metadata=self.metadata.get(doc_id, {}),
            )
            for doc_id, score in scored[:k]
        ]

    def snapshot(self) -> dict[str, str]:
        return dict(self.docs)

    def count(self) -> int:
        return len(self.docs)

    def ready(self) -> bool:
        return True


# Backward-compatible name for the original public API.
VectorStore = InMemoryVectorStore
