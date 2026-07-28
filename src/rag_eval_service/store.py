"""In-memory bag-of-words vector index (no external vector DB required for CI)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

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


@dataclass
class VectorStore:
    """Lightweight sparse vector store used as the retrieval backend in demos and CI."""

    docs: dict[str, str] = field(default_factory=dict)
    _vectors: dict[str, dict[str, float]] = field(default_factory=dict)

    def upsert(self, doc_id: str, text: str) -> None:
        self.docs[doc_id] = text
        self._vectors[doc_id] = _tf(tokenize(text))

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        q = _tf(tokenize(query))
        scored = [(doc_id, cosine(q, vec)) for doc_id, vec in self._vectors.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
