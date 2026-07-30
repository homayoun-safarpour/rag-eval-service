"""Deterministic local embeddings for reproducible retrieval tests."""

from __future__ import annotations

import hashlib
import math

from rag_eval_service.store import tokenize

DEFAULT_EMBEDDING_MODEL = "hashing-v1-384"
DEFAULT_DIMENSIONS = 384


def hashing_embedding(text: str, dimensions: int = DEFAULT_DIMENSIONS) -> list[float]:
    """Map tokens to a signed, normalized feature vector without model downloads."""
    vector = [0.0] * dimensions
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]
