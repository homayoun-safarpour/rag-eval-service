"""Document chunking and ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_eval_service.store import VectorStoreProtocol, tokenize

MIN_CHUNK_SIZE = 10


@dataclass(frozen=True)
class SourceDocument:
    id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class IngestSummary:
    sources: int
    chunks: int
    chunk_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": self.sources,
            "chunks": self.chunks,
            "chunk_ids": self.chunk_ids,
        }


def chunk_text(text: str, size: int = 180, overlap: int = 30) -> list[str]:
    """Split text by tokens with stable overlap and no empty chunks."""
    if size < MIN_CHUNK_SIZE:
        raise ValueError("chunk size must be at least 10 tokens")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk overlap must be between 0 and size - 1")
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(words):
            break
    return chunks


def ingest_documents(
    store: VectorStoreProtocol,
    documents: list[SourceDocument],
    chunk_size: int = 180,
    chunk_overlap: int = 30,
) -> IngestSummary:
    chunk_ids: list[str] = []
    for document in documents:
        if not tokenize(document.text):
            continue
        for index, chunk in enumerate(chunk_text(document.text, chunk_size, chunk_overlap)):
            chunk_id = f"{document.id}::chunk-{index:04d}"
            metadata = {
                **document.metadata,
                "source_id": document.id,
                "chunk_index": index,
            }
            store.upsert(chunk_id, chunk, metadata)
            chunk_ids.append(chunk_id)
    return IngestSummary(
        sources=len(documents),
        chunks=len(chunk_ids),
        chunk_ids=chunk_ids,
    )
