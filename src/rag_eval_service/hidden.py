"""Drop-in hidden RAG: category-filtered retrieve + grounded answer + injection scan.

Other projects import HiddenRAG or call `rag-eval hidden-ask`. No HTTP required.
Retrieved chunks are evidence. Instruction-shaped text is dropped, not obeyed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_eval_service.generate import ExtractiveGenerator
from rag_eval_service.injection import scan_query, scan_text
from rag_eval_service.prompts import CATEGORIES, pack_for
from rag_eval_service.store import InMemoryVectorStore, SearchResult, VectorStoreProtocol


@dataclass(frozen=True)
class HiddenAnswer:
    category: str
    query: str
    answer: str
    contexts: list[dict[str, Any]]
    blocked: bool
    reason: str
    prompt: dict[str, str]
    exit_code: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "query": self.query,
            "answer": self.answer,
            "contexts": self.contexts,
            "blocked": self.blocked,
            "reason": self.reason,
            "prompt": self.prompt,
            "exit_code": self.exit_code,
        }


_SMALL_STORE_SCAN_ALL = 1024


class HiddenRAG:
    """In-process RAG pack for support / runbook / policy / docs corpora."""

    def __init__(self, store: VectorStoreProtocol | None = None) -> None:
        self.store = store or InMemoryVectorStore()
        self._generator = ExtractiveGenerator()

    def upsert(self, doc_id: str, text: str, category: str, **meta: Any) -> None:
        category_key = category.strip().lower()
        if category_key not in CATEGORIES:
            allowed = ", ".join(CATEGORIES)
            raise ValueError(f"unknown category {category!r}; use one of: {allowed}")
        self.store.upsert(doc_id, text, {"category": category_key, **meta})

    def load_pack(self, path: str | Path) -> int:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        docs = payload["docs"] if isinstance(payload, dict) else payload
        count = 0
        for doc in docs:
            self.upsert(str(doc["id"]), str(doc["text"]), str(doc["category"]))
            count += 1
        return count

    def _pool_k(self, k: int) -> int:
        pool = max(k * 4, k)
        try:
            n = int(self.store.count())
        except (TypeError, ValueError, AttributeError):
            return pool
        if n <= 0:
            return pool
        if n <= _SMALL_STORE_SCAN_ALL:
            return n
        return min(n, max(pool, k * 16))

    def ask(self, query: str, category: str, k: int = 3) -> HiddenAnswer:
        if k < 1:
            raise ValueError("k must be >= 1")
        pack = pack_for(category)
        qscan = scan_query(query)
        if qscan.blocked:
            return HiddenAnswer(
                category=pack.category,
                query=query,
                answer="",
                contexts=[],
                blocked=True,
                reason=qscan.reason,
                prompt=pack.render(query, ""),
                exit_code=1,
            )
        raw_hits = self.store.search(query, k=self._pool_k(k))
        dropped_in_category = 0
        clean_hits: list[SearchResult] = []
        for hit in raw_hits:
            in_category = str(hit.metadata.get("category", "")).lower() == pack.category
            if scan_text(hit.text).blocked:
                if in_category:
                    dropped_in_category += 1
                continue
            clean_hits.append(hit)
        best = clean_hits[0].score if clean_hits else 0.0
        floor = max(0.05, 0.45 * best)
        kept = [
            hit
            for hit in clean_hits
            if str(hit.metadata.get("category", "")).lower() == pack.category
            and hit.score >= floor
        ][:k]
        if not kept:
            reason = (
                "all_retrieved_chunks_blocked"
                if dropped_in_category
                else "no_category_match_or_empty_retrieve"
            )
            return HiddenAnswer(
                category=pack.category,
                query=query,
                answer="",
                contexts=[],
                blocked=dropped_in_category > 0,
                reason=reason,
                prompt=pack.render(query, ""),
                exit_code=1 if dropped_in_category else 2,
            )
        evidence = "\n\n".join(item.text for item in kept)
        answer = self._generator.generate(query, kept)
        return HiddenAnswer(
            category=pack.category,
            query=query,
            answer=answer,
            contexts=[
                {
                    "doc_id": item.doc_id,
                    "score": item.score,
                    "text": item.text,
                    "category": pack.category,
                }
                for item in kept
            ],
            blocked=False,
            reason="",
            prompt=pack.render(query, evidence),
            exit_code=0,
        )
