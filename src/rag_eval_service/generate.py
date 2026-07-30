"""Pluggable answer generation with an offline deterministic default."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from rag_eval_service.store import SearchResult, tokenize


class Generator(Protocol):
    name: str

    def generate(self, query: str, contexts: list[SearchResult]) -> str: ...


@dataclass
class ExtractiveGenerator:
    """Select the context sentence with greatest lexical query overlap."""

    name: str = "extractive-v1"

    def generate(self, query: str, contexts: list[SearchResult]) -> str:
        query_tokens = set(tokenize(query))
        candidates: list[tuple[int, int, str]] = []
        for context_index, context in enumerate(contexts):
            normalized = context.text.replace("!", ".").replace("?", ".")
            for raw_sentence in normalized.split("."):
                sentence = raw_sentence.strip()
                if sentence:
                    overlap = len(query_tokens & set(tokenize(sentence)))
                    candidates.append((overlap, -context_index, sentence))
        if not candidates:
            return "No grounded answer found."
        candidates.sort(reverse=True)
        return f"{candidates[0][2]}."


@dataclass
class OpenAICompatibleGenerator:
    """Optional paid/API-backed generation, never used by default tests."""

    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    name: str = "openai-compatible"

    @classmethod
    def from_env(cls) -> OpenAICompatibleGenerator:
        api_key = os.getenv("RAG_LLM_API_KEY", "")
        model = os.getenv("RAG_LLM_MODEL", "")
        if not api_key or not model:
            raise RuntimeError("RAG_LLM_API_KEY and RAG_LLM_MODEL are required")
        return cls(
            model=model,
            api_key=api_key,
            base_url=os.getenv("RAG_LLM_BASE_URL", "https://api.openai.com/v1"),
        )

    def generate(self, query: str, contexts: list[SearchResult]) -> str:
        context_text = "\n\n".join(item.text for item in contexts)
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied context. "
                        "Say when context is insufficient."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nQuestion: {query}",
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read())
        return str(result["choices"][0]["message"]["content"]).strip()
