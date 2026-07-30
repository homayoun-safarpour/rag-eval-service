"""Answer-judge plugins with a deterministic offline default."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import asdict, dataclass
from typing import Protocol

from rag_eval_service.store import SearchResult, tokenize


@dataclass(frozen=True)
class JudgeResult:
    backend: str
    score: float
    passed: bool
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


class Judge(Protocol):
    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[SearchResult],
    ) -> JudgeResult: ...


@dataclass
class LexicalJudge:
    """Measure answer-token support in retrieved context."""

    threshold: float = 0.6

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[SearchResult],
    ) -> JudgeResult:
        del query
        answer_tokens = set(tokenize(answer))
        context_tokens = {
            token
            for context in contexts
            for token in tokenize(context.text)
        }
        score = (
            len(answer_tokens & context_tokens) / len(answer_tokens)
            if answer_tokens
            else 0.0
        )
        return JudgeResult(
            backend="lexical-v1",
            score=score,
            passed=score >= self.threshold,
            rationale=f"{score:.3f} of unique answer tokens are supported by retrieved context",
        )


@dataclass
class OpenAICompatibleJudge:
    """Optional JSON-mode judge for environments with an LLM endpoint."""

    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"

    @classmethod
    def from_env(cls) -> OpenAICompatibleJudge:
        api_key = os.getenv("RAG_JUDGE_API_KEY", "")
        model = os.getenv("RAG_JUDGE_MODEL", "")
        if not api_key or not model:
            raise RuntimeError("RAG_JUDGE_API_KEY and RAG_JUDGE_MODEL are required")
        return cls(
            model=model,
            api_key=api_key,
            base_url=os.getenv("RAG_JUDGE_BASE_URL", "https://api.openai.com/v1"),
        )

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[SearchResult],
    ) -> JudgeResult:
        context_text = "\n\n".join(item.text for item in contexts)
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Judge whether the answer is supported by context. Return JSON with "
                        "score (0..1), passed (boolean), and rationale (string)."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\nAnswer: {answer}\nContext:\n{context_text}"
                    ),
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
        content = json.loads(result["choices"][0]["message"]["content"])
        score = max(0.0, min(1.0, float(content["score"])))
        return JudgeResult(
            backend=f"llm:{self.model}",
            score=score,
            passed=bool(content["passed"]),
            rationale=str(content["rationale"]),
        )
