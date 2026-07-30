"""Hardened FastAPI surface for ingestion, query, and evaluation."""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from rag_eval_service.baseline import check_against_baseline, freeze_baseline
from rag_eval_service.generate import ExtractiveGenerator, OpenAICompatibleGenerator
from rag_eval_service.ingest import SourceDocument, ingest_documents
from rag_eval_service.judges import LexicalJudge, OpenAICompatibleJudge
from rag_eval_service.metrics import evaluate_cases
from rag_eval_service.qdrant import QdrantVectorStore
from rag_eval_service.store import InMemoryVectorStore, VectorStoreProtocol

logger = logging.getLogger("rag_eval_service")
RATE_WINDOW_SECONDS = 60
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    max_request_bytes: int = 1_000_000
    requests_per_minute: int = 120
    max_documents: int = 100
    vector_backend: str = "memory"
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "rag_documents"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            api_key=os.getenv("RAG_API_KEY", ""),
            max_request_bytes=int(os.getenv("RAG_MAX_REQUEST_BYTES", "1000000")),
            requests_per_minute=int(os.getenv("RAG_REQUESTS_PER_MINUTE", "120")),
            max_documents=int(os.getenv("RAG_MAX_DOCUMENTS", "100")),
            vector_backend=os.getenv("RAG_VECTOR_BACKEND", "memory").lower(),
            qdrant_url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "rag_documents"),
        )


class DocIn(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaseIn(BaseModel):
    id: str
    query: str
    relevant_ids: list[str] = Field(default_factory=list)
    answer: str = ""


class EvaluateRequest(BaseModel):
    docs: list[DocIn] = Field(min_length=1, max_length=100)
    cases: list[CaseIn] = Field(min_length=1, max_length=500)
    k: int = Field(default=3, ge=1, le=50)


class CheckRequest(EvaluateRequest):
    baseline: dict[str, Any] | None = None
    freeze: bool = False
    tolerance: float = 0.05


class IngestRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "documents": [
                        {
                            "id": "runbook",
                            "text": "Readiness checks verify the vector store.",
                            "metadata": {"team": "platform"},
                        }
                    ],
                    "chunk_size": 180,
                    "chunk_overlap": 30,
                }
            ]
        }
    )

    documents: list[DocIn] = Field(min_length=1, max_length=100)
    chunk_size: int = Field(default=180, ge=10, le=2_000)
    chunk_overlap: int = Field(default=30, ge=0, le=500)


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "What does readiness verify?",
                    "top_k": 3,
                    "generator": "extractive",
                    "judge": "lexical",
                }
            ]
        }
    )

    query: str = Field(min_length=1, max_length=4_000)
    top_k: int = Field(default=5, ge=1, le=50)
    generator: str = Field(default="extractive", pattern="^(extractive|llm)$")
    judge: str = Field(default="lexical", pattern="^(lexical|llm)$")


def _build_store(docs: list[DocIn]) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    for d in docs:
        store.upsert(d.id, d.text, d.metadata)
    return store


def _configured_store(settings: Settings) -> VectorStoreProtocol:
    if settings.vector_backend == "memory":
        return InMemoryVectorStore()
    if settings.vector_backend == "qdrant":
        return QdrantVectorStore(
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
        )
    raise ValueError(f"unsupported RAG_VECTOR_BACKEND: {settings.vector_backend}")


def create_app(  # noqa: PLR0915
    settings: Settings | None = None,
    store: VectorStoreProtocol | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    store = store or _configured_store(settings)
    rate_windows: dict[str, deque[float]] = defaultdict(deque)
    api = FastAPI(
        title="rag-eval-service",
        description="Ingest, query, score, and regression-gate RAG retrieval.",
        version="0.2.0",
    )
    api.state.store = store
    api.state.settings = settings

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and (
            x_api_key is None or not secrets.compare_digest(x_api_key, settings.api_key)
        ):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    @api.middleware("http")
    async def hardening_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        content_length = int(request.headers.get("content-length", "0") or "0")
        if content_length > settings.max_request_bytes:
            return _json_response(413, {"detail": "request body too large"}, request_id)

        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = rate_windows[client]
        while window and now - window[0] >= RATE_WINDOW_SECONDS:
            window.popleft()
        if len(window) >= settings.requests_per_minute:
            return _json_response(429, {"detail": "rate limit exceeded"}, request_id)
        window.append(now)

        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        logger.info(
            json.dumps(
                {
                    "event": "http_request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1_000, 3),
                },
                sort_keys=True,
            )
        )
        return response

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/ready")
    def ready() -> dict[str, Any]:
        is_ready = store.ready()
        if not is_ready:
            raise HTTPException(status_code=503, detail="vector store unavailable")
        return {"status": "ready", "backend": settings.vector_backend, "documents": store.count()}

    @api.post("/ingest", dependencies=[Depends(require_api_key)])
    def ingest(req: IngestRequest) -> dict[str, Any]:
        if len(req.documents) > settings.max_documents:
            raise HTTPException(status_code=422, detail="document limit exceeded")
        try:
            summary = ingest_documents(
                store,
                [
                    SourceDocument(id=doc.id, text=doc.text, metadata=doc.metadata)
                    for doc in req.documents
                ],
                chunk_size=req.chunk_size,
                chunk_overlap=req.chunk_overlap,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return summary.to_dict()

    @api.post("/query", dependencies=[Depends(require_api_key)])
    def query(req: QueryRequest) -> dict[str, Any]:
        contexts = store.search(req.query, k=req.top_k)
        try:
            generator = (
                ExtractiveGenerator()
                if req.generator == "extractive"
                else OpenAICompatibleGenerator.from_env()
            )
            judge = (
                LexicalJudge()
                if req.judge == "lexical"
                else OpenAICompatibleJudge.from_env()
            )
            answer = generator.generate(req.query, contexts)
            evaluation = judge.evaluate(req.query, answer, contexts)
        except (OSError, RuntimeError, KeyError, ValueError) as exc:
            raise HTTPException(
                status_code=503,
                detail=f"optional model backend failed: {exc}",
            ) from exc
        return {
            "query": req.query,
            "contexts": [
                {
                    "id": item.doc_id,
                    "text": item.text,
                    "score": item.score,
                    "metadata": item.metadata,
                }
                for item in contexts
            ],
            "answer": answer,
            "generator": generator.name,
            "evaluation": evaluation.to_dict(),
        }

    @api.post("/evaluate", dependencies=[Depends(require_api_key)])
    def evaluate(req: EvaluateRequest) -> dict[str, Any]:
        eval_store = _build_store(req.docs)
        report = evaluate_cases(eval_store, [c.model_dump() for c in req.cases], k=req.k)
        return report.to_dict()

    @api.post("/check", dependencies=[Depends(require_api_key)])
    def check(req: CheckRequest) -> dict[str, Any]:
        eval_store = _build_store(req.docs)
        report = evaluate_cases(eval_store, [c.model_dump() for c in req.cases], k=req.k)
        payload = report.to_dict()
        if req.freeze:
            baseline = freeze_baseline(payload, eval_store.docs, tolerance=req.tolerance)
            return {"report": payload, "baseline": baseline, "verdict": {"status": "FROZEN"}}
        verdict = check_against_baseline(payload, eval_store.docs, req.baseline)
        return {"report": payload, "verdict": verdict.to_dict()}

    return api


def _json_response(status_code: int, body: dict[str, Any], request_id: str):
    return JSONResponse(
        status_code=status_code,
        content=body,
        headers={"x-request-id": request_id},
    )


app = create_app()
