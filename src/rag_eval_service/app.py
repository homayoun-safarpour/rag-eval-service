"""FastAPI surface for RAG evaluation and regression checks."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_eval_service.baseline import check_against_baseline, freeze_baseline
from rag_eval_service.metrics import evaluate_cases
from rag_eval_service.store import VectorStore

app = FastAPI(
    title="rag-eval-service",
    description="Score RAG retrieval quality; gate deploys on a frozen metric baseline.",
    version="0.1.0",
)


class DocIn(BaseModel):
    id: str
    text: str


class CaseIn(BaseModel):
    id: str
    query: str
    relevant_ids: list[str] = Field(default_factory=list)
    answer: str = ""


class EvaluateRequest(BaseModel):
    docs: list[DocIn]
    cases: list[CaseIn]
    k: int = 3


class CheckRequest(EvaluateRequest):
    baseline: dict[str, Any] | None = None
    freeze: bool = False
    tolerance: float = 0.05


def _build_store(docs: list[DocIn]) -> VectorStore:
    store = VectorStore()
    for d in docs:
        store.upsert(d.id, d.text)
    return store


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/evaluate")
def evaluate(req: EvaluateRequest) -> dict[str, Any]:
    if not req.docs or not req.cases:
        raise HTTPException(status_code=400, detail="docs and cases are required")
    store = _build_store(req.docs)
    report = evaluate_cases(store, [c.model_dump() for c in req.cases], k=req.k)
    return report.to_dict()


@app.post("/check")
def check(req: CheckRequest) -> dict[str, Any]:
    if not req.docs or not req.cases:
        raise HTTPException(status_code=400, detail="docs and cases are required")
    store = _build_store(req.docs)
    report = evaluate_cases(store, [c.model_dump() for c in req.cases], k=req.k)
    payload = report.to_dict()
    if req.freeze:
        baseline = freeze_baseline(payload, store.docs, tolerance=req.tolerance)
        return {"report": payload, "baseline": baseline, "verdict": {"status": "FROZEN"}}
    verdict = check_against_baseline(payload, store.docs, req.baseline)
    return {"report": payload, "verdict": verdict.to_dict()}
