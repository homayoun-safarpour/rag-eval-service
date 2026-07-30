"""Retrieval metrics and faithfulness proxy (deterministic, no LLM calls)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rag_eval_service.store import VectorStoreProtocol, tokenize


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    hit_at_k: float
    mrr: float
    context_precision: float
    faithfulness_proxy: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvalReport:
    cases: list[CaseResult]
    hit_at_k: float
    mrr: float
    context_precision: float
    faithfulness_proxy: float

    def to_dict(self) -> dict:
        return {
            "n_cases": len(self.cases),
            "hit_at_k": self.hit_at_k,
            "mrr": self.mrr,
            "context_precision": self.context_precision,
            "faithfulness_proxy": self.faithfulness_proxy,
            "cases": [c.to_dict() for c in self.cases],
        }


def _hit_and_mrr(ranked_ids: list[str], relevant: set[str]) -> tuple[float, float]:
    hit = 0.0
    mrr = 0.0
    for i, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            hit = 1.0
            mrr = 1.0 / i
            break
    return hit, mrr


def _precision(ranked_ids: list[str], relevant: set[str]) -> float:
    if not ranked_ids:
        return 0.0
    return sum(1 for d in ranked_ids if d in relevant) / len(ranked_ids)


def _faithfulness_proxy(answer: str, contexts: list[str]) -> float:
    """Share of answer tokens that appear in retrieved context (lexical grounding)."""
    ans = set(tokenize(answer))
    if not ans:
        return 1.0
    ctx = set()
    for c in contexts:
        ctx.update(tokenize(c))
    if not ctx:
        return 0.0
    return len(ans & ctx) / len(ans)


def evaluate_cases(
    store: VectorStoreProtocol,
    cases: list[dict],
    k: int = 3,
) -> EvalReport:
    """Score a list of {id, query, relevant_ids, answer?} against the store."""
    results: list[CaseResult] = []
    for case in cases:
        case_id = str(case["id"])
        query = str(case["query"])
        relevant = {str(x) for x in case.get("relevant_ids", [])}
        answer = str(case.get("answer", ""))
        ranked = store.search(query, k=k)
        ranked_ids = [item.doc_id for item in ranked]
        contexts = [item.text for item in ranked]
        hit, mrr = _hit_and_mrr(ranked_ids, relevant)
        prec = _precision(ranked_ids, relevant)
        faith = _faithfulness_proxy(answer, contexts) if answer else 1.0
        results.append(
            CaseResult(
                case_id=case_id,
                hit_at_k=hit,
                mrr=mrr,
                context_precision=prec,
                faithfulness_proxy=faith,
            )
        )
    n = len(results) or 1
    return EvalReport(
        cases=results,
        hit_at_k=sum(r.hit_at_k for r in results) / n,
        mrr=sum(r.mrr for r in results) / n,
        context_precision=sum(r.context_precision for r in results) / n,
        faithfulness_proxy=sum(r.faithfulness_proxy for r in results) / n,
    )
