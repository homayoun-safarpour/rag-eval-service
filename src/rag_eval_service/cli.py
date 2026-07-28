"""CLI: evaluate / freeze / check RAG fixtures (exit 0 pass / 2 regression)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_eval_service.baseline import (
    check_against_baseline,
    freeze_baseline,
    load_baseline,
    write_baseline,
)
from rag_eval_service.metrics import evaluate_cases
from rag_eval_service.store import VectorStore


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _store_from_corpus(corpus: dict) -> VectorStore:
    store = VectorStore()
    for doc in corpus["docs"]:
        store.upsert(str(doc["id"]), str(doc["text"]))
    return store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="Score RAG retrieval fixtures and gate on a frozen baseline.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("evaluate", help="print aggregate metrics")
    ev.add_argument("--corpus", required=True)
    ev.add_argument("--cases", required=True)
    ev.add_argument("--k", type=int, default=3)
    ev.add_argument("--json", action="store_true")

    fr = sub.add_parser("freeze", help="write baseline JSON from current metrics")
    fr.add_argument("--corpus", required=True)
    fr.add_argument("--cases", required=True)
    fr.add_argument("--out", required=True)
    fr.add_argument("--k", type=int, default=3)
    fr.add_argument("--tolerance", type=float, default=0.05)

    ch = sub.add_parser("check", help="fail closed if metrics drop below baseline")
    ch.add_argument("--corpus", required=True)
    ch.add_argument("--cases", required=True)
    ch.add_argument("--baseline", required=True)
    ch.add_argument("--k", type=int, default=3)

    args = parser.parse_args(argv)
    corpus = _load_json(args.corpus)
    cases = _load_json(args.cases)["cases"]
    store = _store_from_corpus(corpus)
    report = evaluate_cases(store, cases, k=args.k)
    payload = report.to_dict()

    if args.cmd == "evaluate":
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(
                f"hit_at_k={payload['hit_at_k']:.4f} "
                f"mrr={payload['mrr']:.4f} "
                f"context_precision={payload['context_precision']:.4f} "
                f"faithfulness_proxy={payload['faithfulness_proxy']:.4f}"
            )
        return 0

    if args.cmd == "freeze":
        baseline = freeze_baseline(payload, store.docs, tolerance=args.tolerance)
        write_baseline(args.out, baseline)
        print(f"wrote baseline -> {args.out}")
        return 0

    baseline = load_baseline(args.baseline)
    verdict = check_against_baseline(payload, store.docs, baseline)
    print(f"verdict: {verdict.status}")
    print(f"  {verdict.detail}")
    return verdict.exit_code


if __name__ == "__main__":
    sys.exit(main())
