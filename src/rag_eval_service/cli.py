"""CLI: evaluate / freeze / check RAG fixtures (exit 0 pass / 2 regression)."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
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


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
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

    bench = sub.add_parser("benchmark", help="run a pinned retrieval benchmark")
    bench.add_argument("--corpus", required=True)
    bench.add_argument("--cases", required=True)
    bench.add_argument("--out", required=True)
    bench.add_argument("--k", type=int, default=3)
    bench.add_argument("--runs", type=int, default=100)

    args = parser.parse_args(argv)
    corpus = _load_json(args.corpus)
    cases = _load_json(args.cases)["cases"]
    store = _store_from_corpus(corpus)
    started = time.perf_counter()
    report = evaluate_cases(store, cases, k=args.k)
    elapsed = time.perf_counter() - started
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

    if args.cmd == "benchmark":
        latencies_ms: list[float] = []
        for _ in range(max(1, args.runs)):
            run_started = time.perf_counter()
            evaluate_cases(store, cases, k=args.k)
            latencies_ms.append((time.perf_counter() - run_started) * 1_000)
        latencies_ms.sort()
        index_95 = min(len(latencies_ms) - 1, int(len(latencies_ms) * 0.95))
        artifact = {
            "schema_version": 1,
            "engine": "deterministic-sparse-v1",
            "python": platform.python_version(),
            "documents": store.count(),
            "queries": len(cases),
            "k": args.k,
            "warmup_ms": round(elapsed * 1_000, 4),
            "runs": len(latencies_ms),
            "latency_ms": {
                "mean": round(sum(latencies_ms) / len(latencies_ms), 4),
                "p95": round(latencies_ms[index_95], 4),
                "max": round(max(latencies_ms), 4),
            },
            "metrics": {
                key: payload[key]
                for key in ("hit_at_k", "mrr", "context_precision", "faithfulness_proxy")
            },
        }
        Path(args.out).write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(f"wrote benchmark -> {args.out}")
        return 0

    baseline = load_baseline(args.baseline)
    verdict = check_against_baseline(payload, store.docs, baseline)
    print(f"verdict: {verdict.status}")
    print(f"  {verdict.detail}")
    return verdict.exit_code


if __name__ == "__main__":
    sys.exit(main())
