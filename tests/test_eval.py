from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from rag_eval_service.app import app
from rag_eval_service.baseline import (
    check_against_baseline,
    corpus_fingerprint,
    freeze_baseline,
)
from rag_eval_service.metrics import evaluate_cases
from rag_eval_service.store import VectorStore

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"


def _load_fixtures():
    corpus = json.loads((EX / "corpus.json").read_text(encoding="utf-8"))
    cases = json.loads((EX / "cases.json").read_text(encoding="utf-8"))["cases"]
    store = VectorStore()
    for d in corpus["docs"]:
        store.upsert(d["id"], d["text"])
    return store, cases


def test_evaluate_hits_relevant_docs():
    store, cases = _load_fixtures()
    report = evaluate_cases(store, cases, k=2)
    assert report.hit_at_k == 1.0
    assert report.mrr > 0.5
    assert report.faithfulness_proxy > 0.5


def test_check_detects_regression_below_floor():
    store, cases = _load_fixtures()
    report = evaluate_cases(store, cases, k=2).to_dict()
    baseline = freeze_baseline(report, store.docs, tolerance=0.0)
    degraded = dict(report)
    degraded["hit_at_k"] = 0.0
    degraded["mrr"] = 0.0
    degraded["context_precision"] = 0.0
    degraded["faithfulness_proxy"] = 0.0
    verdict = check_against_baseline(degraded, store.docs, baseline)
    assert verdict.status == "REGRESSION"
    assert verdict.exit_code == 2


def test_check_refuses_when_corpus_fingerprint_mismatches():
    store, cases = _load_fixtures()
    report = evaluate_cases(store, cases, k=2).to_dict()
    baseline = freeze_baseline(report, store.docs, tolerance=0.05)
    store.upsert("poison", "unrelated text that changes the corpus hash")
    verdict = check_against_baseline(report, store.docs, baseline)
    assert verdict.status == "CORPUS_DRIFT"
    assert verdict.exit_code == 2
    assert baseline["corpus_sha256"] != corpus_fingerprint(store.docs)


def test_fastapi_evaluate_and_health():
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    corpus = json.loads((EX / "corpus.json").read_text(encoding="utf-8"))
    cases = json.loads((EX / "cases.json").read_text(encoding="utf-8"))
    resp = client.post(
        "/evaluate",
        json={"docs": corpus["docs"], "cases": cases["cases"], "k": 2},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["hit_at_k"] == 1.0
