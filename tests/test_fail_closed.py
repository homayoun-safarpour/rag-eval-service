"""Fail-closed contracts found by the code review of 25 Nov 2026."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_eval_service.app import Settings, create_app
from rag_eval_service.baseline import check_against_baseline, freeze_baseline
from rag_eval_service.cli import main
from rag_eval_service.metrics import evaluate_cases
from rag_eval_service.store import InMemoryVectorStore, VectorStore

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"
CORPUS = str(EX / "corpus.json")
CASES = str(EX / "cases.json")
FAIL_DEMO = EX / "baseline_fail_demo.json"
PACK = str(EX / "hidden_rag" / "pack.json")
TOKEN = "test-token"


def _fixtures():
    corpus = json.loads(Path(CORPUS).read_text(encoding="utf-8"))
    cases = json.loads(Path(CASES).read_text(encoding="utf-8"))["cases"]
    store = VectorStore()
    for doc in corpus["docs"]:
        store.upsert(doc["id"], doc["text"])
    return store, cases


def _check(capsys, cases: str, baseline: str, *extra: str) -> tuple[int, str]:
    code = main(
        ["check", "--corpus", CORPUS, "--cases", cases, "--baseline", baseline, *extra]
    )
    return code, capsys.readouterr().out


def _write_baseline(tmp_path: Path, mutate) -> str:
    data = json.loads(FAIL_DEMO.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


# Finding 1: the baseline pins the cases and k.


def test_check_refuses_cases_without_answers_and_other_k(tmp_path, capsys):
    cases = json.loads(Path(CASES).read_text(encoding="utf-8"))
    for case in cases["cases"]:
        case.pop("answer", None)
    stripped = tmp_path / "cases_no_answers.json"
    stripped.write_text(json.dumps(cases), encoding="utf-8")
    code, out = _check(capsys, str(stripped), str(FAIL_DEMO), "--k", "1")
    assert code == 2
    assert "verdict: CASES_DRIFT" in out
    assert "PASS" not in out


def test_check_refuses_other_k_on_the_same_cases(capsys):
    code, out = _check(capsys, CASES, str(EX / "baseline_v1.json"), "--k", "1")
    assert code == 2
    assert "verdict: CASES_DRIFT" in out


def test_check_refuses_changed_cases_in_library():
    store, cases = _fixtures()
    baseline = freeze_baseline(evaluate_cases(store, cases, k=3).to_dict(), store.docs)
    edited = [dict(case) for case in cases]
    edited[0]["answer"] = ""
    verdict = check_against_baseline(
        evaluate_cases(store, edited, k=3).to_dict(), store.docs, baseline
    )
    assert verdict.status == "CASES_DRIFT"
    assert verdict.exit_code == 2


def test_check_uses_the_pinned_k_when_none_is_given(tmp_path, capsys):
    out_path = tmp_path / "baseline_k2.json"
    freeze = ["freeze", "--corpus", CORPUS, "--cases", CASES, "--out", str(out_path)]
    assert main([*freeze, "--k", "2"]) == 0
    assert json.loads(out_path.read_text(encoding="utf-8"))["k"] == 2
    capsys.readouterr()
    code, out = _check(capsys, CASES, str(out_path))
    assert code == 0
    assert "verdict: PASS" in out


def test_baseline_without_cases_pin_is_refused(tmp_path, capsys):
    def drop_pins(data):
        for key in ("k", "n_cases", "cases_sha256"):
            data.pop(key, None)

    code, out = _check(capsys, CASES, _write_baseline(tmp_path, drop_pins))
    assert code == 2
    assert "verdict: INVALID_BASELINE" in out


# Finding 2: non-finite or out-of-range numbers in the baseline.


def test_nan_tolerance_in_baseline_is_refused(tmp_path, capsys):
    path = _write_baseline(tmp_path, lambda d: d.update(tolerance=float("nan")))
    code, out = _check(capsys, CASES, path)
    assert code == 2
    assert "verdict: INVALID_BASELINE" in out
    assert "tolerance" in out


def test_nan_pinned_metric_is_refused(tmp_path, capsys):
    path = _write_baseline(
        tmp_path, lambda d: d["metrics"].update(context_precision=float("nan"))
    )
    code, out = _check(capsys, CASES, path)
    assert code == 2
    assert "verdict: INVALID_BASELINE" in out
    assert "context_precision" in out


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("inf"), "0.1", True])
def test_out_of_range_tolerance_is_refused(bad):
    store, cases = _fixtures()
    report = evaluate_cases(store, cases, k=3).to_dict()
    baseline = freeze_baseline(report, store.docs)
    baseline["tolerance"] = bad
    verdict = check_against_baseline(report, store.docs, baseline)
    assert verdict.status == "INVALID_BASELINE"
    assert verdict.exit_code == 2


# Finding 3: a baseline without metrics.


def test_baseline_without_metrics_is_refused_not_crashed(tmp_path, capsys):
    code, out = _check(capsys, CASES, _write_baseline(tmp_path, lambda d: d.pop("metrics")))
    assert code == 2
    assert "verdict: INVALID_BASELINE" in out
    assert "metrics" in out


def test_api_check_with_baseline_without_metrics_returns_verdict_not_500():
    corpus = json.loads(Path(CORPUS).read_text(encoding="utf-8"))
    cases = json.loads(Path(CASES).read_text(encoding="utf-8"))["cases"]
    client = TestClient(create_app(Settings(api_key=TOKEN), InMemoryVectorStore()))
    response = client.post(
        "/check",
        headers={"x-api-key": TOKEN},
        json={"docs": corpus["docs"], "cases": cases, "baseline": {"version": 2}},
    )
    assert response.status_code == 200
    assert response.json()["verdict"]["status"] == "INVALID_BASELINE"
    assert response.json()["verdict"]["exit_code"] == 2


# Finding 4: hidden-ask input and usage errors get their own exit code.


def test_hidden_ask_bad_k_exits_3_not_blocked(capsys):
    code = main(
        ["hidden-ask", "--pack", PACK, "--query", "refunds", "--category", "support", "--k", "0"]
    )
    captured = capsys.readouterr()
    assert code == 3
    assert "blocked=" not in captured.out
    assert captured.err.startswith("error: ")


def test_hidden_ask_missing_pack_exits_3(tmp_path, capsys):
    missing = str(tmp_path / "no_pack.json")
    code = main(["hidden-ask", "--pack", missing, "--query", "refunds", "--category", "support"])
    assert code == 3
    assert capsys.readouterr().err.startswith("error: ")


def test_cli_usage_error_exits_3_not_2():
    with pytest.raises(SystemExit) as exc:
        main(["hidden-ask", "--pack", PACK, "--query", "refunds", "--category", "billing"])
    assert exc.value.code == 3


# Finding 5: no API key configured.


def test_api_refuses_protected_routes_when_no_key_is_configured():
    client = TestClient(create_app(Settings(), InMemoryVectorStore()))
    response = client.post("/ingest", json={"documents": [{"id": "a", "text": "hello world"}]})
    assert response.status_code == 503
    assert "RAG_API_KEY" in response.json()["detail"]
    assert client.post("/query", json={"query": "hello"}).status_code == 503
    assert client.get("/health").status_code == 200


def test_api_runs_open_only_with_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    monkeypatch.setenv("RAG_ALLOW_NO_AUTH", "1")
    client = TestClient(create_app(store=InMemoryVectorStore()))
    response = client.post("/ingest", json={"documents": [{"id": "a", "text": "hello world"}]})
    assert response.status_code == 200


# Finding 6: a chunked body carries no Content-Length.


def test_chunked_body_over_the_limit_is_refused():
    client = TestClient(
        create_app(Settings(api_key=TOKEN, max_request_bytes=20), InMemoryVectorStore())
    )

    def body():
        yield b'{"query": "this body exceeds twenty bytes by a lot"}'

    response = client.post(
        "/query",
        content=body(),
        headers={"content-type": "application/json", "x-api-key": TOKEN},
    )
    assert response.status_code == 413


def test_chunked_body_under_the_limit_still_works():
    client = TestClient(create_app(Settings(api_key=TOKEN), InMemoryVectorStore()))

    def body():
        yield b'{"query": "hello"}'

    response = client.post(
        "/query",
        content=body(),
        headers={"content-type": "application/json", "x-api-key": TOKEN},
    )
    assert response.status_code == 200
