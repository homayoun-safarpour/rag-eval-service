from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_freeze_and_check_pass(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    freeze = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_eval_service.cli",
            "freeze",
            "--corpus",
            str(ROOT / "examples" / "corpus.json"),
            "--cases",
            str(ROOT / "examples" / "cases.json"),
            "--out",
            str(baseline),
            "--tolerance",
            "0.05",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert freeze.returncode == 0, freeze.stderr
    assert baseline.exists()
    data = json.loads(baseline.read_text(encoding="utf-8"))
    assert "corpus_sha256" in data
    check = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_eval_service.cli",
            "check",
            "--corpus",
            str(ROOT / "examples" / "corpus.json"),
            "--cases",
            str(ROOT / "examples" / "cases.json"),
            "--baseline",
            str(baseline),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stdout + check.stderr
    assert "PASS" in check.stdout


def test_cli_benchmark_writes_metrics_and_latency_evidence(tmp_path: Path):
    output = tmp_path / "benchmark.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_eval_service.cli",
            "benchmark",
            "--corpus",
            str(ROOT / "examples" / "benchmark_corpus_v1.json"),
            "--cases",
            str(ROOT / "examples" / "benchmark_cases_v1.json"),
            "--out",
            str(output),
            "--k",
            "3",
            "--runs",
            "5",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["documents"] == 12
    assert evidence["queries"] == 8
    assert evidence["metrics"]["hit_at_k"] == 1.0
    assert evidence["latency_ms"]["p95"] >= 0.0
