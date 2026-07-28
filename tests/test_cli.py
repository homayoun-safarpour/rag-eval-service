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
