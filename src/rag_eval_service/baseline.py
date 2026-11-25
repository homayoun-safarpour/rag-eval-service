"""Frozen-metric baseline gate for RAG eval regression checks."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

METRIC_KEYS = ("hit_at_k", "mrr", "context_precision", "faithfulness_proxy")
BASELINE_VERSION = 2
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GateVerdict:
    # PASS | REGRESSION | MISSING_BASELINE | INVALID_BASELINE | CORPUS_DRIFT | CASES_DRIFT
    status: str
    detail: str
    exit_code: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "detail": self.detail,
            "exit_code": self.exit_code,
        }


def corpus_fingerprint(docs: dict[str, str]) -> str:
    payload = json.dumps(docs, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cases_fingerprint(cases: list[dict[str, Any]]) -> str:
    payload = json.dumps(cases, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze_baseline(
    report: dict[str, Any],
    docs: dict[str, str],
    tolerance: float = 0.05,
) -> dict[str, Any]:
    return {
        "version": BASELINE_VERSION,
        "tolerance": tolerance,
        "corpus_sha256": corpus_fingerprint(docs),
        "cases_sha256": report["cases_sha256"],
        "n_cases": int(report["n_cases"]),
        "k": int(report["k"]),
        "metrics": {k: float(report[k]) for k in METRIC_KEYS},
    }


def write_baseline(path: str | Path, baseline: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")


def load_baseline(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _is_unit_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _is_count(value: Any, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def baseline_problem(baseline: Any) -> str | None:
    """Return why a baseline cannot be used, or None when its shape is sound."""
    if not isinstance(baseline, dict):
        return "baseline is not a JSON object"
    metrics = baseline.get("metrics")
    checks = [
        (
            _is_unit_number(baseline.get("tolerance", 0.0)),
            "tolerance must be a finite number between 0 and 1",
        ),
        (isinstance(baseline.get("corpus_sha256"), str), "corpus_sha256 is missing"),
        (
            isinstance(baseline.get("cases_sha256"), str)
            and bool(_SHA256.match(baseline["cases_sha256"]))
            and _is_count(baseline.get("n_cases"), 0)
            and _is_count(baseline.get("k"), 1),
            "baseline does not pin cases_sha256, n_cases and k; re-freeze it",
        ),
        (isinstance(metrics, dict), "metrics is missing"),
    ]
    if isinstance(metrics, dict):
        checks += [
            (
                _is_unit_number(metrics.get(key)),
                f"metrics.{key} must be a finite number between 0 and 1",
            )
            for key in METRIC_KEYS
        ]
    return next((message for ok, message in checks if not ok), None)


def check_against_baseline(
    report: dict[str, Any],
    docs: dict[str, str],
    baseline: dict[str, Any] | None,
) -> GateVerdict:
    if baseline is None:
        return GateVerdict("MISSING_BASELINE", "no baseline provided", 2)
    problem = baseline_problem(baseline)
    if problem:
        return GateVerdict("INVALID_BASELINE", problem, 2)
    if baseline["corpus_sha256"] != corpus_fingerprint(docs):
        return GateVerdict(
            "CORPUS_DRIFT",
            "corpus_sha256 no longer matches the pin; re-freeze after intentional corpus change",
            2,
        )
    mismatches = [
        f"{key}={str(report.get(key))[:12]} (pinned={str(baseline[key])[:12]})"
        for key in ("k", "n_cases", "cases_sha256")
        if report.get(key) != baseline[key]
    ]
    if mismatches:
        return GateVerdict(
            "CASES_DRIFT",
            "cases or k differ from the pin: " + "; ".join(mismatches),
            2,
        )
    tol = float(baseline.get("tolerance", 0.0))
    pinned = baseline["metrics"]
    drops: list[str] = []
    for key in METRIC_KEYS:
        floor = float(pinned[key]) - tol
        actual = float(report[key])
        if not actual >= floor:
            drops.append(f"{key}={actual:.4f} < floor={floor:.4f} (pinned={pinned[key]:.4f})")
    if drops:
        return GateVerdict("REGRESSION", "; ".join(drops), 2)
    return GateVerdict("PASS", "all metrics within tolerance of pinned baseline", 0)
