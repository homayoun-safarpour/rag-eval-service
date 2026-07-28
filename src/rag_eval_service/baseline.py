"""Frozen-metric baseline gate for RAG eval regression checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

METRIC_KEYS = ("hit_at_k", "mrr", "context_precision", "faithfulness_proxy")


@dataclass(frozen=True)
class GateVerdict:
    status: str  # PASS | REGRESSION | MISSING_BASELINE | CORPUS_DRIFT
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


def freeze_baseline(
    report: dict[str, Any],
    docs: dict[str, str],
    tolerance: float = 0.05,
) -> dict[str, Any]:
    return {
        "version": 1,
        "tolerance": tolerance,
        "corpus_sha256": corpus_fingerprint(docs),
        "metrics": {k: float(report[k]) for k in METRIC_KEYS},
    }


def write_baseline(path: str | Path, baseline: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")


def load_baseline(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_against_baseline(
    report: dict[str, Any],
    docs: dict[str, str],
    baseline: dict[str, Any] | None,
) -> GateVerdict:
    if baseline is None:
        return GateVerdict("MISSING_BASELINE", "no baseline provided", 2)
    if baseline.get("corpus_sha256") != corpus_fingerprint(docs):
        return GateVerdict(
            "CORPUS_DRIFT",
            "corpus_sha256 no longer matches the pin; re-freeze after intentional corpus change",
            2,
        )
    tol = float(baseline.get("tolerance", 0.0))
    pinned = baseline.get("metrics", {})
    drops: list[str] = []
    for key in METRIC_KEYS:
        floor = float(pinned[key]) - tol
        actual = float(report[key])
        if actual < floor:
            drops.append(f"{key}={actual:.4f} < floor={floor:.4f} (pinned={pinned[key]:.4f})")
    if drops:
        return GateVerdict("REGRESSION", "; ".join(drops), 2)
    return GateVerdict("PASS", "all metrics within tolerance of pinned baseline", 0)
