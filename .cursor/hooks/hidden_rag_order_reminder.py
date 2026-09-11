#!/usr/bin/env python3
"""Fail-open: warn if HiddenRAG ask() applies floor/k before scan_text."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path


def _read_payload(*, timeout_s: float = 1.0) -> dict:
    result: dict = {}

    def _reader() -> None:
        nonlocal result
        try:
            raw = sys.stdin.read()
            if raw.strip():
                result = json.loads(raw)
        except Exception:
            result = {}

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)
    return result


def _ask_body(text: str) -> str | None:
    marker = "    def ask("
    start = text.find(marker)
    if start < 0:
        return None
    return text[start:]


def main() -> int:
    payload = _read_payload()
    file_path = str(payload.get("file_path") or payload.get("path") or "")
    norm = file_path.replace("\\", "/").lower()
    if "hidden.py" not in norm:
        print(json.dumps({}))
        return 0
    path = Path(file_path)
    if not path.is_file():
        print(json.dumps({}))
        return 0
    try:
        body = _ask_body(path.read_text(encoding="utf-8"))
    except OSError:
        print(json.dumps({}))
        return 0
    if body is None:
        print(json.dumps({}))
        return 0
    scan_at = body.find("scan_text(hit.text)")
    floor_at = body.find("0.45")
    cutoff_at = body.find("][:k]")
    if 0 <= scan_at < floor_at < cutoff_at:
        print(json.dumps({}))
        return 0
    print(
        json.dumps(
            {
                "additional_context": (
                    "HiddenRAG ask() must scan_text on retrieved hits before the "
                    "score floor and k cutoff. Run tests/test_hidden_rag.py."
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
