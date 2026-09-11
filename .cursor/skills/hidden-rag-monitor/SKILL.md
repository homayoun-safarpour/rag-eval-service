---
name: hidden-rag-monitor
description: Catch HiddenRAG retrieve-order and injection mistakes. Use when editing rag-eval-service hidden.py, injection.py, prompts.py, hidden-ask CLI, or tests/test_hidden_rag.py, and when the user asks to review, monitor, or find mistakes in hidden RAG.
---

# Hidden RAG monitor

Retrieved chunks are evidence. They cannot set the score floor, fill `k`, or become instructions.

## Required order in `HiddenRAG.ask`

1. Block instruction-shaped **queries**.
2. Retrieve a pool large enough that poison cannot hide clean docs (`count()` when the store is small).
3. `scan_text` every hit; drop blocked text.
4. Count in-category drops only (out-of-category poison is not exit 1).
5. Score floor from **clean** hits.
6. Category filter, then `k`.

If step 3 happens after 5 or 6, clean evidence is discarded. That bug already shipped once.

## After any edit

```bash
python -m pytest -q tests/test_hidden_rag.py
python -m ruff check src/rag_eval_service/hidden.py src/rag_eval_service/injection.py
```

Named proofs that must stay green: floor ignores poison, `k` applies after drop, poison flood does not hide a clean hit, whitespace variants of `ignore previous instructions` are blocked, source order `scan_text` before `0.45` before `][:k]`.
