# Interview talking points : rag-eval-service

Five CLI-backed points for a technical screen (no resume recap).

- **`python scripts/run_example.py`** : ingests the bundled corpus, runs `/query`, prints contexts, extractive answer, and lexical evaluation with no model download.
- **`rag-eval evaluate --corpus examples/corpus.json --cases examples/cases.json --json`** : aggregate retrieval metrics on disk; use before you freeze a baseline.
- **`rag-eval freeze --corpus examples/corpus.json --cases examples/cases.json --out /tmp/baseline.json`** : writes hit@k floors plus `corpus_sha256` so later checks reject a swapped knowledge base.
- **`rag-eval check --corpus examples/corpus.json --cases examples/cases.json --baseline examples/baseline_v1.json`** : exit **0** on pass, exit **2** when metrics fall below the frozen floor or corpus fingerprint mismatches.
- **`pytest -q`** : named proofs include `test_check_detects_regression_below_floor`, `test_check_refuses_when_corpus_fingerprint_mismatches`, and offline `test_query_path_returns_context_answer_and_offline_evaluation`.
