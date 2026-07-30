# Production RAG Evaluation System

## BENCHMARK GATE

- [x] CI matrix targets Python 3.10, 3.11, and 3.12.
- [x] Named claim tests cover ingestion/query, API key, limits, persistent adapter,
  missing baseline, corpus drift, offline judge, and regression exit code.
- [x] `python scripts/run_example.py` reproduces the committed service transcript.
- [x] README-only fresh-clone path is under 30 minutes and needs no paid API.
- [x] Non-root Docker image and Compose topology are declared and CI-built.
- [x] `rag-eval check` dogfoods the frozen baseline in CI.
- [x] Public-doc retrieval fixture and 500-run report are committed under `examples/`.
- [x] `docs/INTERVIEW.md` carries three questions, a two-minute path, and limitations.
- [ ] Hosted deployment: not claimed and not required for this gate.

External benchmark section is N/A. The public-doc fixture is a project regression
set, not a claim of parity with BEIR, RAGAS, or another field leaderboard.

## Threat model

1. A corpus changes while an old score is reused: `corpus_sha256` fails closed.
2. A caller floods or oversized-payloads one process: body and per-client request
   limits reject it. Distributed rate limiting remains an infrastructure concern.
3. A vector database becomes unavailable: `/ready` returns 503.
4. An LLM provider is absent or unstable: default generation and judging stay local
   and deterministic; optional model failures return 503.
5. A persistent adapter silently drops metadata: the adapter contract test inspects
   payload and result mapping.

## Interview gate

1. Why is the vector-store protocol narrower than a vendor SDK?
2. What does the lexical judge prove, and what can it not prove?
3. Why should readiness check Qdrant while liveness should not?

Two-minute path:

```bash
pip install -e ".[dev]"
python scripts/run_example.py
rag-eval check --corpus examples/corpus.json --cases examples/cases.json \
  --baseline examples/baseline_v1.json
pytest -q
```

Limitation: hashing embeddings are deterministic and deployment-friendly, but they
do not capture semantic equivalence as well as a fitted dense embedding model.

## Daily ticks

- [x] W1 Extract vector-store contract and keep deterministic in-memory search.
- [x] W2 Add persistent Qdrant adapter and pinned hashing embeddings.
- [x] W3 Add chunked ingestion and retrieval-plus-generation query path.
- [x] W4 Add lexical and optional OpenAI-compatible generation/judge plugins.
- [x] W5 Add API key, request bounds, structured logs, liveness, and readiness.
- [x] W6 Add non-root image, Compose topology, and CI container gate.
- [x] W7 Run public-doc benchmark, update docs, tests, and interview evidence.

**NEXT TICK:** Replace hashing embeddings only when a pinned labeled benchmark
shows a candidate dense model improves recall enough to justify its runtime cost.

Remaining red checks: none for the accepted `rag-production` scope. Hosted service
and field-leaderboard claims remain deliberately out of scope.
