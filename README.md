# rag-eval-service

**RAG demos look fine in a notebook and then silently degrade after a corpus edit. This FastAPI service scores retrieval quality and fails the check when metrics drop below a frozen baseline.**

[![CI](https://github.com/homayoun-safarpour/rag-eval-service/actions/workflows/ci.yml/badge.svg)](https://github.com/homayoun-safarpour/rag-eval-service/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## The problem

Teams ship a RAG stack, then change the knowledge base without re-running eval. Hit rate and answer grounding drift. Logs still say "200 OK". You need a service that turns retrieval metrics into a deploy gate, not a slide in a weekly deck.

## Threat model (when this fails in production)

| Failure | What it looks like | What this repo does |
| --- | --- | --- |
| Silent corpus swap | Docs replaced; old scores still cited | `CORPUS_DRIFT` if `corpus_sha256` no longer matches the pin |
| Softened floor | Tolerance widened until anything passes | Baseline stores explicit tolerance; CI runs `check` not `evaluate` |
| LLM-flaky CI | Judge model grades differently tomorrow | Deterministic retrieval metrics + lexical faithfulness proxy; no model calls in the default path |
| Missing baseline | Script "evaluates" and always exits 0 | `MISSING_BASELINE` → exit `2` |
| Notebook-only eval | Metrics live in a private Colab | FastAPI `/evaluate` + `/check` and a CLI with the same contracts |

## Install

```bash
git clone https://github.com/homayoun-safarpour/rag-eval-service
cd rag-eval-service
pip install -e ".[dev]"
```

Python 3.10+. Docker image builds with the included Dockerfile.

## Quickstart

```bash
rag-eval evaluate --corpus examples/corpus.json --cases examples/cases.json
rag-eval check --corpus examples/corpus.json --cases examples/cases.json \
  --baseline examples/baseline_v1.json
```

Real output from this repository:

```
$ rag-eval evaluate --corpus examples/corpus.json --cases examples/cases.json
hit_at_k=1.0000 mrr=1.0000 context_precision=0.3333 faithfulness_proxy=1.0000

$ rag-eval check --corpus examples/corpus.json --cases examples/cases.json \
    --baseline examples/baseline_v1.json
verdict: PASS
  all metrics within tolerance of pinned baseline
```

API (optional):

```bash
uvicorn rag_eval_service.app:app --port 8000
# POST /evaluate and POST /check with the same JSON shapes as examples/
```

## How we did it

1. **Chose upstream patterns.** Field demand for RAG metrics and CI gates shows up in [explodinggradients/ragas](https://github.com/explodinggradients/ragas) (Apache-2.0) and small harnesses such as [camposvinicius/rag-evals](https://github.com/camposvinicius/rag-evals) (MIT). A full Ragas fork pulls LLM judges and heavy deps that break the under-30-minute bar.
2. **Restyled into one instrument.** MIT package `rag-eval-service`: FastAPI + CLI, in-memory sparse vector store for CI, hit@k / MRR / context precision / faithfulness proxy.
3. **Sharp improvement.** Frozen-metric regression gate with `corpus_sha256` so corpus edits cannot hide under old numbers. Named tests: `test_check_detects_regression_below_floor`, `test_check_refuses_when_corpus_fingerprint_mismatches`.
4. **Reproduce committed artifacts:**

```bash
rag-eval freeze --corpus examples/corpus.json --cases examples/cases.json \
  --out examples/baseline_v1.json --tolerance 0.05
```

## Compose with the rest of the stack

| Repo | Role next to this |
| --- | --- |
| [judge-reliability-kit](https://github.com/homayoun-safarpour/judge-reliability-kit) | When you *do* add an LLM judge on answers, diagnose panel disagreement |
| [judge-drift-sentinel](https://github.com/homayoun-safarpour/judge-drift-sentinel) | Detect judge drift on frozen anchors (exit `0`/`2`) |
| [trace-gate](https://github.com/homayoun-safarpour/trace-gate) | Gate agent tool-use trajectories the same way this gates retrieval |
| [agent-loop-engine](https://github.com/homayoun-safarpour/agent-loop-engine) | Wire `rag-eval check` as a quality gate (exit `2` = repair) |

## Docker

```bash
docker build -t rag-eval-service .
docker run --rm -p 8000:8000 rag-eval-service
```

## Topics

`rag` · `evaluation` · `fastapi` · `vector-search` · `llmops` · `ci-cd` · `python`

## License

MIT. Author: Homayoun Safarpour · [LinkedIn](https://www.linkedin.com/in/homayoun-safarpour/)
