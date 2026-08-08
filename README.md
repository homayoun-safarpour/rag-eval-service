# rag-eval-service

**A RAG endpoint can stay healthy while retrieval quality, corpus identity, and answer grounding fail. This service turns those failures into tested API and CI contracts.**

[![CI](https://github.com/homayoun-safarpour/rag-eval-service/actions/workflows/ci.yml/badge.svg)](https://github.com/homayoun-safarpour/rag-eval-service/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

`rag-eval-service` ingests and chunks documents, retrieves context, generates a
grounded answer, evaluates that answer, and fails CI when frozen retrieval metrics
regress. The ordinary path is deterministic and needs no model download or paid API.

## The problem

Teams often monitor HTTP availability but not the behavior behind the endpoint. A
corpus replacement can invalidate old scores. A missing baseline can turn an eval
command into an always-green script. An LLM judge can make CI depend on a provider.
This service keeps those concerns separate and exposes each as a named contract.

| Contract | Behavior | Named proof |
| --- | --- | --- |
| Persistent storage | Qdrant REST adapter stores vectors, text, and metadata | `test_qdrant_adapter_persists_payload_and_maps_search_results` |
| End-to-end RAG | `/ingest` then `/query` returns contexts, answer, and evaluation | `test_query_path_returns_context_answer_and_offline_evaluation` |
| Offline default | Extractive generator and lexical judge use no API key | `test_offline_generator_and_judge_need_no_api_key` |
| Corpus identity | Changed document content returns `CORPUS_DRIFT` | `test_check_refuses_when_corpus_fingerprint_mismatches` |
| Metric floor | A drop below the frozen floor returns exit `2` | `test_check_detects_regression_below_floor` |
| API access | Protected routes reject a missing or wrong key | `test_api_key_protects_mutating_and_query_routes` |
| Request controls | Oversized and over-rate requests return `413` and `429` | `test_request_size_limit_fails_before_handler`, `test_rate_limit_returns_429` |

## Architecture

```text
POST /ingest -> token chunks -> VectorStoreProtocol -> memory (CI)
                                                \-> Qdrant (persistent)

POST /query  -> retrieve -> extractive generator -> lexical judge
                         \-> optional LLM generator/judge

rag-eval check -> hit@k + MRR + context precision + grounding
               -> corpus_sha256 + frozen floors -> exit 0 / 2
```

The Qdrant path uses a pinned 384-dimensional hashing embedder. That choice makes
the Compose path reproducible and free of model downloads. The adapter and embedder
are separate, so a production embedding model can replace it without changing API
or gate contracts.

## Install

Interview pack: [docs/INTERVIEW.md](docs/INTERVIEW.md).

Claim boundaries: [docs/RELIABILITY_CARD.md](docs/RELIABILITY_CARD.md).

```bash
git clone https://github.com/homayoun-safarpour/rag-eval-service
cd rag-eval-service
pip install -e ".[dev]"
```

Python 3.10 or newer is supported.

## One-command offline path

```bash
python scripts/run_example.py
```

The command performs authenticated ingestion and query in-process. Its captured
output is committed at `examples/service_transcript_v1.json`. It returns one chunk,
the retrieved context, an extractive answer, and a lexical judge score of `1.0`.

Run the frozen gate:

```bash
rag-eval check --corpus examples/corpus.json --cases examples/cases.json \
  --baseline examples/baseline_v1.json
```

Real output:

```text
verdict: PASS
  all metrics within tolerance of pinned baseline
```

## API

Start the deterministic in-memory service:

```bash
RAG_API_KEY=replace-me uvicorn rag_eval_service.app:app --port 8000
```

Ingest:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "content-type: application/json" -H "x-api-key: replace-me" \
  -d '{"documents":[{"id":"runbook","text":"Readiness checks verify the vector store.","metadata":{"team":"platform"}}],"chunk_size":30,"chunk_overlap":0}'
```

Query:

```bash
curl -X POST http://localhost:8000/query \
  -H "content-type: application/json" -H "x-api-key: replace-me" \
  -d '{"query":"What does readiness verify?","top_k":3,"generator":"extractive","judge":"lexical"}'
```

`GET /health` checks the API process. `GET /ready` checks the configured vector
store and reports its document count. Every response carries `x-request-id`;
request logs are structured JSON.

### Optional LLM path

The default test suite never calls an LLM. To use an OpenAI-compatible endpoint:

```bash
export RAG_LLM_API_KEY=...
export RAG_LLM_MODEL=...
export RAG_JUDGE_API_KEY=...
export RAG_JUDGE_MODEL=...
```

Set `"generator":"llm"` and/or `"judge":"llm"` in `/query`. Separate credentials
allow generation and judging to use different providers. Provider failures return
`503`; they do not fall through to an unreported lexical score.

## Persistent Compose path

```bash
docker compose up --build
```

Compose starts the non-root API container and pinned `qdrant/qdrant:v1.15.3`, with
Qdrant storage on the `qdrant-data` volume. The local API key is
`local-development-key`; replace it outside local development.

Configuration:

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_VECTOR_BACKEND` | `memory` | `memory` or `qdrant` |
| `QDRANT_URL` | `http://qdrant:6333` | Persistent store endpoint |
| `QDRANT_COLLECTION` | `rag_documents` | Collection name |
| `RAG_API_KEY` | empty | Protects ingest, query, evaluate, and check |
| `RAG_MAX_REQUEST_BYTES` | `1000000` | Content-length limit |
| `RAG_REQUESTS_PER_MINUTE` | `120` | Per-process, per-client limit |

## Benchmark evidence

The committed public-document fixture contains 12 documents and eight queries,
pinned to public FastAPI, Qdrant, CPython, and Uvicorn documentation versions.

```bash
rag-eval benchmark \
  --corpus examples/benchmark_corpus_v1.json \
  --cases examples/benchmark_cases_v1.json \
  --out examples/benchmark_public_docs_v1.json \
  --k 3 --runs 500
```

The captured run produced hit@3 `1.0000`, MRR `1.0000`, context precision
`0.3333`, lexical faithfulness `0.9110`, and local p95 evaluation latency
`1.0192 ms`. Read `examples/BENCHMARK_REPORT.md` before using the numbers. This is
a small regression fixture, not a standard field leaderboard.

## CI and extension points

GitHub Actions runs ruff, pytest, and `rag-eval check` on Python 3.10, 3.11, and
3.12. A separate job starts the Compose stack, asserts UID `10001`, and exercises
ingestion, Qdrant payload persistence, retrieval, generation, and judging through
the containerized API.

Extension interfaces:

- implement `VectorStoreProtocol` for another persistent index;
- implement `Generator` for another answer backend;
- implement `Judge` for another evaluation policy;
- replace `hashing_embedding` and freeze a labeled baseline for the new embedder.

See `CONTRIBUTING.md` for bounded extension tasks.

## Limitations

- Hashing embeddings provide lexical features, not learned semantic similarity.
- The in-process rate limiter is not a distributed quota system.
- Qdrant network behavior is covered by an adapter contract test and Compose build;
  the default unit suite does not require a running Qdrant service.
- The lexical judge measures token support. It does not establish factual
  correctness, completeness, or citation entailment.
- No hosted endpoint is claimed.

## Related instruments

- [judge-reliability-kit](https://github.com/homayoun-safarpour/judge-reliability-kit)
  diagnoses why an LLM judge panel disagrees.
- [judge-drift-sentinel](https://github.com/homayoun-safarpour/judge-drift-sentinel)
  detects judge drift on frozen anchors.
- [agent-loop-engine](https://github.com/homayoun-safarpour/agent-loop-engine)
  can consume `rag-eval check` exit codes as a repair gate.
- [judge-field-guide](https://github.com/homayoun-safarpour/judge-field-guide)
  link-checked map of the LLM-judge tool ecosystem.

## Field alignment

Ireland AI-first QA language (golden floors, drift, offline eval) maps to `rag-eval check` exit codes. Claim boundaries: [docs/RELIABILITY_CARD.md](docs/RELIABILITY_CARD.md).

## License

MIT. Author: Homayoun Safarpour · [LinkedIn](https://www.linkedin.com/in/homayoun-safarpour/)
