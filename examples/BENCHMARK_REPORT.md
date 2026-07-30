# Retrieval benchmark report

## Scope

`public-ai-infrastructure-docs-v1` is a committed regression fixture with 12 short
documents and eight queries derived from pinned public documentation for FastAPI
0.116.1, Qdrant 1.15.3, CPython 3.12.11, and Uvicorn 0.35.0. It checks this
service's deterministic retrieval path. It is not a standard leaderboard and does
not establish performance on long documents, multilingual corpora, or semantic
paraphrases.

## Reproduce

```bash
rag-eval benchmark \
  --corpus examples/benchmark_corpus_v1.json \
  --cases examples/benchmark_cases_v1.json \
  --out examples/benchmark_public_docs_v1.json \
  --k 3 --runs 500
```

Environment: Windows, CPython 3.13.2. Committed result:

- hit@3: `1.0000`
- MRR: `1.0000`
- context precision@3: `0.3333`
- lexical faithfulness proxy: `0.9110`
- 500-run mean evaluation latency: `0.7178 ms`
- p95 evaluation latency: `1.0192 ms`
- maximum evaluation latency: `2.9041 ms`

Latency is local process time for 12 in-memory documents. It excludes HTTP,
Qdrant, embedding-model, and network latency. The exact latency numbers are
machine-dependent; retrieval metrics are deterministic.

## Honest verdict

The fixture proves that the frozen sparse retriever returns every labeled document
within the top three on this small public-doc set and that the benchmark command
writes a reproducible artifact. The low context precision is expected because each
query labels one relevant document while `k=3`; it is not hidden by the report.

This result does not prove dense-retrieval quality. The Qdrant adapter uses pinned
384-dimensional hashing embeddings so Compose works without a model download.
Production users should replace that embedder when semantic recall matters and
then freeze a new baseline against their own labeled corpus.
