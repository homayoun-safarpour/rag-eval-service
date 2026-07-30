# Interview gate: rag-eval-service

## Three questions

1. **Why pin `corpus_sha256` on the baseline?**
   Without it, someone can swap the knowledge base, keep the same case IDs, and make hit@k look fine while answering from different content. Fingerprint mismatch forces an intentional re-freeze.

2. **Why use a hashing embedder with Qdrant?**
   It gives the persistent path stable vector dimensions and no model-download dependency. It proves the adapter, persistence, and deployment contracts. It does not make a semantic-retrieval claim.

3. **What does the lexical judge measure without an LLM?**
   Lexical grounding: share of answer tokens that appear in retrieved context. It is a CI-cheap signal, not a substitute for a judge model on open-ended answers.

## Two-minute path

```bash
git clone https://github.com/homayoun-safarpour/rag-eval-service
cd rag-eval-service
pip install -e ".[dev]"
python scripts/run_example.py
rag-eval check --corpus examples/corpus.json --cases examples/cases.json \
  --baseline examples/baseline_v1.json
pytest -q
```

Expect a context, grounded extractive answer, lexical pass, `verdict: PASS`, and a
green suite including `test_query_path_returns_context_answer_and_offline_evaluation`.

## Limitations

- Hashing embeddings favor deterministic lexical retrieval over semantic recall.
- The rate limiter is per process, not shared across replicas.
- The optional LLM judge is a plugin. Default CI does not validate a provider,
  model version, calibration set, or cost budget.
