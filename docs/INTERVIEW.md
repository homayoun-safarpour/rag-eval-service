# Interview gate — rag-eval-service

## Three questions

1. **Why pin `corpus_sha256` on the baseline?**  
   Without it, someone can swap the knowledge base, keep the same case IDs, and make hit@k look fine while answering from different content. Fingerprint mismatch forces an intentional re-freeze.

2. **What does the faithfulness proxy measure without an LLM?**  
   Lexical grounding: share of answer tokens that appear in retrieved context. It is a CI-cheap signal, not a substitute for a judge model on open-ended answers.

3. **How does this sit next to judge-reliability-kit and judge-drift-sentinel?**  
   This repo gates *retrieval* quality. Kit/sentinel gate *judge panel* agreement and drift. Same exit idea (`0`/`2`); different object under test. Wire both into agent-loop-engine as separate gates.

## Two-minute demo

```bash
git clone https://github.com/homayoun-safarpour/rag-eval-service
cd rag-eval-service
pip install -e ".[dev]"
rag-eval check --corpus examples/corpus.json --cases examples/cases.json \
  --baseline examples/baseline_v1.json
pytest -q
```

Expect: `verdict: PASS`, tests green including `test_check_refuses_when_corpus_fingerprint_mismatches`.

## One limitation

The default vector store is sparse bag-of-words. Production embeddings (dense models, pgvector, managed indexes) will need an adapter; the metric and gate contracts stay the same.
