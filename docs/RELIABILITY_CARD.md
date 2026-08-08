# Reliability card — rag-eval-service

| Field | Value |
| --- | --- |
| **Job** | Fail-closed API + CI contracts for RAG retrieval quality and corpus identity |
| **Primary metrics** | hit@k, MRR, context precision, grounding; `corpus_sha256` floors |
| **Named failures** | `CORPUS_DRIFT`, regression below frozen floors (exit `2`) |
| **Claim** | HTTP health is not retrieval health; corpus swaps invalidate old scores |
| **Not claimed** | Replaces production vector DB ops; default path needs no paid LLM |

## Field alignment

Matches AI-first QA language (golden / frozen floors, drift, offline eval). Pairs with `judge-drift-sentinel` when an optional LLM judge is used online.
