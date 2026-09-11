# Reliability card — rag-eval-service

| Field | Value |
| --- | --- |
| **Job** | Fail-closed API + CI contracts for RAG retrieval quality and corpus identity; drop-in hidden RAG for other projects |
| **Primary metrics** | hit@k, MRR, context precision, grounding; `corpus_sha256` floors; hidden-ask exit 0/1/2 |
| **Named failures** | `CORPUS_DRIFT`, regression below frozen floors (exit `2`), injection blocked (hidden-ask exit `1`) |
| **Claim** | HTTP health is not retrieval health; corpus swaps invalidate old scores; retrieved chunks cannot override the task |
| **Not claimed** | Replaces production vector DB ops; default path needs no paid LLM; prompt filters catch every injection |

## Field alignment

Matches AI-first QA language (golden / frozen floors, drift, offline eval). Pairs with `judge-drift-sentinel` when an optional LLM judge is used online.
