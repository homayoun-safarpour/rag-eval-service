# Reliability card: rag-eval-service

| Field | Value |
| --- | --- |
| **Job** | Fail-closed API + CI contracts for RAG retrieval quality and corpus identity; drop-in hidden RAG for other projects |
| **Primary metrics** | hit@k, MRR, context precision, grounding; `corpus_sha256` floors; hidden-ask exit 0/1/2 |
| **Named failures** | `CORPUS_DRIFT`, regression below frozen floors (exit `2`), injection blocked (hidden-ask exit `1`) |
| **Claim** | HTTP health is not retrieval health; corpus swaps invalidate old scores; retrieved chunks cannot override the task |
| **Not claimed** | Replaces production vector DB ops; default path needs no paid LLM; prompt filters catch every injection; the scan catches task tampering (a chunk that quietly changes a fact, such as a refund limit, with no instruction words), see "Beyond Instruction Override: Benchmarking Indirect Prompt Injection with Task-Tampering Attacks", arXiv:2611.07640 |

## Field alignment

Matches AI-first QA language (golden / frozen floors, drift, offline eval). Pairs with `judge-drift-sentinel` when an optional LLM judge is used online.

## Known limit: task tampering

The injection scan looks for instruction-shaped text. A chunk that changes a fact the answer depends on, for example a refund limit, and uses no instruction words is not blocked and can reach the answer. This attack class is described in "Beyond Instruction Override: Benchmarking Indirect Prompt Injection with Task-Tampering Attacks" (arXiv:2611.07640). `tests/test_hidden_rag.py::test_claim_limit_task_tampering_not_caught` pins the limit.
