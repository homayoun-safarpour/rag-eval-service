# Hidden RAG pack

Drop this JSON into another project. Import `HiddenRAG`, or call `rag-eval hidden-ask`.
No HTTP. Retrieved text is evidence. Instruction-shaped chunks are dropped.

| Category | Use | Example query |
| --- | --- | --- |
| `support` | product FAQ | When are refunds issued? |
| `runbook` | ops steps | What do I do when ready returns 503? |
| `policy` | declared rules | May CI call a paid LLM judge? |
| `docs` | library / CLI | How do other projects call hidden RAG? |

Exit `0` grounded, `1` injection blocked, `2` no category match.

```bash
python scripts/run_hidden_rag.py
rag-eval hidden-ask --pack examples/hidden_rag/pack.json \
  --query "When are refunds issued?" --category support
```
