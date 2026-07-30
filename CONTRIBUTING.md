# Contributing

## Local checks

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -q
rag-eval check --corpus examples/corpus.json --cases examples/cases.json \
  --baseline examples/baseline_v1.json
```

Changes to retrieval behavior must include a labeled case and a regenerated
benchmark artifact. Do not widen baseline tolerance to make a regression pass.

## Extension contracts

Small first contributions:

1. Add a `VectorStoreProtocol` adapter and contract test for another local store.
2. Add a generator or judge plugin with a deterministic fake-provider test.
3. Add a benchmark corpus loader that records source URL, version, and license.
4. Add Redis-backed rate limiting as an opt-in deployment adapter.

Keep the default suite offline. Optional service and model tests must skip cleanly
when credentials or containers are absent.

## Pull requests

- Name the claim test that would fail if the change were false.
- Include real command output when a public metric changes.
- Update limitations when a capability remains partial.
- Do not commit credentials, local environment files, provider responses, or user
  documents.
