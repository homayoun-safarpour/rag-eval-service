from __future__ import annotations

import json

from fastapi.testclient import TestClient

from rag_eval_service.app import Settings, create_app
from rag_eval_service.generate import ExtractiveGenerator
from rag_eval_service.ingest import SourceDocument, chunk_text, ingest_documents
from rag_eval_service.judges import LexicalJudge
from rag_eval_service.qdrant import QdrantVectorStore
from rag_eval_service.store import InMemoryVectorStore, SearchResult

TEST_AUTH_TOKEN = "test-token"


def test_ingestion_chunks_documents_and_preserves_source_metadata():
    store = InMemoryVectorStore()
    text = " ".join(f"token-{index}" for index in range(30))
    summary = ingest_documents(
        store,
        [SourceDocument("guide", text, {"url": "https://example.test/guide"})],
        chunk_size=10,
        chunk_overlap=2,
    )
    assert summary.sources == 1
    assert summary.chunks == 4
    assert store.count() == 4
    assert store.metadata["guide::chunk-0000"]["source_id"] == "guide"


def test_chunking_rejects_overlap_that_could_stall():
    try:
        chunk_text("some words for a chunk", size=10, overlap=10)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("invalid overlap should fail")


def test_query_path_returns_context_answer_and_offline_evaluation():
    store = InMemoryVectorStore()
    api = create_app(Settings(api_key=TEST_AUTH_TOKEN), store)
    client = TestClient(api)
    ingest = client.post(
        "/ingest",
        headers={"x-api-key": TEST_AUTH_TOKEN},
        json={
            "documents": [
                {
                    "id": "shipping",
                    "text": "Express shipping arrives in two business days.",
                    "metadata": {"source": "handbook"},
                }
            ],
            "chunk_size": 20,
            "chunk_overlap": 0,
        },
    )
    assert ingest.status_code == 200

    response = client.post(
        "/query",
        headers={"x-api-key": TEST_AUTH_TOKEN},
        json={"query": "When does express shipping arrive?", "top_k": 1},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["contexts"][0]["metadata"]["source"] == "handbook"
    assert body["generator"] == "extractive-v1"
    assert body["evaluation"]["backend"] == "lexical-v1"
    assert body["evaluation"]["passed"] is True


def test_api_key_protects_mutating_and_query_routes():
    client = TestClient(
        create_app(Settings(api_key=TEST_AUTH_TOKEN), InMemoryVectorStore())
    )
    assert client.post("/ingest", json={"documents": []}).status_code == 401
    assert client.post("/query", json={"query": "test"}).status_code == 401
    assert client.get("/health").status_code == 200


def test_request_size_limit_fails_before_handler():
    client = TestClient(
        create_app(Settings(max_request_bytes=20), InMemoryVectorStore())
    )
    response = client.post("/query", json={"query": "this body exceeds twenty bytes"})
    assert response.status_code == 413
    assert "x-request-id" in response.headers


def test_rate_limit_returns_429():
    client = TestClient(
        create_app(Settings(requests_per_minute=1), InMemoryVectorStore())
    )
    assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 429


def test_http_requests_emit_structured_log_and_request_id(caplog):
    caplog.set_level("INFO", logger="rag_eval_service")
    client = TestClient(create_app(Settings(), InMemoryVectorStore()))
    response = client.get("/health", headers={"x-request-id": "test-request-123"})
    assert response.headers["x-request-id"] == "test-request-123"
    records = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "rag_eval_service"
    ]
    assert records[-1]["event"] == "http_request"
    assert records[-1]["request_id"] == "test-request-123"
    assert records[-1]["status"] == 200


def test_ready_reports_backend_and_document_count():
    store = InMemoryVectorStore()
    store.upsert("one", "one document")
    response = TestClient(create_app(Settings(), store)).get("/ready")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "backend": "memory",
        "documents": 1,
    }


def test_optional_llm_path_fails_explicitly_without_configuration(monkeypatch):
    monkeypatch.delenv("RAG_LLM_API_KEY", raising=False)
    monkeypatch.delenv("RAG_LLM_MODEL", raising=False)
    store = InMemoryVectorStore()
    store.upsert("one", "A grounded context.")
    response = TestClient(create_app(Settings(), store)).post(
        "/query",
        json={"query": "What is grounded?", "generator": "llm"},
    )
    assert response.status_code == 503
    assert "RAG_LLM_API_KEY" in response.json()["detail"]


def test_offline_generator_and_judge_need_no_api_key():
    contexts = [
        SearchResult(
            "refund",
            "Unused products qualify for a refund within thirty days.",
            0.9,
        )
    ]
    answer = ExtractiveGenerator().generate("When can I get a refund?", contexts)
    result = LexicalJudge().evaluate("When can I get a refund?", answer, contexts)
    assert "refund" in answer
    assert result.score == 1.0
    assert result.passed is True


def test_qdrant_adapter_persists_payload_and_maps_search_results():
    calls: list[tuple[str, str, dict | None]] = []

    def fake_transport(method: str, url: str, payload: dict | None):
        calls.append((method, url, payload))
        if method == "GET":
            return {"result": {"status": "green"}}
        if url.endswith("/points/search"):
            return {
                "result": [
                    {
                        "id": "uuid",
                        "score": 0.88,
                        "payload": {
                            "doc_id": "policy::chunk-0000",
                            "text": "Refunds are available.",
                            "metadata": {"source_id": "policy"},
                        },
                    }
                ]
            }
        return {"status": "ok"}

    store = QdrantVectorStore(
        url="http://qdrant.test:6333",
        collection="test_docs",
        transport=fake_transport,
    )
    store.upsert("policy::chunk-0000", "Refunds are available.", {"source_id": "policy"})
    result = store.search("refund", k=1)
    assert result[0].doc_id == "policy::chunk-0000"
    assert result[0].score == 0.88
    upsert_payload = next(
        payload
        for method, url, payload in calls
        if method == "PUT" and "points?wait=true" in url
    )
    assert upsert_payload is not None
    assert len(upsert_payload["points"][0]["vector"]) == 384
    assert upsert_payload["points"][0]["payload"]["doc_id"] == "policy::chunk-0000"
