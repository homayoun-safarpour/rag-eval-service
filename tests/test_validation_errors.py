"""Exercise diagnostic rendering without serving or sending special input."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from rag_eval_service.app import Settings, create_app
from rag_eval_service.store import InMemoryVectorStore


def _app():
    return create_app(Settings(allow_no_auth=True), InMemoryVectorStore())


def _render(error):
    handler = _app().exception_handlers[RequestValidationError]
    return asyncio.run(handler(None, RequestValidationError([error])))


def _error(value=None):
    return {
        "type": "value_error",
        "loc": ("body", "field"),
        "msg": "Invalid value",
        "input": value,
        "ctx": {"example": "not part of the response"},
    }


@pytest.mark.parametrize("value", [None, 7, {"unexpected": "value"}])
def test_validation_detail_omits_raw_input_and_context(value):
    response = _render(_error(value))
    assert response.status_code == 422
    assert json.loads(response.body) == {
        "detail": [{"type": "value_error", "loc": ["body", "field"], "msg": "Invalid value"}]
    }


def test_validation_detail_can_render_non_json_input():
    response = _render(_error(object()))
    assert response.status_code == 422
    assert set(json.loads(response.body)["detail"][0]) == {"type", "loc", "msg"}


@pytest.mark.parametrize("query", [{"query": 7}, {"query": "Readiness", "top_k": 0}])
def test_ordinary_invalid_request_keeps_status_and_request_id(query):
    response = TestClient(_app()).post(
        "/query", json=query, headers={"x-request-id": "validation-control"}
    )
    assert response.status_code == 422
    assert response.headers["x-request-id"] == "validation-control"


@pytest.mark.parametrize(
    "text", ["café", "λ 🙂", "".join(map(chr, [0x7F, 0x80, 0xD7FF, 0xE000, 0x10FFFF]))]
)
def test_valid_unicode_diagnostic_text_is_preserved(text):
    error = _error()
    error["msg"] = text
    response = _render(error)
    assert response.status_code == 422
    assert json.loads(response.body.decode("utf-8"))["detail"][0]["msg"] == text


@pytest.mark.parametrize("boundary", [0xD800, 0xDFFF])
@pytest.mark.parametrize("field", ["type", "loc", "msg"])
def test_unencodable_diagnostic_text_is_normalized(boundary, field):
    error = _error()
    if field == "loc":
        error[field] = ("body", "field" + chr(boundary))
    else:
        error[field] = "invalid text " + chr(boundary)
    response = _render(error)
    assert response.status_code == 422
    detail = json.loads(response.body.decode("utf-8"))["detail"][0]
    assert detail[field] == (["body", "field?"] if field == "loc" else "invalid text ?")
