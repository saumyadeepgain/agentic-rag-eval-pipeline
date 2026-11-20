"""The LLM client: temperature 0, schema validation, the SQLite cache (and its
provider namespacing), retries, costs, and the real OpenAI SDK driven through an
``httpx.MockTransport`` so request and response formats are exercised offline."""

from __future__ import annotations

import json

import httpx
import numpy as np
import openai
import pytest
from conftest import ScriptedChat, make_cfg, run

from src.llm import LLMClient, LLMError, MissingAPIKeyError, OfflineBackend, OpenAIBackend, parse_json_output
from src.schemas import GradeDecision
from src.utils import ResponseCache

MESSAGES = [{"role": "system", "content": "Grade."}, {"role": "user", "content": "Question: q"}]
GRADE = {"reasoning": "fine", "relevant": True}


def structured(llm: LLMClient, **kwargs):
    args = {
        "task": "grade",
        "role": "agent",
        "messages": MESSAGES,
        "schema": GradeDecision,
        "payload": {},
        "max_tokens": 50,
    }
    args.update(kwargs)
    return run(llm.structured(**args))


def test_parse_json_output_tolerates_fences_and_surrounding_text():
    assert parse_json_output('```json\n{"reasoning": "r", "relevant": false}\n```', GradeDecision).relevant is False
    assert parse_json_output('Sure: {"reasoning": "r", "relevant": true} done', GradeDecision).relevant is True
    with pytest.raises(ValueError):
        parse_json_output("no object here", GradeDecision)


def test_cache_serves_repeated_calls_for_free(tmp_path):
    cfg = make_cfg(tmp_path, overrides={"cache.enabled": True})
    chat = ScriptedChat(defaults={"grade": lambda r: GRADE})
    llm = LLMClient(cfg, chat_backend=chat, embed_backend=OfflineBackend(), cache=ResponseCache(cfg.cache.path))
    first, usage1 = structured(llm)
    second, usage2 = structured(llm)
    assert first == second and len(chat.requests) == 1
    assert not usage1.cached and usage1.attempts == 1
    assert usage2.cached and usage2.attempts == 0 and usage2.prompt_tokens == usage1.prompt_tokens
    structured(llm, max_tokens=51)  # a different request is a different key
    assert len(chat.requests) == 2


def test_cache_is_namespaced_by_provider(tmp_path):
    """Offline heuristics must never be served to a real-model run (and vice versa)."""
    path = tmp_path / "shared.sqlite"
    offline_cfg = make_cfg(tmp_path)
    offline = LLMClient(
        offline_cfg, chat_backend=ScriptedChat(defaults={"grade": lambda r: GRADE}), cache=ResponseCache(path)
    )
    structured(offline)
    real_cfg = make_cfg(tmp_path, overrides={"llm.provider": "openai"})
    chat = ScriptedChat(defaults={"grade": lambda r: GRADE})
    real = LLMClient(real_cfg, chat_backend=chat, embed_backend=OfflineBackend(), cache=ResponseCache(path))
    _, usage = structured(real)
    assert not usage.cached and len(chat.requests) == 1


def test_every_call_uses_temperature_zero_and_states_the_schema(tmp_path):
    cfg = make_cfg(tmp_path)
    chat = ScriptedChat(defaults={"grade": lambda r: GRADE})
    structured(LLMClient(cfg, chat_backend=chat, embed_backend=OfflineBackend()))
    request = chat.requests[0]
    assert request.temperature == 0.0 and request.seed == cfg.llm.seed
    assert '"relevant"' in request.messages[0]["content"] and request.messages[0]["role"] == "system"


def test_non_transient_errors_are_not_retried(tmp_path):
    cfg = make_cfg(tmp_path, overrides={"llm.max_retries": 5})
    chat = ScriptedChat(defaults={"grade": lambda r: ValueError("bad request")})
    with pytest.raises(LLMError, match="bad request") as info:
        structured(LLMClient(cfg, chat_backend=chat, embed_backend=OfflineBackend()))
    assert len(chat.requests) == 1 and info.value.usage is not None


def test_costs_use_the_configured_prices(tmp_path):
    cfg = make_cfg(tmp_path, overrides={"llm.provider": "openai"})
    llm = LLMClient(cfg, chat_backend=ScriptedChat(), embed_backend=OfflineBackend())
    assert llm.cost("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.15 + 0.60)
    assert llm.cost("unknown-model", 10, 10) is None


def test_missing_api_key_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = make_cfg(tmp_path, overrides={"llm.provider": "openai"})
    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        LLMClient(cfg)


def test_embeddings_are_normalised_and_cached_per_text(tmp_path):
    cfg = make_cfg(tmp_path, overrides={"cache.enabled": True})
    llm = LLMClient(
        cfg, chat_backend=ScriptedChat(), embed_backend=OfflineBackend(), cache=ResponseCache(cfg.cache.path)
    )
    vectors, usage = run(llm.embed(["alpha beta", "gamma"]))
    assert vectors.shape == (2, 384) and np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
    again, usage2 = run(llm.embed(["gamma", "alpha beta"]))
    assert usage2.cached and np.allclose(again[0], vectors[1]) and np.allclose(again[1], vectors[0])


# ──────────────────────────────────────────────────────────────── real SDK, mocked HTTP
def completion(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49},
    }


def sdk_client(handler) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key="test-key",
        base_url="http://mock.local/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def test_openai_sdk_request_format_and_rate_limit_retry(tmp_path):
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if len(seen) == 1:
            return httpx.Response(429, json={"error": {"message": "slow down", "type": "rate_limit"}})
        return httpx.Response(200, json=completion(json.dumps(GRADE)))

    cfg = make_cfg(tmp_path, overrides={"llm.provider": "openai", "llm.max_retries": 3})
    backend = OpenAIBackend(cfg, client=sdk_client(handler))
    llm = LLMClient(cfg, chat_backend=backend, embed_backend=OfflineBackend())
    decision, usage = structured(llm)
    assert decision.relevant is True
    assert usage.attempts == 2 and (usage.prompt_tokens, usage.completion_tokens) == (42, 7)
    assert usage.cost_usd == pytest.approx((42 * 0.15 + 7 * 0.60) / 1e6)
    body = seen[-1]
    assert body["temperature"] == 0.0 and body["model"] == cfg.models.agent
    assert body["response_format"] == {"type": "json_object"} and body["seed"] == cfg.llm.seed


def test_openai_sdk_json_schema_mode(tmp_path):
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=completion(json.dumps(GRADE)))

    cfg = make_cfg(
        tmp_path, overrides={"llm.provider": "openai", "llm.response_format": "json_schema", "llm.seed": None}
    )
    llm = LLMClient(cfg, chat_backend=OpenAIBackend(cfg, client=sdk_client(handler)), embed_backend=OfflineBackend())
    structured(llm)
    assert seen[0]["response_format"]["type"] == "json_schema"
    assert seen[0]["response_format"]["json_schema"]["name"] == "GradeDecision" and "seed" not in seen[0]


def test_openai_sdk_client_errors_fail_fast(tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            400, json={"error": {"message": "unsupported parameter", "type": "invalid_request_error"}}
        )

    cfg = make_cfg(tmp_path, overrides={"llm.provider": "openai", "llm.max_retries": 4})
    llm = LLMClient(cfg, chat_backend=OpenAIBackend(cfg, client=sdk_client(handler)), embed_backend=OfflineBackend())
    with pytest.raises(LLMError, match="BadRequestError"):
        structured(llm)
    assert len(calls) == 1


def test_openai_sdk_embeddings(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        data = [{"object": "embedding", "index": i, "embedding": [3.0, 4.0]} for i in reversed(range(len(texts)))]
        return httpx.Response(
            200, json={"object": "list", "data": data, "model": "m", "usage": {"prompt_tokens": 5, "total_tokens": 5}}
        )

    cfg = make_cfg(tmp_path, overrides={"llm.provider": "openai"})
    backend = OpenAIBackend(cfg, client=sdk_client(handler))
    llm = LLMClient(cfg, chat_backend=backend, embed_backend=backend)
    vectors, usage = run(llm.embed(["a", "b"]))
    assert np.allclose(vectors, [[0.6, 0.8], [0.6, 0.8]]) and usage.prompt_tokens == 5
    assert usage.cost_usd == pytest.approx(5 * 0.02 / 1e6)
