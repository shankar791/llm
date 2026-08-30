"""
Unit and regression test suite for provider-agnostic LLM foundation (ai.llm).
Uses MockLLMProvider and injected MockTransport for 100% offline, deterministic execution.
"""
from __future__ import annotations
import json
import os
import pytest
import httpx

from ai.llm.base import LLMResponse, LLMProvider
from ai.llm.config import LLMConfig
from ai.llm.errors import (
    LLMError,
    LLMConfigurationError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMNetworkError,
    LLMResponseError,
)
from ai.llm.provider import OpenAICompatibleProvider, MockLLMProvider, get_llm_provider


# ============================================================
# 1. Config & Secrecy Tests
# ============================================================

def test_llm_config_defaults():
    cfg = LLMConfig()
    assert cfg.provider == "openai_compatible"
    assert cfg.model == "qwen/qwen3-14b:free"
    assert cfg.base_url == "https://openrouter.ai/api/v1"
    assert cfg.timeout == 30.0
    assert cfg.max_retries == 3
    assert cfg.api_key is None


def test_llm_config_api_key_masked_in_repr():
    cfg = LLMConfig(api_key="sk-secret-1234567890abcdef")
    repr_str = repr(cfg)
    assert "sk-secret-1234567890abcdef" not in repr_str
    assert "***" in repr_str


def test_llm_config_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_MODEL", "anthropic/claude-3.5-sonnet")
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1/")
    monkeypatch.setenv("LLM_API_KEY", "sk-or-test-key")
    monkeypatch.setenv("LLM_TIMEOUT", "45.0")
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")

    cfg = LLMConfig.from_env()
    assert cfg.provider == "openrouter"
    assert cfg.model == "anthropic/claude-3.5-sonnet"
    assert cfg.base_url == "https://openrouter.ai/api/v1"
    assert cfg.api_key == "sk-or-test-key"
    assert cfg.timeout == 45.0
    assert cfg.max_retries == 5


# ============================================================
# 2. LLMResponse & Structured Output Tests
# ============================================================

def test_llm_response_json_parsing():
    raw_json = '{"task": "change", "target": "construction", "confidence": 0.92}'
    resp = LLMResponse(
        content=raw_json,
        model="gpt-4o-mini",
        provider="openai_compatible",
        latency_ms=145.2,
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        finish_reason="stop",
    )

    parsed = resp.json()
    assert parsed["task"] == "change"
    assert parsed["target"] == "construction"
    assert parsed["confidence"] == 0.92
    assert resp.latency_ms == 145.2
    assert resp.usage["total_tokens"] == 30


def test_llm_response_invalid_json_raises_value_error():
    resp = LLMResponse(
        content="This is plain text, not JSON.",
        model="gpt-4o-mini",
        provider="openai_compatible",
        latency_ms=100.0,
    )
    with pytest.raises(ValueError, match="LLM output is not valid JSON"):
        resp.json()


# ============================================================
# 3. MockLLMProvider Tests
# ============================================================

def test_mock_llm_provider_sync_and_async():
    mock_p = MockLLMProvider(default_response='{"task": "vqa", "answer": "cropland"}')

    # Sync
    resp_sync = mock_p.generate_sync([{"role": "user", "content": "What is here?"}])
    assert resp_sync.model == "mock-llm-v1"
    assert resp_sync.provider == "mock"
    assert resp_sync.json()["task"] == "vqa"

    # Async
    import asyncio
    resp_async = asyncio.run(mock_p.generate([{"role": "user", "content": "What is here?"}]))
    assert resp_async.json()["answer"] == "cropland"
    assert len(mock_p.call_history) == 2


def test_mock_llm_provider_custom_handler():
    def custom_echo(messages, **kwargs):
        user_msg = messages[-1]["content"]
        return json.dumps({"echo": user_msg})

    mock_p = MockLLMProvider(custom_handler=custom_echo)
    resp = mock_p.generate_sync([{"role": "user", "content": "Hello SatQuery"}])
    assert resp.json()["echo"] == "Hello SatQuery"


# ============================================================
# 4. Factory Resolution Tests
# ============================================================

def test_get_llm_provider_factory():
    cfg_mock = LLMConfig(provider="mock")
    provider = get_llm_provider(cfg_mock)
    assert isinstance(provider, MockLLMProvider)

    cfg_openai = LLMConfig(provider="openai_compatible", api_key="test-key")
    provider_openai = get_llm_provider(cfg_openai)
    assert isinstance(provider_openai, OpenAICompatibleProvider)

    cfg_invalid = LLMConfig(provider="unsupported_vendor_xyz")
    with pytest.raises(LLMConfigurationError, match="Unsupported LLM provider"):
        get_llm_provider(cfg_invalid)


# ============================================================
# 5. HTTP Client & Mock Transport Tests
# ============================================================

def test_openai_compatible_successful_generation():
    """Test successful completion using httpx mock transport."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        assert body["model"] == "qwen/qwen3-14b:free"
        assert body["messages"][0]["content"] == "Classify query"
        assert body["response_format"] == {"type": "json_object"}

        response_payload = {
            "id": "chatcmpl-test-123",
            "model": "qwen/qwen3-14b:free",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '{"intent": "change_detection", "confidence": 0.95}',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 25, "completion_tokens": 15, "total_tokens": 40},
        }
        return httpx.Response(200, json=response_payload)

    transport = httpx.MockTransport(mock_handler)
    cfg = LLMConfig(provider="openai_compatible", api_key="sk-mock-key")
    provider = OpenAICompatibleProvider(cfg, transport=transport)

    resp = provider.generate_sync(
        messages=[{"role": "user", "content": "Classify query"}],
        response_format={"type": "json_object"},
    )

    assert resp.content == '{"intent": "change_detection", "confidence": 0.95}'
    assert resp.json()["intent"] == "change_detection"
    assert resp.usage["total_tokens"] == 40
    assert resp.latency_ms >= 0


def test_openai_compatible_auth_error_not_retried():
    """Test HTTP 401 raises LLMAuthenticationError immediately without retrying."""
    call_count = 0

    def mock_auth_fail(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(401, json={"error": {"message": "Invalid API key provided"}})

    transport = httpx.MockTransport(mock_auth_fail)
    cfg = LLMConfig(provider="openai_compatible", api_key="invalid-key", max_retries=3)
    provider = OpenAICompatibleProvider(cfg, transport=transport)

    with pytest.raises(LLMAuthenticationError) as exc_info:
        provider.generate_sync([{"role": "user", "content": "test"}])

    assert exc_info.value.status_code == 401
    assert "Invalid API key" in str(exc_info.value)
    assert call_count == 1  # Must not retry 401


def test_openai_compatible_rate_limit_and_transient_retry():
    """Test HTTP 429 rate limit triggers retries with backoff."""
    call_count = 0

    def mock_transient_then_succeed(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})
        return httpx.Response(200, json={
            "model": "gpt-4o-mini",
            "choices": [{"message": {"content": '{"success": true}'}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 10},
        })

    transport = httpx.MockTransport(mock_transient_then_succeed)
    cfg = LLMConfig(provider="openai_compatible", api_key="valid-key", max_retries=3)
    provider = OpenAICompatibleProvider(cfg, transport=transport)

    resp = provider.generate_sync([{"role": "user", "content": "test"}])
    assert resp.json()["success"] is True
    assert call_count == 3  # Succeeded on 3rd attempt


def test_openai_compatible_malformed_response():
    """Test handling of unexpected/malformed response payload structure."""
    def mock_bad_body(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected_field": []})

    transport = httpx.MockTransport(mock_bad_body)
    cfg = LLMConfig(provider="openai_compatible", api_key="valid-key")
    provider = OpenAICompatibleProvider(cfg, transport=transport)

    with pytest.raises(LLMResponseError, match="no choices"):
        provider.generate_sync([{"role": "user", "content": "test"}])


# ============================================================
# 6. Optional Live Integration Test (Gated)
# ============================================================

@pytest.mark.skipif(
    os.environ.get("LLM_INTEGRATION_TEST", "false").lower() != "true",
    reason="Live integration test requires LLM_INTEGRATION_TEST=true and valid LLM_API_KEY",
)
def test_live_llm_integration():
    """Live call to configured LLM endpoint only when explicitly enabled."""
    cfg = LLMConfig.from_env()
    assert cfg.api_key is not None, "LLM_API_KEY required for live test"
    provider = get_llm_provider(cfg)
    resp = provider.generate_sync(
        messages=[{"role": "user", "content": "Respond with JSON: {\"ping\": \"pong\"}"}],
        response_format={"type": "json_object"},
    )
    assert resp.json().get("ping") == "pong"
