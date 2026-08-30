"""
Unit tests for Step 12C Resilient Multi-Model OpenRouter Vision Layer.
Tests:
- Multi-model configuration & candidate resolution
- Task-level routing (VQA, Caption, Grounding)
- Transient error fallback (429, 500, timeout)
- Account-level rate limit fail-fast (free-models-per-day)
- Grounding unsupported fallback & strict box validation
- Observability metadata in VisionResponse & Tool results
"""
from __future__ import annotations
import json
import pytest
import numpy as np
from PIL import Image

from ai.vision.base import GroundingBox, GroundingResult, VisionResponse
from ai.vision.config import (
    DEFAULT_VISION_PRIMARY_MODEL,
    DEFAULT_VISION_SECONDARY_MODEL,
    DEFAULT_VISION_TERTIARY_MODEL,
    VisionConfig,
)
from ai.vision.errors import (
    VisionAuthenticationError,
    VisionError,
    VisionRateLimitError,
    VisionResponseError,
    VisionTimeoutError,
)
from ai.vision.openrouter_qwen import (
    OpenRouterVisionProvider,
    _is_account_level_rate_limit,
    _parse_and_validate_grounding,
)
from tools.captioning import CaptioningTool
from tools.grounding import GroundingTool
from tools.vqa import VQATool


def _create_dummy_image(w: int = 100, h: int = 100) -> Image.Image:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:50, :50] = [255, 0, 0]
    return Image.fromarray(arr)


# ============================================================
# 1. Configuration & Task Routing Tests
# ============================================================

def test_vision_config_defaults():
    cfg = VisionConfig()
    assert cfg.primary_model == "google/gemma-4-26b-a4b-it:free"
    assert cfg.secondary_model == "google/gemma-4-31b-it:free"
    assert cfg.tertiary_model == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

    # Task defaults
    assert cfg.vqa_model == "google/gemma-4-26b-a4b-it:free"
    assert cfg.caption_model == "google/gemma-4-26b-a4b-it:free"
    assert cfg.ground_model == "google/gemma-4-31b-it:free"

    # Candidate lists
    assert cfg.get_candidate_models_for_task("vqa") == [
        "google/gemma-4-26b-a4b-it:free",
        "google/gemma-4-31b-it:free",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    ]
    assert cfg.get_candidate_models_for_task("caption") == [
        "google/gemma-4-26b-a4b-it:free",
        "google/gemma-4-31b-it:free",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    ]
    assert cfg.get_candidate_models_for_task("ground") == [
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    ]


def test_vision_config_custom_env(monkeypatch):
    monkeypatch.setenv("VISION_PRIMARY_MODEL", "google/gemma-4-26b-a4b-it:free")
    monkeypatch.setenv("VISION_SECONDARY_MODEL", "google/gemma-4-31b-it:free")
    monkeypatch.setenv("VISION_TERTIARY_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")
    monkeypatch.setenv("VISION_GROUND_MODEL", "google/gemma-4-31b-it:free")
    monkeypatch.setenv("VISION_GROUND_FALLBACKS", "google/gemma-4-26b-a4b-it:free,nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")

    cfg = VisionConfig.from_env()
    assert cfg.ground_model == "google/gemma-4-31b-it:free"
    assert cfg.get_candidate_models_for_task("ground") == [
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    ]


# ============================================================
# 2. Account-Level Rate Limit Tests
# ============================================================

def test_account_level_rate_limit_detection():
    assert _is_account_level_rate_limit("Rate limit exceeded for free-models-per-day") is True
    assert _is_account_level_rate_limit("Daily request limit reached for free models") is True
    assert _is_account_level_rate_limit("Standard transient 429: concurrency limit") is False
    assert _is_account_level_rate_limit("") is False


def test_account_level_rate_limit_fails_fast(monkeypatch):
    cfg = VisionConfig(api_key="sk-test-key", max_retries=0)
    provider = OpenRouterVisionProvider(config=cfg)

    call_count = 0

    def mock_single_request(payload, model_name):
        nonlocal call_count
        call_count += 1
        raise VisionRateLimitError(
            "Account daily free limit exhausted: free-models-per-day exceeded",
            is_account_limit=True,
        )

    monkeypatch.setattr(provider, "_execute_single_request", mock_single_request)

    with pytest.raises(VisionRateLimitError) as exc_info:
        provider.analyze_image_sync(_create_dummy_image(), prompt="Test", task="vqa")

    assert exc_info.value.is_account_limit is True
    assert call_count == 1


# ============================================================
# 3. Multi-Model Transient Failover Tests
# ============================================================

def test_vqa_transient_429_fallback_to_secondary(monkeypatch):
    cfg = VisionConfig(api_key="sk-test-key", max_retries=0)
    provider = OpenRouterVisionProvider(config=cfg)

    attempted = []

    def mock_single_request(payload, model_name):
        attempted.append(model_name)
        if model_name == cfg.primary_model:
            raise VisionRateLimitError(
                "Upstream provider concurrency limit (429)",
                is_account_limit=False,
            )
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "Runway and hangars visible."}}
            ]
        }

    monkeypatch.setattr(provider, "_execute_single_request", mock_single_request)

    resp = provider.analyze_image_sync(_create_dummy_image(), prompt="What is visible?", task="vqa")

    assert resp.selected_model == cfg.secondary_model
    assert resp.attempted_models == [cfg.primary_model, cfg.secondary_model]
    assert resp.fallback_used is True
    assert resp.fallback_reason == "upstream_rate_limit"
    assert "Runway and hangars" in resp.text


def test_caption_500_server_error_fallback_to_tertiary(monkeypatch):
    cfg = VisionConfig(api_key="sk-test-key", max_retries=0)
    provider = OpenRouterVisionProvider(config=cfg)

    attempted = []

    def mock_single_request(payload, model_name):
        attempted.append(model_name)
        if model_name == cfg.primary_model:
            raise VisionResponseError("502 Bad Gateway", status_code=502)
        elif model_name == cfg.secondary_model:
            raise VisionTimeoutError("Timed out waiting for upstream")
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "Scene of dense urban housing."}}
            ]
        }

    monkeypatch.setattr(provider, "_execute_single_request", mock_single_request)

    resp = provider.analyze_image_sync(_create_dummy_image(), prompt="Describe scene", task="caption")

    assert resp.selected_model == cfg.tertiary_model
    assert resp.attempted_models == [cfg.primary_model, cfg.secondary_model, cfg.tertiary_model]
    assert resp.fallback_used is True
    assert resp.fallback_reason in {"provider_timeout", "upstream_server_error"}
    assert "dense urban housing" in resp.text


def test_auth_error_does_not_fallback(monkeypatch):
    cfg = VisionConfig(api_key="sk-test-key", max_retries=0)
    provider = OpenRouterVisionProvider(config=cfg)

    call_count = 0

    def mock_single_request(payload, model_name):
        nonlocal call_count
        call_count += 1
        raise VisionAuthenticationError("HTTP 401 Unauthorized")

    monkeypatch.setattr(provider, "_execute_single_request", mock_single_request)

    with pytest.raises(VisionAuthenticationError):
        provider.analyze_image_sync(_create_dummy_image(), prompt="Test", task="vqa")

    assert call_count == 1


# ============================================================
# 4. Grounding Structured Validation & Fallback Tests
# ============================================================

def test_grounding_fallback_when_primary_returns_text_only(monkeypatch):
    cfg = VisionConfig(api_key="sk-test-key", max_retries=0)
    provider = OpenRouterVisionProvider(config=cfg)

    attempted = []

    def mock_single_request(payload, model_name):
        attempted.append(model_name)
        if model_name == cfg.ground_model:
            return {
                "choices": [
                    {"message": {"role": "assistant", "content": "I see several buildings on the east side."}}
                ]
            }
        boxes_json = json.dumps({
            "objects": [
                {"label": "building", "box": [0.1, 0.2, 0.4, 0.5]},
                {"label": "building", "box": [0.6, 0.7, 0.8, 0.9]},
            ]
        })
        return {
            "choices": [
                {"message": {"role": "assistant", "content": boxes_json}}
            ]
        }

    monkeypatch.setattr(provider, "_execute_single_request", mock_single_request)

    resp = provider.analyze_image_sync(_create_dummy_image(), prompt="Locate buildings", task="ground")

    assert resp.selected_model == cfg.primary_model
    assert resp.attempted_models == [cfg.ground_model, cfg.primary_model]
    assert resp.fallback_used is True
    assert resp.fallback_reason == "grounding_unsupported"
    assert resp.grounding is not None
    assert len(resp.grounding.objects) == 2
    assert resp.grounding.objects[0].label == "building"


def test_grounding_validation_rejects_reversed_coordinates():
    valid_text = json.dumps({"objects": [{"label": "tank", "box": [0.1, 0.2, 0.3, 0.4]}]})
    res = _parse_and_validate_grounding(valid_text, 100, 100)
    assert res is not None
    assert len(res.objects) == 1

    invalid_text = json.dumps({"objects": [{"label": "tank", "box": [0.8, 0.2, 0.3, 0.4]}]})
    res2 = _parse_and_validate_grounding(invalid_text, 100, 100)
    assert res2 is None


# ============================================================
# 5. Tool Integration & Metadata Observability
# ============================================================

def test_vqa_tool_observability_metadata():
    custom_resp = VisionResponse(
        text="Port cranes detected.",
        selected_model="google/gemma-4-31b-it:free",
        attempted_models=["google/gemma-4-26b-a4b-it:free", "google/gemma-4-31b-it:free"],
        fallback_used=True,
        fallback_reason="upstream_rate_limit",
        latency_ms=850.0,
        provider="openrouter",
        model="google/gemma-4-31b-it:free",
    )

    class MockObservableVisionProvider:
        def analyze_image_sync(self, *args, **kwargs):
            return custom_resp

    tool = VQATool(mode="real", vision_provider=MockObservableVisionProvider())
    result = tool.run(query="Identify equipment", image_bytes=_create_dummy_image(), mode="real")

    meta = result["metadata"]
    assert meta["selected_model"] == "google/gemma-4-31b-it:free"
    assert meta["attempted_models"] == ["google/gemma-4-26b-a4b-it:free", "google/gemma-4-31b-it:free"]
    assert meta["fallback_used"] is True
    assert meta["fallback_reason"] == "upstream_rate_limit"
    assert meta["latency_ms"] == 850.0


# ============================================================
# 6. Live Matrix Probe Rate-Limit Distinction Tests
# ============================================================

def test_live_matrix_probe_upstream_429_returns_rate_limited(monkeypatch):
    """Test that upstream provider 429 records RATE_LIMITED without claiming model failure."""
    from tests.evaluation.live_vision_model_matrix import run_model_probe

    upstream_429_err = json.dumps({
        "error": {
            "message": "Provider returned error",
            "code": 429,
            "metadata": {
                "raw": "google/gemma-4-26b-a4b-it:free is temporarily rate-limited upstream.",
                "provider_name": "Google AI Studio",
                "provider_error_code": "429"
            }
        }
    })

    def mock_http(payload, api_key, timeout=45.0):
        return 429, {}, 120.0, upstream_429_err

    monkeypatch.setattr("tests.evaluation.live_vision_model_matrix._execute_http", mock_http)

    res = run_model_probe("google/gemma-4-26b-a4b-it:free", "sk-test", "data:fake1", "data:fake2", 100, 100)

    assert res["vqa"]["status"] == "RATE_LIMITED"
    assert res["vqa"]["error_type"] == "429"
    assert res["vqa"]["provider_name"] == "Google AI Studio"
    assert res["account_quota_exhausted"] is False
    assert res["overall_status"] == "RATE_LIMITED"


def test_live_matrix_probe_account_quota_sets_exhausted_flag(monkeypatch):
    """Test that account quota exhaustion sets account_quota_exhausted=True to halt further probes."""
    from tests.evaluation.live_vision_model_matrix import run_model_probe

    account_limit_err = json.dumps({
        "error": {
            "message": "Rate limit exceeded for free-models-per-day",
            "code": 429,
        }
    })

    def mock_http(payload, api_key, timeout=45.0):
        return 429, {}, 95.0, account_limit_err

    monkeypatch.setattr("tests.evaluation.live_vision_model_matrix._execute_http", mock_http)

    res = run_model_probe("google/gemma-4-26b-a4b-it:free", "sk-test", "data:fake1", "data:fake2", 100, 100)

    assert res["vqa"]["status"] == "ACCOUNT_RATE_LIMIT"
    assert res["account_quota_exhausted"] is True
    assert res["overall_status"] == "ACCOUNT_RATE_LIMIT"

