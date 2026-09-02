"""
Unit tests verifying the 3-tier fallback cascade:
Tier 1: GeoChat
Tier 2: OpenRouter
Tier 3: Synthetic Output
With full step-by-step diagnostic journey reporting.
"""
import pytest
import numpy as np
from PIL import Image

from tools.vqa import VQATool
from tools.captioning import CaptioningTool
from tools.grounding import GroundingTool
from backend.agent import execute
from backend.rasterio_utils import RasterInput
from ai.vision.base import VisionResponse


def _make_dummy_image(w=100, h=100):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:50, :50] = [255, 0, 0]
    return Image.fromarray(arr)


def test_vqa_tier1_geochat_success(monkeypatch):
    """When GeoChat is live, Tier 1 succeeds directly."""
    dummy_resp = VisionResponse(
        text="Dense forest and agricultural parcels.",
        provider="geochat",
        model="GeoChat-7B",
        selected_model="GeoChat-7B",
        latency_ms=120.0,
    )

    class MockGeoChatProvider:
        def __init__(self, *args, **kwargs):
            pass

        def analyze_image_sync(self, *args, **kwargs):
            return dummy_resp

    monkeypatch.setattr("ai.vision.geochat.GeoChatVisionProvider", MockGeoChatProvider)

    tool = VQATool(mode="real")
    result = tool.run(query="What is visible?", image_bytes=_make_dummy_image(), mode="real")

    meta = result["metadata"]
    assert meta["active_tier"] == "geochat"
    assert meta["provider"] == "geochat"
    assert meta["model"] == "GeoChat-7B"
    assert meta["fallback_used"] is False
    assert len(meta["tier_journey"]) == 1
    assert meta["tier_journey"][0]["status"] == "success"


def test_vqa_tier2_openrouter_fallback(monkeypatch):
    """When GeoChat is down and OpenRouter is live, falls back to Tier 2 OpenRouter."""
    class FailingGeoChatProvider:
        def __init__(self, *args, **kwargs):
            pass

        def analyze_image_sync(self, *args, **kwargs):
            raise ConnectionError("GeoChat microservice unreachable at port 8000")

    dummy_or_resp = VisionResponse(
        text="Airport runways and commercial buildings.",
        provider="openrouter",
        model="google/gemma-4-26b-a4b-it:free",
        selected_model="google/gemma-4-26b-a4b-it:free",
        latency_ms=450.0,
    )

    class MockOpenRouterProvider:
        def __init__(self, *args, **kwargs):
            pass

        def analyze_image_sync(self, *args, **kwargs):
            return dummy_or_resp

    monkeypatch.setattr("ai.vision.geochat.GeoChatVisionProvider", FailingGeoChatProvider)
    monkeypatch.setattr("ai.vision.openrouter_qwen.OpenRouterVisionProvider", MockOpenRouterProvider)

    tool = VQATool(mode="real")
    result = tool.run(query="Describe structures", image_bytes=_make_dummy_image(), mode="real")

    meta = result["metadata"]
    assert meta["active_tier"] == "openrouter"
    assert meta["provider"] == "openrouter"
    assert meta["model"] == "google/gemma-4-26b-a4b-it:free"
    assert meta["fallback_used"] is True
    assert len(meta["tier_journey"]) == 2
    assert meta["tier_journey"][0]["status"] == "failed"
    assert meta["tier_journey"][0]["provider"] == "geochat"
    assert meta["tier_journey"][1]["status"] == "success"
    assert meta["tier_journey"][1]["provider"] == "openrouter"


def test_vqa_tier3_synthetic_fallback(monkeypatch):
    """When both GeoChat and OpenRouter are down, falls back to Tier 3 Synthetic Output."""
    class FailingGeoChatProvider:
        def __init__(self, *args, **kwargs):
            pass

        def analyze_image_sync(self, *args, **kwargs):
            raise ConnectionError("GeoChat connection refused")

    class FailingOpenRouterProvider:
        def __init__(self, *args, **kwargs):
            pass

        def analyze_image_sync(self, *args, **kwargs):
            raise RuntimeError("HTTP 429 Rate Limit - Account Quota Exhausted")

    monkeypatch.setattr("ai.vision.geochat.GeoChatVisionProvider", FailingGeoChatProvider)
    monkeypatch.setattr("ai.vision.openrouter_qwen.OpenRouterVisionProvider", FailingOpenRouterProvider)

    tool = VQATool(mode="real")
    result = tool.run(query="What land cover is visible?", image_bytes=_make_dummy_image(), mode="real")

    meta = result["metadata"]
    assert meta["active_tier"] == "synthetic"
    assert meta["provider"] == "synthetic"
    assert meta["model"] == "Synthetic Spectral Baseline"
    assert meta["fallback_used"] is True
    assert len(meta["tier_journey"]) == 3
    assert meta["tier_journey"][0]["status"] == "failed"
    assert meta["tier_journey"][1]["status"] == "failed"
    assert meta["tier_journey"][2]["status"] == "success"
    assert "Step 1:" in meta["fallback_reason"]
    assert "Step 2:" in meta["fallback_reason"]
    assert "Step 3:" in meta["fallback_reason"]


def test_caption_tier3_synthetic_metadata(monkeypatch):
    """CaptioningTool correctly sets synthetic metadata on fallback instead of claiming GeoChat."""
    class FailingGeoChatProvider:
        def __init__(self, *args, **kwargs):
            pass

        def analyze_image_sync(self, *args, **kwargs):
            raise ConnectionError("GeoChat down")

    class FailingOpenRouterProvider:
        def __init__(self, *args, **kwargs):
            pass

        def analyze_image_sync(self, *args, **kwargs):
            raise RuntimeError("OpenRouter down")

    monkeypatch.setattr("ai.vision.geochat.GeoChatVisionProvider", FailingGeoChatProvider)
    monkeypatch.setattr("ai.vision.openrouter_qwen.OpenRouterVisionProvider", FailingOpenRouterProvider)

    tool = CaptioningTool(mode="real")
    result = tool.run(image_bytes=_make_dummy_image(), mode="real")

    meta = result["metadata"]
    assert meta["active_tier"] == "synthetic"
    assert meta["provider"] == "synthetic"
    assert meta["model"] == "Synthetic Spectral Captioner"
    assert meta["fallback_used"] is True
    assert len(meta["tier_journey"]) == 3


def test_grounding_tier3_synthetic_metadata(monkeypatch):
    """GroundingTool correctly handles full fallback cascade to Tier 3."""
    class FailingGeoChatProvider:
        def __init__(self, *args, **kwargs):
            pass

        def analyze_image_sync(self, *args, **kwargs):
            raise ConnectionError("GeoChat down")

    class FailingOpenRouterProvider:
        def __init__(self, *args, **kwargs):
            pass

        def analyze_image_sync(self, *args, **kwargs):
            raise RuntimeError("OpenRouter down")

    monkeypatch.setattr("ai.vision.geochat.GeoChatVisionProvider", FailingGeoChatProvider)
    monkeypatch.setattr("ai.vision.openrouter_qwen.OpenRouterVisionProvider", FailingOpenRouterProvider)

    tool = GroundingTool(mode="real")
    result = tool.run(query="Locate water bodies", image_bytes=_make_dummy_image(), mode="real")

    meta = result["metadata"]
    assert meta["active_tier"] == "synthetic"
    assert meta["provider"] == "synthetic"
    assert meta["fallback_used"] is True
    assert len(meta["tier_journey"]) == 3


def test_backend_agent_execute_end_to_end_synthetic(monkeypatch):
    """End-to-end backend execute propagates active_tier and tier_journey when remote VLMs are down."""
    class FailingGeoChatProvider:
        def __init__(self, *args, **kwargs):
            pass

        def analyze_image_sync(self, *args, **kwargs):
            raise ConnectionError("GeoChat down")

    class FailingOpenRouterProvider:
        def __init__(self, *args, **kwargs):
            pass

        def analyze_image_sync(self, *args, **kwargs):
            raise RuntimeError("OpenRouter down")

    monkeypatch.setattr("ai.vision.geochat.GeoChatVisionProvider", FailingGeoChatProvider)
    monkeypatch.setattr("ai.vision.openrouter_qwen.OpenRouterVisionProvider", FailingOpenRouterProvider)

    img = _make_dummy_image()
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    raster = RasterInput(filename="test_opt.png", data=png_bytes)
    result = execute(query="What objects are in this image?", rasters=[raster])

    assert "answer" in result
    assert result["active_tier"] == "synthetic"
    assert len(result["tier_journey"]) >= 1
    assert "trace" in result
    trace_steps = result["trace"]["steps"]
    assert any(s.get("action") == "compose_answer" for s in trace_steps)
