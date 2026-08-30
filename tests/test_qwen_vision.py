"""
Unit tests for OpenRouterQwenVisionProvider and tool integration (T1_VQA, T2_Caption, T3_Ground).
Uses mocked HTTP requests to ensure 100% offline testing.
"""
from __future__ import annotations
import io
import json
import pytest
import numpy as np
from PIL import Image

from ai.vision.config import VisionConfig
from ai.vision.errors import (
    GroundingParseError,
    VisionAuthenticationError,
    VisionRateLimitError,
    VisionTimeoutError,
)
from ai.vision.mock import MockVisionProvider
from ai.vision.openrouter_qwen import OpenRouterQwenVisionProvider, _encode_image_to_data_url
from tools.captioning import CaptioningTool
from tools.grounding import GroundingTool
from tools.vqa import VQATool


# Helper: create dummy PIL image
def _create_test_image(width: int = 100, height: int = 100) -> Image.Image:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:50, :50] = [255, 0, 0]
    return Image.fromarray(arr)


def test_image_encoding_formats():
    img = _create_test_image(200, 150)
    
    # 1. PIL Image
    data_url_pil, (w1, h1) = _encode_image_to_data_url(img)
    assert data_url_pil.startswith("data:image/jpeg;base64,")
    assert (w1, h1) == (200, 150)

    # 2. Numpy array
    arr = np.array(img)
    data_url_arr, (w2, h2) = _encode_image_to_data_url(arr)
    assert data_url_arr.startswith("data:image/jpeg;base64,")
    assert (w2, h2) == (200, 150)

    # 3. Raw JPEG bytes
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    data_url_bytes, (w3, h3) = _encode_image_to_data_url(buf.getvalue())
    assert data_url_bytes.startswith("data:image/jpeg;base64,")
    assert (w3, h3) == (200, 150)


def test_qwen_provider_missing_api_key():
    cfg = VisionConfig(api_key=None)
    provider = OpenRouterQwenVisionProvider(config=cfg)
    with pytest.raises(VisionAuthenticationError) as exc_info:
        provider.analyze_image_sync(_create_test_image(), prompt="What is this?")
    assert "OPENROUTER_API_KEY is not set" in str(exc_info.value)


def test_qwen_provider_vqa_mocked_http(monkeypatch):
    mock_openrouter_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The image shows a commercial airport with aircraft docked at boarding gates.",
                }
            }
        ]
    }

    cfg = VisionConfig(
        api_key="sk-test-key",
        model="qwen/qwen-2.5-vl-7b-instruct:free",
        vqa_model="qwen/qwen-2.5-vl-7b-instruct:free",
    )
    provider = OpenRouterQwenVisionProvider(config=cfg)

    def mock_execute(payload):
        assert payload["model"] == "qwen/qwen-2.5-vl-7b-instruct:free"
        assert len(payload["messages"]) == 2
        return mock_openrouter_resp

    monkeypatch.setattr(provider, "_execute_http_request", mock_execute)

    resp = provider.analyze_image_sync(_create_test_image(), prompt="Describe the airport tarmac.", task="vqa")
    assert "commercial airport" in resp.text
    assert resp.provider == "openrouter"
    assert resp.model == "qwen/qwen-2.5-vl-7b-instruct:free"


def test_qwen_provider_caption_mocked_http(monkeypatch):
    mock_openrouter_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Satellite scene showing agricultural crop parcels bordering a reservoir.",
                }
            }
        ]
    }

    cfg = VisionConfig(api_key="sk-test-key")
    provider = OpenRouterQwenVisionProvider(config=cfg)
    monkeypatch.setattr(provider, "_execute_http_request", lambda p: mock_openrouter_resp)

    resp = provider.analyze_image_sync(_create_test_image(), prompt="Provide scene caption", task="caption")
    assert "agricultural crop parcels" in resp.text


def test_qwen_provider_grounding_mocked_http(monkeypatch):
    grounding_payload = {
        "objects": [
            {"label": "aircraft", "box": [0.1, 0.2, 0.4, 0.5]},
            {"label": "hangar", "box": [0.6, 0.7, 0.85, 0.95]},
        ]
    }
    mock_openrouter_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(grounding_payload),
                }
            }
        ]
    }

    cfg = VisionConfig(api_key="sk-test-key")
    provider = OpenRouterQwenVisionProvider(config=cfg)
    monkeypatch.setattr(provider, "_execute_http_request", lambda p: mock_openrouter_resp)

    resp = provider.analyze_image_sync(_create_test_image(), prompt="Locate aircraft and hangars", task="ground")
    assert resp.grounding is not None
    assert len(resp.grounding.objects) == 2
    assert resp.grounding.objects[0].label == "aircraft"
    assert resp.grounding.objects[0].box == [0.1, 0.2, 0.4, 0.5]


def test_qwen_provider_grounding_malformed_json_returns_empty_objects(monkeypatch):
    mock_openrouter_resp = {
        "choices": [{"message": {"role": "assistant", "content": "Raw non-JSON string."}}]
    }

    cfg = VisionConfig(api_key="sk-test-key")
    provider = OpenRouterQwenVisionProvider(config=cfg)
    monkeypatch.setattr(provider, "_execute_http_request", lambda p: mock_openrouter_resp)

    resp = provider.analyze_image_sync(_create_test_image(), prompt="Locate aircraft", task="ground")
    assert resp.grounding is not None
    assert len(resp.grounding.objects) == 0


# ============================================================
# Tool Integration Tests with VisionProvider
# ============================================================

def test_vqa_tool_with_vision_provider():
    mock_provider = MockVisionProvider(default_vqa_response="The image shows 5 container vessels.")
    tool = VQATool(mode="real", vision_provider=mock_provider)

    img = _create_test_image()
    res = tool.run(query="How many container vessels are docked?", image_bytes=img, mode="real")

    assert res["tool_id"] == "T1_VQA"
    assert res["answer"] == "The image shows 5 container vessels."
    assert res["metadata"]["provider"] == "mock_vision"
    assert res["confidence"] is None
    assert res["confidence_status"] == "uncalibrated"


def test_caption_tool_with_vision_provider():
    mock_provider = MockVisionProvider(default_caption_response="Overview of forested mountainous terrain.")
    tool = CaptioningTool(mode="real", vision_provider=mock_provider)

    img = _create_test_image()
    res = tool.run(image_bytes=img, mode="real")

    assert res["tool_id"] == "T2_Caption"
    assert res["answer"] == "Overview of forested mountainous terrain."
    assert res["metadata"]["provider"] == "mock_vision"


def test_grounding_tool_with_vision_provider():
    custom_boxes = [
        {"label": "solar_panel", "box": [0.1, 0.2, 0.5, 0.6]},
    ]
    mock_provider = MockVisionProvider(default_ground_objects=custom_boxes)
    tool = GroundingTool(mode="real", vision_provider=mock_provider)

    img = _create_test_image(width=1000, height=800)
    res = tool.run(query="solar panels", image_bytes=img, mode="real")

    assert res["tool_id"] == "T3_Ground"
    assert "Detected 1 spatial region(s)" in res["answer"]
    assert len(res["evidence"]) == 1
    evidence_item = res["evidence"][0]
    assert evidence_item["label"] == "solar_panel"
    # [ymin, xmin, ymax, xmax] in pixel coordinates for 1000x800 image:
    # xmin=100, xmax=500, ymin=160, ymax=480
    assert evidence_item["bbox_pixels"] == [160, 100, 480, 500]
    assert evidence_item["coverage_pct"] > 0.0
