"""
Unit tests for VisionProvider interface, configuration, and GroundingBox data models.
"""
from __future__ import annotations
import asyncio
import pytest
from pydantic import ValidationError

from ai.vision.base import GroundingBox, GroundingResult, VisionResponse
from ai.vision.config import VisionConfig
from ai.vision.mock import MockVisionProvider
from ai.vision import get_vision_provider


def test_vision_config_defaults_and_masking():
    cfg = VisionConfig(api_key="sk-or-v1-secret12345")
    assert cfg.provider == "qwen_openrouter"
    assert cfg.model == "qwen/qwen-2.5-vl-7b-instruct:free"
    assert "sk-or-v1" not in repr(cfg)
    assert "***" in repr(cfg)


def test_vision_config_from_env(monkeypatch):
    monkeypatch.setenv("VISION_PROVIDER", "mock")
    monkeypatch.setenv("VISION_MODEL", "custom-vl-model")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-openrouter")
    monkeypatch.setenv("VISION_TIMEOUT", "30.0")
    monkeypatch.setenv("VISION_MAX_RETRIES", "3")

    cfg = VisionConfig.from_env()
    assert cfg.provider == "mock"
    assert cfg.model == "custom-vl-model"
    assert cfg.api_key == "test-key-openrouter"
    assert cfg.timeout == 30.0
    assert cfg.max_retries == 3


def test_grounding_box_validation_success():
    box = GroundingBox(label="storage_tank", box=[0.1, 0.2, 0.4, 0.5])
    assert box.label == "storage_tank"
    assert box.box == [0.1, 0.2, 0.4, 0.5]


def test_grounding_box_invalid_coordinates_rejected():
    # x0 > x1
    with pytest.raises(ValidationError):
        GroundingBox(label="building", box=[0.8, 0.2, 0.3, 0.5])

    # y0 > y1
    with pytest.raises(ValidationError):
        GroundingBox(label="building", box=[0.1, 0.9, 0.4, 0.5])

    # Empty label
    with pytest.raises(ValidationError):
        GroundingBox(label="", box=[0.1, 0.2, 0.4, 0.5])


def test_grounding_box_to_pixel_conversion():
    # Normalized coordinates
    box_norm = GroundingBox(label="runway", box=[0.1, 0.2, 0.6, 0.8])
    pixel_box = box_norm.to_pixel_box(width=1000, height=500)
    # [ymin, xmin, ymax, xmax]
    assert pixel_box == [100, 100, 400, 600]

    # Already in pixel space
    box_pixel = GroundingBox(label="bridge", box=[50.0, 60.0, 150.0, 200.0])
    pixel_box2 = box_pixel.to_pixel_box(width=500, height=500)
    assert pixel_box2 == [60, 50, 200, 150]


def test_mock_vision_provider_sync_and_async():
    mock_provider = MockVisionProvider()

    # VQA
    vqa_resp = mock_provider.analyze_image_sync(b"fake_bytes", prompt="What is visible?", task="vqa")
    assert "[MOCK QWEN]" in vqa_resp.text
    assert vqa_resp.grounding is None

    # Caption
    caption_resp = mock_provider.analyze_image_sync(b"fake_bytes", prompt="Describe", task="caption")
    assert "High-resolution satellite view" in caption_resp.text

    # Grounding
    ground_resp = mock_provider.analyze_image_sync(b"fake_bytes", prompt="Locate buildings", task="ground")
    assert ground_resp.grounding is not None
    assert len(ground_resp.grounding.objects) == 2
    assert ground_resp.grounding.objects[0].label == "building"

    # Async
    async_resp = asyncio.run(mock_provider.analyze_image(b"fake_bytes", prompt="Async test", task="vqa"))
    assert async_resp.text == mock_provider.default_vqa


def test_get_vision_provider_factory():
    cfg_mock = VisionConfig(provider="mock")
    provider = get_vision_provider(cfg_mock)
    assert isinstance(provider, MockVisionProvider)
