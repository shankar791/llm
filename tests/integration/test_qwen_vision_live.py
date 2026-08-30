"""
Live integration test for Qwen2.5-VL via OpenRouter.
Gated with VISION_INTEGRATION_TEST=true to ensure zero accidental calls during standard test runs.
"""
from __future__ import annotations
import io
import os
import time
import numpy as np
from PIL import Image
import pytest

from ai.vision.config import VisionConfig
from ai.vision.openrouter_qwen import OpenRouterQwenVisionProvider


def _create_sample_satellite_patch() -> Image.Image:
    """Generate a sample synthetic remote sensing image patch for live testing."""
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    # Background green (vegetation)
    arr[:, :] = [34, 139, 34]
    # Blue reservoir (water)
    arr[120:200, 100:220] = [30, 144, 255]
    # Gray building (structure)
    arr[30:80, 40:90] = [169, 169, 169]
    return Image.fromarray(arr)


@pytest.mark.skipif(
    os.environ.get("VISION_INTEGRATION_TEST", "false").lower() != "true",
    reason="Live integration test requires VISION_INTEGRATION_TEST=true and valid OPENROUTER_API_KEY",
)
def test_live_qwen_vision_provider():
    cfg = VisionConfig.from_env()
    assert cfg.api_key is not None, "OPENROUTER_API_KEY must be configured for live integration test"

    provider = OpenRouterQwenVisionProvider(config=cfg)
    test_img = _create_sample_satellite_patch()

    print("\n" + "=" * 60)
    print("LIVE QWEN2.5-VL VISION PROVIDER AUDIT TRACE")
    print("=" * 60)
    print(f"Provider: {cfg.provider}")
    print(f"Model:    {cfg.model}")

    # 1. Live VQA
    t0 = time.perf_counter()
    vqa_resp = provider.analyze_image_sync(test_img, prompt="What features and colors are visible in this image?", task="vqa")
    vqa_latency = (time.perf_counter() - t0) * 1000.0
    print(f"\n[1] VQA Task (Latency: {vqa_latency:.2f} ms)")
    print(f"Answer: {vqa_resp.text}")
    assert len(vqa_resp.text) > 0

    # 2. Live Caption
    t0 = time.perf_counter()
    cap_resp = provider.analyze_image_sync(test_img, prompt="Describe the satellite scene.", task="caption")
    cap_latency = (time.perf_counter() - t0) * 1000.0
    print(f"\n[2] Caption Task (Latency: {cap_latency:.2f} ms)")
    print(f"Caption: {cap_resp.text}")
    assert len(cap_resp.text) > 0

    # 3. Live Grounding
    t0 = time.perf_counter()
    ground_resp = provider.analyze_image_sync(test_img, prompt="Locate the water body and building", task="ground")
    ground_latency = (time.perf_counter() - t0) * 1000.0
    print(f"\n[3] Grounding Task (Latency: {ground_latency:.2f} ms)")
    print(f"Grounding result: {ground_resp.grounding}")
    print("=" * 60)

    assert ground_resp.grounding is not None
