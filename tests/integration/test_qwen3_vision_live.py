"""
Live integration test for Qwen3-VL-8B via OpenRouter.
Gated with VISION_INTEGRATION_TEST=true to ensure zero accidental calls during standard test runs.
"""
from __future__ import annotations
import os
import time
import numpy as np
from PIL import Image
import pytest

from ai.vision.config import VisionConfig, MODEL_SLUGS
from ai.vision.openrouter_qwen import OpenRouterQwenVisionProvider


def _create_sample_satellite_patch() -> Image.Image:
    """Generate a sample synthetic remote sensing image patch for live testing."""
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[:, :] = [34, 139, 34]          # vegetation
    arr[120:200, 100:220] = [30, 144, 255] # water
    arr[30:80, 40:90] = [169, 169, 169]    # building
    return Image.fromarray(arr)


@pytest.mark.skipif(
    os.environ.get("VISION_INTEGRATION_TEST", "false").lower() != "true",
    reason="Live integration test requires VISION_INTEGRATION_TEST=true and valid OPENROUTER_API_KEY",
)
def test_live_qwen3_vision_provider():
    cfg = VisionConfig.from_env()
    assert cfg.api_key is not None, "OPENROUTER_API_KEY must be configured for live integration test"

    qwen3_slug = MODEL_SLUGS["qwen3"]
    provider = OpenRouterQwenVisionProvider(config=cfg)
    test_img = _create_sample_satellite_patch()

    print("\n" + "=" * 60)
    print("LIVE QWEN3-VL-8B VISION PROVIDER AUDIT TRACE")
    print("=" * 60)
    print(f"Provider: OpenRouter")
    print(f"Model ID: {qwen3_slug}")

    # 1. Live VQA
    t0 = time.perf_counter()
    vqa_resp = provider.analyze_image_sync(
        test_img,
        prompt="What distinct surface features and colors are visible in this satellite imagery?",
        task="vqa",
        model=qwen3_slug,
    )
    vqa_latency = (time.perf_counter() - t0) * 1000.0
    print(f"\n[1] Qwen3 VQA Task (Latency: {vqa_latency:.2f} ms)")
    print(f"Answer: {vqa_resp.text}")
    assert len(vqa_resp.text) > 0

    # 2. Live Caption
    t0 = time.perf_counter()
    cap_resp = provider.analyze_image_sync(
        test_img,
        prompt="Provide a detailed remote sensing caption of this scene.",
        task="caption",
        model=qwen3_slug,
    )
    cap_latency = (time.perf_counter() - t0) * 1000.0
    print(f"\n[2] Qwen3 Caption Task (Latency: {cap_latency:.2f} ms)")
    print(f"Caption: {cap_resp.text}")
    assert len(cap_resp.text) > 0

    # 3. Live Grounding
    t0 = time.perf_counter()
    ground_resp = provider.analyze_image_sync(
        test_img,
        prompt="Locate the water body and building",
        task="ground",
        model=qwen3_slug,
    )
    ground_latency = (time.perf_counter() - t0) * 1000.0
    print(f"\n[3] Qwen3 Grounding Task (Latency: {ground_latency:.2f} ms)")
    print(f"Grounding result: {ground_resp.grounding}")
    print("=" * 60)

    assert ground_resp.grounding is not None
