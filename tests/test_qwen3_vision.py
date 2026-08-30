"""
Unit and regression tests for Qwen3-VL-8B vision provider integration and comparative evaluation.
Uses mocked HTTP responses and MockVisionProvider for 100% offline testing.
"""
from __future__ import annotations
import json
import os
import pytest
from PIL import Image

from ai.vision.base import VisionResponse, GroundingBox, GroundingResult
from ai.vision.config import VisionConfig, resolve_model_slug, MODEL_SLUGS
from ai.vision.errors import GroundingParseError, VisionAuthenticationError
from ai.vision.mock import MockVisionProvider
from ai.vision.openrouter_qwen import OpenRouterQwenVisionProvider
from tests.evaluation.compare_vlm_models import run_model_benchmark, generate_comparison_report
from tools.vqa import VQATool
from tools.captioning import CaptioningTool
from tools.grounding import GroundingTool


def test_qwen3_model_slug_resolution():
    assert resolve_model_slug("qwen3") == "qwen/qwen3-vl-8b-instruct"
    assert resolve_model_slug("qwen3-vl") == "qwen/qwen3-vl-8b-instruct"
    assert resolve_model_slug("qwen3-vl-8b") == "qwen/qwen3-vl-8b-instruct"
    assert resolve_model_slug("qwen3_free") == "qwen/qwen3-vl-8b-instruct:free"
    assert resolve_model_slug("qwen25") == "qwen/qwen-2.5-vl-7b-instruct:free"


def test_task_specific_model_selection_in_config(monkeypatch):
    monkeypatch.setenv("VISION_MODEL", "qwen25")
    monkeypatch.setenv("VISION_VQA_MODEL", "qwen3")
    monkeypatch.setenv("VISION_GROUND_MODEL", "qwen3")
    monkeypatch.setenv("VISION_CAPTION_MODEL", "qwen25")

    cfg = VisionConfig.from_env()
    assert cfg.model == MODEL_SLUGS["qwen25_free"]
    assert cfg.get_model_for_task("vqa") == MODEL_SLUGS["qwen3"]
    assert cfg.get_model_for_task("ground") == MODEL_SLUGS["qwen3"]
    assert cfg.get_model_for_task("caption") == MODEL_SLUGS["qwen25_free"]


def test_qwen3_vqa_mocked_http(monkeypatch):
    mock_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Qwen3 analysis: Commercial aircraft docked along terminal concourse.",
                }
            }
        ]
    }

    cfg = VisionConfig(model=MODEL_SLUGS["qwen3"], api_key="sk-test-key")
    provider = OpenRouterQwenVisionProvider(config=cfg)

    def mock_exec(payload):
        assert payload["model"] == MODEL_SLUGS["qwen3"]
        return mock_resp

    monkeypatch.setattr(provider, "_execute_http_request", mock_exec)

    dummy_img = Image.new("RGB", (100, 100))
    resp = provider.analyze_image_sync(dummy_img, prompt="What is visible?", task="vqa")
    assert "Qwen3 analysis" in resp.text
    assert resp.model == MODEL_SLUGS["qwen3"]


def test_qwen3_grounding_mocked_http(monkeypatch):
    grounding_data = {
        "objects": [
            {"label": "building", "box": [0.12, 0.15, 0.45, 0.55]},
            {"label": "storage_tank", "box": [0.60, 0.65, 0.85, 0.90]},
        ]
    }
    mock_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(grounding_data),
                }
            }
        ]
    }

    cfg = VisionConfig(model=MODEL_SLUGS["qwen3"], api_key="sk-test-key")
    provider = OpenRouterQwenVisionProvider(config=cfg)
    monkeypatch.setattr(provider, "_execute_http_request", lambda p: mock_resp)

    dummy_img = Image.new("RGB", (200, 200))
    resp = provider.analyze_image_sync(dummy_img, prompt="Locate structures", task="ground")
    assert resp.grounding is not None
    assert len(resp.grounding.objects) == 2
    assert resp.grounding.objects[0].label == "building"
    assert resp.grounding.objects[0].box == [0.12, 0.15, 0.45, 0.55]


def test_qwen3_tools_integration():
    mock_provider = MockVisionProvider(default_vqa_response="[Qwen3] 4 storage tanks detected.")
    
    # T1
    vqa_tool = VQATool(mode="real", vision_provider=mock_provider)
    res_vqa = vqa_tool.run(query="How many storage tanks?", image_bytes=Image.new("RGB", (50, 50)), mode="real")
    assert res_vqa["tool_id"] == "T1_VQA"
    assert "4 storage tanks" in res_vqa["answer"]

    # T2
    cap_tool = CaptioningTool(mode="real", vision_provider=mock_provider)
    res_cap = cap_tool.run(image_bytes=Image.new("RGB", (50, 50)), mode="real")
    assert res_cap["tool_id"] == "T2_Caption"

    # T3
    grd_tool = GroundingTool(mode="real", vision_provider=mock_provider)
    res_grd = grd_tool.run(query="storage tanks", image_bytes=Image.new("RGB", (500, 500)), mode="real")
    assert res_grd["tool_id"] == "T3_Ground"
    assert len(res_grd["evidence"]) > 0


def test_offline_comparative_benchmark_runner():
    """Test full comparative benchmark suite execution with mock provider."""
    samples_path = os.path.join(os.path.dirname(__file__), "evaluation", "qwen_vlm_samples.json")
    with open(samples_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    def mock_qwen25_handler(image_input, prompt, task, **kwargs):
        if task == "ground":
            boxes = [GroundingBox(label="obj", box=[0.15, 0.20, 0.40, 0.45])]
            return VisionResponse(
                text=json.dumps({"objects": [b.model_dump() for b in boxes]}),
                grounding=GroundingResult(objects=boxes),
                latency_ms=1200.0,
                provider="mock",
                model=MODEL_SLUGS["qwen25_free"],
            )
        return VisionResponse(
            text=f"Qwen2.5 observed {prompt} features including vegetation, building, water, parcel, aircraft, ship, highway.",
            latency_ms=950.0,
            provider="mock",
            model=MODEL_SLUGS["qwen25_free"],
        )

    def mock_qwen3_handler(image_input, prompt, task, **kwargs):
        if task == "ground":
            boxes = [
                GroundingBox(label="obj", box=[0.15, 0.20, 0.40, 0.45]),
                GroundingBox(label="obj", box=[0.50, 0.60, 0.75, 0.85]),
            ]
            return VisionResponse(
                text=json.dumps({"objects": [b.model_dump() for b in boxes]}),
                grounding=GroundingResult(objects=boxes),
                latency_ms=1450.0,
                provider="mock",
                model=MODEL_SLUGS["qwen3"],
            )
        return VisionResponse(
            text=f"Qwen3 high-precision analysis of {prompt}: dense building footprints, transportation highway, water, agricultural parcel, aircraft, ship.",
            latency_ms=1100.0,
            provider="mock",
            model=MODEL_SLUGS["qwen3"],
        )

    p_a = MockVisionProvider(custom_handler=mock_qwen25_handler)
    p_b = MockVisionProvider(custom_handler=mock_qwen3_handler)
    img = Image.new("RGB", (256, 256))

    res_a = run_model_benchmark(p_a, MODEL_SLUGS["qwen25_free"], samples, img)
    res_b = run_model_benchmark(p_b, MODEL_SLUGS["qwen3"], samples, img)

    if res_a["successful_samples"] != len(samples):
        print("\nERRORS in res_a:", [s.get("error") for s in res_a["sample_results"] if s.get("error")])

    assert res_a["successful_samples"] == len(samples)
    assert res_b["successful_samples"] == len(samples)

    report_md = generate_comparison_report(res_a, res_b)
    assert "Qwen2.5-VL-7B" in report_md
    assert "Qwen3-VL-8B" in report_md
    assert os.path.exists(os.path.join(os.path.dirname(__file__), "evaluation", "benchmark_run.json"))
