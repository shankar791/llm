"""
Benchmark Runner and Test Suite for Multimodal Remote Sensing Vision Models.
Evaluates T1_VQA, T2_Caption, T3_Grounding, and Operational reliability.
"""
from __future__ import annotations
import json
import os
import time
import pytest
import numpy as np
from PIL import Image

from ai.vision.base import GroundingBox, GroundingResult, VisionResponse
from ai.vision.mock import MockVisionProvider
from tests.evaluation.evaluator import (
    compute_box_iou,
    evaluate_grounding_accuracy,
    evaluate_vqa_quality,
    evaluate_caption_quality,
    OperationalBenchmarkSummary,
)


def _load_samples():
    path = os.path.join(os.path.dirname(__file__), "qwen_vlm_samples.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_iou_computation_accuracy():
    # Identical boxes -> IoU = 1.0
    assert compute_box_iou([0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]) == 1.0

    # Half overlap -> intersection = 0.5 * 1.0 = 0.5, union = 1.0 + 1.0 - 0.5 = 1.5 -> IoU = 0.333
    iou = compute_box_iou([0.0, 0.0, 1.0, 1.0], [0.5, 0.0, 1.5, 1.0])
    assert round(iou, 3) == 0.333

    # Disjoint boxes -> IoU = 0.0
    assert compute_box_iou([0.0, 0.0, 0.2, 0.2], [0.5, 0.5, 0.8, 0.8]) == 0.0


def test_grounding_evaluation_metrics():
    # 2 GT boxes, 2 Pred boxes (1 exact match, 1 slight offset)
    gt_boxes = [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8]]
    pred_boxes = [[0.1, 0.1, 0.3, 0.3], [0.52, 0.48, 0.82, 0.78]]

    eval_res = evaluate_grounding_accuracy(pred_boxes, gt_boxes, iou_threshold=0.5)
    assert eval_res.precision == 1.0
    assert eval_res.recall == 1.0
    assert eval_res.f1_score == 1.0
    assert eval_res.mean_iou > 0.80
    assert eval_res.valid_box_syntax_rate == 1.0


def test_vqa_quality_evaluation():
    ans = "The image shows dense residential buildings with asphalt roadways."
    res = evaluate_vqa_quality(ans, target_keywords=["building", "residential"])
    assert res.keyword_recall == 1.0
    assert res.has_substantive_answer is True
    assert res.score >= 0.90


def test_caption_quality_evaluation():
    caption = "High-resolution satellite view showing agricultural crop parcels bordering an urban water reservoir."
    res = evaluate_caption_quality(caption, expected_concepts=["agricultural", "reservoir"])
    assert res.concept_coverage == 1.0
    assert res.domain_terminology_density >= 0.6
    assert res.quality_score >= 0.80


def test_full_benchmark_suite_execution():
    """
    Execute end-to-end evaluation across the 10 representative remote sensing samples.
    """
    samples = _load_samples()
    tracker = OperationalBenchmarkSummary()

    # Create realistic test mock provider
    def benchmark_handler(image_input, prompt, task, **kwargs):
        tracker.total_requests += 1
        t0 = time.perf_counter()
        
        if task == "vqa":
            ans = f"Detailed analysis indicates prominent {prompt} features across the scene."
            lat = 12.0
            tracker.successful_requests += 1
            tracker.latencies_ms.append(lat)
            return VisionResponse(text=ans, latency_ms=lat, provider="mock", model="qwen/qwen-2.5-vl-7b-instruct:free")
        
        elif task == "caption":
            cap = "Satellite imagery overview exhibiting dense urban infrastructure and adjacent agricultural land cover."
            lat = 15.0
            tracker.successful_requests += 1
            tracker.latencies_ms.append(lat)
            return VisionResponse(text=cap, latency_ms=lat, provider="mock", model="qwen/qwen-2.5-vl-7b-instruct:free")
        
        elif task == "ground":
            tracker.structured_json_total_count += 1
            boxes = [
                GroundingBox(label="object", box=[0.15, 0.20, 0.45, 0.55]),
            ]
            tracker.structured_json_success_count += 1
            tracker.successful_requests += 1
            lat = 18.0
            tracker.latencies_ms.append(lat)
            return VisionResponse(
                text=json.dumps({"objects": [b.model_dump() for b in boxes]}),
                grounding=GroundingResult(objects=boxes),
                latency_ms=lat,
                provider="mock",
                model="qwen/qwen-2.5-vl-7b-instruct:free",
            )
        
        return VisionResponse(text="Unknown", latency_ms=5.0)

    provider = MockVisionProvider(custom_handler=benchmark_handler)
    dummy_img = Image.new("RGB", (256, 256), color=(50, 150, 50))

    # Run through all 10 benchmark samples
    for sample in samples:
        resp = provider.analyze_image_sync(dummy_img, prompt=sample["question"], task=sample["task"])
        assert resp is not None

    # Verify operational reliability metrics
    assert tracker.total_requests == len(samples)
    assert tracker.successful_requests == len(samples)
    assert tracker.failure_rate == 0.0
    assert tracker.mean_latency_ms > 0.0
    assert tracker.structured_output_reliability == 100.0
