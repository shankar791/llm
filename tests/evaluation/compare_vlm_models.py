"""
Comparative Evaluation Engine for Vision-Language Models on Satellite Imagery (STEP 9B).
Compares Model A (Qwen2.5-VL-7B) vs Model B (Qwen3-VL-8B) on identical benchmark samples.
Outputs reproducible benchmark metrics and exports benchmark_run.json.
"""
from __future__ import annotations
import datetime
import json
import os
import statistics
import time
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

from ai.vision.base import VisionProvider, VisionResponse, GroundingBox, GroundingResult
from ai.vision.config import VisionConfig, MODEL_SLUGS
from ai.vision.mock import MockVisionProvider
from ai.vision.openrouter_qwen import OpenRouterQwenVisionProvider
from tests.evaluation.evaluator import (
    compute_box_iou,
    evaluate_grounding_accuracy,
    evaluate_vqa_quality,
    evaluate_caption_quality,
    GroundingEvaluationResult,
    VQAEvaluationResult,
    CaptionEvaluationResult,
)


def _create_sample_image(width: int = 512, height: int = 512) -> Image.Image:
    """Generate a clean synthetic satellite-like image patch."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :] = [34, 139, 34]          # vegetation green
    arr[200:350, 150:400] = [30, 144, 255] # water blue
    arr[50:150, 50:150] = [169, 169, 169]  # building gray
    return Image.fromarray(arr)


def run_model_benchmark(
    provider: VisionProvider,
    model_name: str,
    samples: List[Dict[str, Any]],
    image_input: Any,
    rate_limit_delay_s: float = 0.0,
) -> Dict[str, Any]:
    """
    Execute benchmark evaluation over a dataset for a specific model.
    """
    vqa_scores: List[float] = []
    vqa_substantive: List[bool] = []
    caption_coverages: List[float] = []
    caption_qualities: List[float] = []
    grounding_ious: List[float] = []
    grounding_precisions: List[float] = []
    grounding_recalls: List[float] = []
    grounding_f1s: List[float] = []
    grounding_syntax_valid: List[bool] = []

    latencies_ms: List[float] = []
    failed_count = 0
    rate_limit_count = 0
    sample_results = []

    for idx, sample in enumerate(samples):
        task = sample["task"]
        query = sample["question"]
        t0 = time.perf_counter()

        try:
            resp = provider.analyze_image_sync(
                image_input=image_input,
                prompt=query,
                task=task,
                model=model_name,
            )
            lat = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat)

            # Task 1: VQA
            if task == "vqa":
                expected_kws = sample.get("expected_keywords", [sample.get("target_feature", "")])
                vqa_res = evaluate_vqa_quality(resp.text, expected_kws)
                vqa_scores.append(vqa_res.keyword_recall)
                vqa_substantive.append(vqa_res.has_substantive_answer)
                sample_results.append({
                    "id": sample["id"],
                    "task": task,
                    "status": "success",
                    "latency_ms": round(lat, 2),
                    "keyword_recall": vqa_res.keyword_recall,
                    "substantive": vqa_res.has_substantive_answer,
                })

            # Task 2: Caption
            elif task == "caption":
                expected_concepts = sample.get("expected_concepts", [sample.get("target_feature", "")])
                cap_res = evaluate_caption_quality(resp.text, expected_concepts)
                caption_coverages.append(cap_res.concept_coverage)
                caption_qualities.append(cap_res.quality_score)
                sample_results.append({
                    "id": sample["id"],
                    "task": task,
                    "status": "success",
                    "latency_ms": round(lat, 2),
                    "concept_coverage": cap_res.concept_coverage,
                    "quality_score": cap_res.quality_score,
                })

            # Task 3: Grounding
            elif task == "ground":
                ref_boxes = sample.get("reference_boxes")
                pred_boxes = []
                if resp.grounding and resp.grounding.objects:
                    pred_boxes = [obj.box for obj in resp.grounding.objects]
                    grounding_syntax_valid.append(True)
                else:
                    grounding_syntax_valid.append(False)

                if ref_boxes:
                    grd_res = evaluate_grounding_accuracy(pred_boxes, ref_boxes, iou_threshold=0.5)
                    grounding_ious.extend(grd_res.iou_scores)
                    grounding_precisions.append(grd_res.precision)
                    grounding_recalls.append(grd_res.recall)
                    grounding_f1s.append(grd_res.f1_score)
                    sample_results.append({
                        "id": sample["id"],
                        "task": task,
                        "status": "success",
                        "latency_ms": round(lat, 2),
                        "mean_iou": grd_res.mean_iou,
                        "precision": grd_res.precision,
                        "recall": grd_res.recall,
                        "f1_score": grd_res.f1_score,
                    })
                else:
                    sample_results.append({
                        "id": sample["id"],
                        "task": task,
                        "status": "success",
                        "latency_ms": round(lat, 2),
                        "mean_iou": "N/A (No GT boxes)",
                    })

        except Exception as e:
            failed_count += 1
            if "429" in str(e):
                rate_limit_count += 1
            sample_results.append({
                "id": sample["id"],
                "task": task,
                "status": "error",
                "error": str(e),
            })

        if rate_limit_delay_s > 0:
            time.sleep(rate_limit_delay_s)

    total_samples = len(samples)
    successful_samples = total_samples - failed_count

    mean_lat = statistics.mean(latencies_ms) if latencies_ms else 0.0
    median_lat = statistics.median(latencies_ms) if latencies_ms else 0.0
    p95_lat = np.percentile(latencies_ms, 95) if latencies_ms else 0.0

    return {
        "model_name": model_name,
        "total_samples": total_samples,
        "successful_samples": successful_samples,
        "failed_samples": failed_count,
        "rate_limit_count": rate_limit_count,
        "failure_rate_pct": round((failed_count / total_samples) * 100.0, 2) if total_samples else 0.0,
        "latencies": {
            "mean_ms": round(mean_lat, 2),
            "median_ms": round(median_lat, 2),
            "p95_ms": round(p95_lat, 2),
        },
        "vqa_metrics": {
            "semantic_keyword_recall_pct": round(statistics.mean(vqa_scores) * 100.0, 2) if vqa_scores else 0.0,
            "substantive_rate_pct": round((sum(vqa_substantive) / len(vqa_substantive)) * 100.0, 2) if vqa_substantive else 0.0,
            "count": len(vqa_scores),
        },
        "caption_metrics": {
            "concept_coverage_pct": round(statistics.mean(caption_coverages) * 100.0, 2) if caption_coverages else 0.0,
            "quality_score_mean": round(statistics.mean(caption_qualities), 3) if caption_qualities else 0.0,
            "count": len(caption_coverages),
        },
        "grounding_metrics": {
            "valid_syntax_rate_pct": round((sum(grounding_syntax_valid) / len(grounding_syntax_valid)) * 100.0, 2) if grounding_syntax_valid else 0.0,
            "mean_iou": round(statistics.mean(grounding_ious), 3) if grounding_ious else 0.0,
            "median_iou": round(statistics.median(grounding_ious), 3) if grounding_ious else 0.0,
            "precision": round(statistics.mean(grounding_precisions), 3) if grounding_precisions else 0.0,
            "recall": round(statistics.mean(grounding_recalls), 3) if grounding_recalls else 0.0,
            "f1_score": round(statistics.mean(grounding_f1s), 3) if grounding_f1s else 0.0,
            "evaluated_boxes_count": len(grounding_ious),
        },
        "sample_results": sample_results,
    }


def generate_comparison_report(
    result_a: Dict[str, Any],
    result_b: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """Generate a clean Markdown comparison report and save benchmark metadata."""
    m_a = result_a["model_name"]
    m_b = result_b["model_name"]

    lines = [
        "# SatQuery AI — Vision-Language Model Benchmark Comparison",
        f"**Date**: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Dataset**: `tests/evaluation/qwen_vlm_samples.json` ({result_a['total_samples']} samples)",
        "",
        "| Evaluation Metric | Model A: Qwen2.5-VL-7B | Model B: Qwen3-VL-8B | Delta (B - A) |",
        "|---|---:|---:|---:|",
        f"| **VQA Semantic Keyword Recall** | {result_a['vqa_metrics']['semantic_keyword_recall_pct']}% | {result_b['vqa_metrics']['semantic_keyword_recall_pct']}% | {result_b['vqa_metrics']['semantic_keyword_recall_pct'] - result_a['vqa_metrics']['semantic_keyword_recall_pct']:+.2f}% |",
        f"| **VQA Substantive Answer Rate** | {result_a['vqa_metrics']['substantive_rate_pct']}% | {result_b['vqa_metrics']['substantive_rate_pct']}% | {result_b['vqa_metrics']['substantive_rate_pct'] - result_a['vqa_metrics']['substantive_rate_pct']:+.2f}% |",
        f"| **Caption Concept Coverage** | {result_a['caption_metrics']['concept_coverage_pct']}% | {result_b['caption_metrics']['concept_coverage_pct']}% | {result_b['caption_metrics']['concept_coverage_pct'] - result_a['caption_metrics']['concept_coverage_pct']:+.2f}% |",
        f"| **Caption Quality Score** | {result_a['caption_metrics']['quality_score_mean']} | {result_b['caption_metrics']['quality_score_mean']} | {result_b['caption_metrics']['quality_score_mean'] - result_a['caption_metrics']['quality_score_mean']:+.3f} |",
        f"| **Grounding Valid Box Rate** | {result_a['grounding_metrics']['valid_syntax_rate_pct']}% | {result_b['grounding_metrics']['valid_syntax_rate_pct']}% | {result_b['grounding_metrics']['valid_syntax_rate_pct'] - result_a['grounding_metrics']['valid_syntax_rate_pct']:+.2f}% |",
        f"| **Grounding Mean IoU** | {result_a['grounding_metrics']['mean_iou']} | {result_b['grounding_metrics']['mean_iou']} | {result_b['grounding_metrics']['mean_iou'] - result_a['grounding_metrics']['mean_iou']:+.3f} |",
        f"| **Grounding Precision** | {result_a['grounding_metrics']['precision']} | {result_b['grounding_metrics']['precision']} | {result_b['grounding_metrics']['precision'] - result_a['grounding_metrics']['precision']:+.3f} |",
        f"| **Grounding Recall** | {result_a['grounding_metrics']['recall']} | {result_b['grounding_metrics']['recall']} | {result_b['grounding_metrics']['recall'] - result_a['grounding_metrics']['recall']:+.3f} |",
        f"| **Grounding F1 Score** | {result_a['grounding_metrics']['f1_score']} | {result_b['grounding_metrics']['f1_score']} | {result_b['grounding_metrics']['f1_score'] - result_a['grounding_metrics']['f1_score']:+.3f} |",
        f"| **Mean Latency (ms)** | {result_a['latencies']['mean_ms']} ms | {result_b['latencies']['mean_ms']} ms | {result_b['latencies']['mean_ms'] - result_a['latencies']['mean_ms']:+.2f} ms |",
        f"| **p95 Latency (ms)** | {result_a['latencies']['p95_ms']} ms | {result_b['latencies']['p95_ms']} ms | {result_b['latencies']['p95_ms'] - result_a['latencies']['p95_ms']:+.2f} ms |",
        f"| **Failure Rate (%)** | {result_a['failure_rate_pct']}% | {result_b['failure_rate_pct']}% | {result_b['failure_rate_pct'] - result_a['failure_rate_pct']:+.2f}% |",
        f"| **Rate Limit 429 Hits** | {result_a['rate_limit_count']} | {result_b['rate_limit_count']} | {result_b['rate_limit_count'] - result_a['rate_limit_count']:+d} |",
    ]

    report = "\n".join(lines)

    # Save benchmark_run.json
    run_meta = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evaluator_version": "1.2.0",
        "dataset": "tests/evaluation/qwen_vlm_samples.json",
        "dataset_samples": result_a["total_samples"],
        "model_a": result_a,
        "model_b": result_b,
    }

    meta_path = output_path or os.path.join(os.path.dirname(__file__), "benchmark_run.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    return report
