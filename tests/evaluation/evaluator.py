"""
Evaluation and Benchmarking Suite for Multimodal Remote Sensing Vision Models.
Computes quantitative quality metrics for T1_VQA, T2_Caption, T3_Grounding, and Operational reliability.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def compute_box_iou(
    box1: List[float],
    box2: List[float],
) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes [x0, y0, x1, y1].
    Handles both normalized coordinates [0.0, 1.0] and pixel coordinates.
    """
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    # Determine intersection rectangle
    inter_x0 = max(x0_1, x0_2)
    inter_y0 = max(y0_1, y0_2)
    inter_x1 = min(x1_1, x1_2)
    inter_y1 = min(y1_1, y1_2)

    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h

    # Compute individual areas
    area1 = max(0.0, (x1_1 - x0_1) * (y1_1 - y0_1))
    area2 = max(0.0, (x1_2 - x0_2) * (y1_2 - y0_2))
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


@dataclass
class GroundingEvaluationResult:
    """Quantitative results for T3_Grounding spatial localization."""
    iou_scores: List[float]
    mean_iou: float
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    valid_box_syntax_rate: float


def evaluate_grounding_accuracy(
    predicted_boxes: List[List[float]],
    ground_truth_boxes: List[List[float]],
    iou_threshold: float = 0.5,
) -> GroundingEvaluationResult:
    """
    Evaluate predicted bounding boxes against ground truth reference boxes.
    """
    if not ground_truth_boxes:
        valid_rate = 1.0 if not predicted_boxes else 0.0
        return GroundingEvaluationResult(
            iou_scores=[],
            mean_iou=0.0,
            true_positives=0,
            false_positives=len(predicted_boxes),
            false_negatives=0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            valid_box_syntax_rate=valid_rate,
        )

    # 1. Validate box syntax
    valid_syntax_count = 0
    clean_preds = []
    for b in predicted_boxes:
        if len(b) == 4 and b[0] <= b[2] and b[1] <= b[3]:
            valid_syntax_count += 1
            clean_preds.append(b)

    syntax_rate = valid_syntax_count / len(predicted_boxes) if predicted_boxes else 1.0

    # 2. Match predictions to ground truth via highest IoU
    matched_gt = set()
    iou_scores = []
    tp = 0

    for pred in clean_preds:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(ground_truth_boxes):
            iou = compute_box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        iou_scores.append(best_iou)
        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(clean_preds) - tp
    fn = len(ground_truth_boxes) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

    return GroundingEvaluationResult(
        iou_scores=iou_scores,
        mean_iou=round(mean_iou, 3),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1_score=round(f1, 3),
        valid_box_syntax_rate=round(syntax_rate, 3),
    )


@dataclass
class VQAEvaluationResult:
    """Quantitative results for T1_VQA answer quality."""
    keyword_recall: float
    has_substantive_answer: bool
    word_count: int
    score: float


def evaluate_vqa_quality(
    answer: str,
    target_keywords: List[str],
) -> VQAEvaluationResult:
    """
    Evaluate VQA response quality based on semantic keywords and substantive length.
    """
    ans_clean = answer.lower()
    if not target_keywords:
        hit_rate = 1.0
    else:
        hits = sum(1 for kw in target_keywords if kw.lower() in ans_clean)
        hit_rate = hits / len(target_keywords)

    words = answer.split()
    has_content = len(words) >= 4 and not any(neg in ans_clean for neg in ["cannot answer", "unable to determine"])

    # Quality score: 70% keyword relevance, 30% completeness
    score = (hit_rate * 0.7) + (0.3 if has_content else 0.0)

    return VQAEvaluationResult(
        keyword_recall=round(hit_rate, 3),
        has_substantive_answer=has_content,
        word_count=len(words),
        score=round(score, 3),
    )


@dataclass
class CaptionEvaluationResult:
    """Quantitative results for T2_Caption quality."""
    concept_coverage: float
    domain_terminology_density: float
    word_count: int
    quality_score: float


def evaluate_caption_quality(
    caption: str,
    expected_concepts: List[str],
) -> CaptionEvaluationResult:
    """
    Evaluate caption richness, terminology, and conceptual coverage.
    """
    cap_lower = caption.lower()
    rs_domain_terms = [
        "satellite", "aerial", "land cover", "terrain", "canopy", "vegetation",
        "agricultural", "urban", "waterbody", "reservoir", "infrastructure", "structure",
        "parcel", "coastal", "footprint", "scene", "density", "residential", "industrial"
    ]

    # 1. Concept coverage
    if not expected_concepts:
        coverage = 1.0
    else:
        hits = sum(1 for c in expected_concepts if c.lower() in cap_lower)
        coverage = hits / len(expected_concepts)

    # 2. Domain terminology density
    term_hits = sum(1 for t in rs_domain_terms if t in cap_lower)
    term_density = min(1.0, term_hits / 3.0)

    # 3. Length score (optimal: 15-80 words)
    words = caption.split()
    len_score = 1.0 if 15 <= len(words) <= 100 else 0.5

    quality = (coverage * 0.5) + (term_density * 0.3) + (len_score * 0.2)

    return CaptionEvaluationResult(
        concept_coverage=round(coverage, 3),
        domain_terminology_density=round(term_density, 3),
        word_count=len(words),
        quality_score=round(quality, 3),
    )


@dataclass
class OperationalBenchmarkSummary:
    """Operational reliability and latency tracking summary."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    structured_json_success_count: int = 0
    structured_json_total_count: int = 0
    latencies_ms: List[float] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round((self.failed_requests / self.total_requests) * 100.0, 2)

    @property
    def mean_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return round(sum(self.latencies_ms) / len(self.latencies_ms), 2)

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = min(len(sorted_lat) - 1, math.ceil(0.95 * len(sorted_lat)) - 1)
        return round(sorted_lat[idx], 2)

    @property
    def structured_output_reliability(self) -> float:
        if self.structured_json_total_count == 0:
            return 100.0
        return round((self.structured_json_success_count / self.structured_json_total_count) * 100.0, 2)
