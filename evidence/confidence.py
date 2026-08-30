"""
Confidence aggregation for SatQuery AI.
Fuses per-tool confidence scores into a single score, handling calibrated and uncalibrated models.
"""
from __future__ import annotations
import math
from typing import Optional, List, Dict, Any


class ConfidenceAggregator:
    """
    Fuses per-tool confidence scores into a single calibrated confidence.
    Gracefully handles uncalibrated models (returning None / uncalibrated).
    """

    TOOL_WEIGHTS: dict[str, float] = {
        "T1_VQA": 0.90,
        "T2_Caption": 0.70,
        "T3_Ground": 0.85,
        "T4_Change": 0.88,
        "T5_OpticalSAR": 0.80,
    }
    FLOOR: float = 0.05
    CEILING: float = 0.97

    def aggregate(self, results: List[Dict[str, Any]]) -> Optional[float]:
        """
        Compute the aggregated confidence score.
        Returns None if all tools report uncalibrated / unavailable confidence.
        """
        if not results:
            return None

        log_sum = 0.0
        weight_sum = 0.0
        valid_count = 0

        for r in results:
            raw_c = r.get("confidence")
            status = r.get("confidence_status", "calibrated")

            if raw_c is not None and status != "uncalibrated":
                try:
                    c_val = float(raw_c)
                    c = max(self.FLOOR, min(self.CEILING, c_val))
                    w = self.TOOL_WEIGHTS.get(r.get("tool_id", ""), 0.75)
                    log_sum += w * math.log(c)
                    weight_sum += w
                    valid_count += 1
                except (ValueError, TypeError):
                    continue

        if valid_count == 0 or weight_sum == 0:
            return None

        geo_mean = math.exp(log_sum / weight_sum)
        return max(self.FLOOR, min(self.CEILING, geo_mean))
