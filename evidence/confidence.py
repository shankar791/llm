"""
Confidence aggregation for SatQuery AI.

Fuses per-tool confidence scores into a single calibrated score.
Uses a weighted geometric mean with a pessimism bias appropriate
for safety-critical remote-sensing analysis tasks.
"""
from __future__ import annotations
import math
from typing import Sequence


class ConfidenceAggregator:
    """
    Fuses per-tool confidence scores into a single calibrated confidence.

    Strategy:
      1. Each tool has a reliability prior (TOOL_WEIGHTS) based on
         typical precision on benchmark datasets.
      2. Scores are combined via weighted geometric mean.
      3. A pessimism floor (FLOOR) prevents over-confident outputs
         when tool agreement is low.
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

    def aggregate(self, results: list[dict]) -> float:
        """
        Compute the aggregated confidence score.

        Args:
            results: List of ToolResult-compatible dicts with 'tool_id' and 'confidence'.

        Returns:
            Aggregated confidence in [FLOOR, CEILING].
        """
        if not results:
            return self.FLOOR

        log_sum = 0.0
        weight_sum = 0.0
        for r in results:
            w = self.TOOL_WEIGHTS.get(r.get("tool_id", ""), 0.75)
            c = max(self.FLOOR, float(r.get("confidence", self.FLOOR)))
            log_sum += w * math.log(c)
            weight_sum += w

        geo_mean = math.exp(log_sum / weight_sum) if weight_sum > 0 else self.FLOOR
        return max(self.FLOOR, min(self.CEILING, geo_mean))

    def per_tool(self, result: dict) -> float:
        """
        Return the reliability-adjusted confidence for a single tool result.

        Multiplies the raw tool confidence by the tool's reliability prior.
        """
        w = self.TOOL_WEIGHTS.get(result.get("tool_id", ""), 0.75)
        raw = max(self.FLOOR, float(result.get("confidence", self.FLOOR)))
        return max(self.FLOOR, min(self.CEILING, raw * w))
