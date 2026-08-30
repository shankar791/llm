"""
Deterministic Anti-Hallucination Post-Validator for SatQuery AI.
Inspects structured LLM synthesis output against supplied evidence, GIS metrics, and valid evidence IDs.
"""
from __future__ import annotations
import math
import re
from typing import Any, Dict, List, Set

from .schema import PostValidationResult, SynthesisPayload


class SynthesisValidator:
    """
    Validates LLM-generated synthesis against authoritative GIS metrics and supplied evidence IDs.
    Catches numerical hallucinations, fabricated evidence references, and ungrounded certainty claims.
    """

    AREA_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:ha\b|hectares?\b)", re.IGNORECASE)
    PERCENT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
    COUNT_PATTERN = re.compile(r"(\d+)\s*(?:regions?\b|clusters?\b|polygons?\b|features?\b)", re.IGNORECASE)
    CONFIDENCE_CLAIM_PATTERN = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%\s*(?:confiden|certain)", re.IGNORECASE)

    def validate(
        self,
        payload: SynthesisPayload,
        evidence_context: Dict[str, Any],
    ) -> PostValidationResult:
        """
        Perform deterministic consistency validation.

        Args:
            payload: The parsed LLM SynthesisPayload.
            evidence_context: Dictionary containing:
                - 'valid_evidence_ids': Set[str] (e.g. {'E1', 'E2'})
                - 'gis_metrics': Dict[str, Any] (e.g. {'area_ha': 12.4, 'polygon_count': 3, 'change_fraction': 0.07})
                - 'confidence': Optional[float]
                - 'confidence_status': str ('calibrated' | 'uncalibrated')
                - 'known_dates': Set[str] (e.g. {'2020', '2024'})
        """
        violations: List[str] = []
        valid_ids: Set[str] = set(evidence_context.get("valid_evidence_ids", []))
        gis_metrics: Dict[str, Any] = evidence_context.get("gis_metrics", {})
        conf_status: str = evidence_context.get("confidence_status", "uncalibrated")

        combined_text = f"{payload.answer} " + " ".join(c.text for c in payload.claims)

        # 1. Check Evidence IDs — strictly no fake/invented IDs allowed
        for idx, claim in enumerate(payload.claims):
            for eid in claim.evidence_ids:
                if eid not in valid_ids:
                    violations.append(
                        f"Claim #{idx + 1} referenced non-existent evidence ID '{eid}'. "
                        f"Valid IDs are: {sorted(list(valid_ids)) or 'none'}."
                    )

        # 2. Check Area Values against authoritative GIS metrics
        expected_area_ha = gis_metrics.get("area_ha")
        if expected_area_ha is not None:
            found_areas = self.AREA_PATTERN.findall(combined_text)
            for raw_val in found_areas:
                try:
                    val = float(raw_val)
                    # Allow minor rounding tolerance (0.15 ha or 2%)
                    if not (math.isclose(val, expected_area_ha, rel_tol=0.03) or abs(val - expected_area_ha) < 0.2):
                        violations.append(
                            f"Hallucinated area metric: claimed {val} hectares, "
                            f"but authoritative GIS measured {expected_area_ha:.2f} hectares."
                        )
                except ValueError:
                    pass

        # 3. Check Region / Cluster Counts against GIS metrics
        expected_count = gis_metrics.get("polygon_count")
        if expected_count is not None and expected_count > 0:
            found_counts = self.COUNT_PATTERN.findall(combined_text)
            for raw_cnt in found_counts:
                try:
                    cnt = int(raw_cnt)
                    if cnt != expected_count and abs(cnt - expected_count) > 0:
                        # If the text explicitly mentions a contradictory region count
                        violations.append(
                            f"Hallucinated region count: claimed {cnt} regions/clusters, "
                            f"but GIS polygonization detected exactly {expected_count}."
                        )
                except ValueError:
                    pass

        # 4. Check Fabricated Calibrated Confidence
        if conf_status != "calibrated":
            found_conf = self.CONFIDENCE_CLAIM_PATTERN.findall(combined_text)
            if found_conf:
                violations.append(
                    f"Claimed calibrated confidence ({found_conf[0]}%) in synthesis, "
                    f"but model confidence is marked '{conf_status}'."
                )

        is_valid = len(violations) == 0
        return PostValidationResult(
            is_valid=is_valid,
            violations=violations,
            sanitized_answer=payload.answer if is_valid else None,
        )
