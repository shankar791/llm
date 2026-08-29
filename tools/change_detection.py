"""
ChangeDetectionTool — Bi-temporal change detection between two satellite images.
Wraps ChangeFormer bi-temporal transformer.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
from .base import BaseTool


class ChangeDetectionTool(BaseTool):
    """Detect land-cover and structural changes across two temporal acquisitions."""
    tool_id = "T4_Change"
    description = "Bi-temporal change detection via ChangeFormer"

    def run(self, image_bytes_t0: Optional[bytes] = None,
            image_bytes_t1: Optional[bytes] = None,
            modality: str = "optical", **kwargs: Any) -> Dict[str, Any]:
        """
        Execute bi-temporal change detection.

        Returns:
            Dict conforming to schemas.models.ToolResult with change metrics and polygon footprints.
        """
        answer = "Change analysis detected 14.25 hectares (7.2% of surveyed extent) of new structural change distributed across 14 distinct clusters."
        confidence = 0.88

        evidence = [
            {
                "tool_id": self.tool_id,
                "label": "construction_change_cluster_1",
                "coverage_pct": 4.1,
                "bbox_pixels": [120, 340, 480, 810],
                "geojson_feature": {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[77.58, 12.97], [77.59, 12.97], [77.59, 12.98], [77.58, 12.98], [77.58, 12.97]]]
                    },
                    "properties": {"cluster_id": 1, "area_ha": 8.1, "severity": "MODERATE"}
                }
            },
            {
                "tool_id": self.tool_id,
                "label": "construction_change_cluster_2",
                "coverage_pct": 3.1,
                "bbox_pixels": [500, 200, 720, 450],
                "geojson_feature": {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[77.60, 12.96], [77.61, 12.96], [77.61, 12.97], [77.60, 12.97], [77.60, 12.96]]]
                    },
                    "properties": {"cluster_id": 2, "area_ha": 6.15, "severity": "MODERATE"}
                }
            }
        ]

        return {
            "tool_id": self.tool_id,
            "answer": answer,
            "confidence": confidence,
            "evidence": evidence,
            "evidence_image_b64": None,
            "metadata": {
                "mock": True,
                "change_fraction": 0.072,
                "total_change_ha": 14.25,
                "n_clusters": 14,
                "severity": "MODERATE",
                "seasonal_filtered": False
            }
        }
