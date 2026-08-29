"""
CaptioningTool — Structured scene description for satellite imagery.
Wraps GeoChat captioning head.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
from .base import BaseTool


class CaptioningTool(BaseTool):
    """Generate structured natural-language descriptions of satellite imagery."""
    tool_id = "T2_Caption"
    description = "Scene captioning via GeoChat remote-sensing captioning head"

    def run(self, image_bytes: Optional[bytes] = None, modality: str = "optical",
            **kwargs: Any) -> Dict[str, Any]:
        """
        Execute scene captioning.

        Returns:
            Dict conforming to schemas.models.ToolResult.
        """
        answer = "High-resolution satellite capture depicting mixed suburban terrain with planned road networks, agricultural parcels, and scattered commercial clusters."
        confidence = 0.88

        evidence = [
            {
                "tool_id": self.tool_id,
                "label": "Scene context",
                "coverage_pct": 100.0,
                "bbox_pixels": None,
                "geojson_feature": None
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
                "modality": modality,
                "terrain_type": "suburban_mixed"
            }
        }
