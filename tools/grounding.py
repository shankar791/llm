"""
GroundingTool — Text-guided spatial localization in satellite imagery.
Wraps GeoChat region-grounding capability.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
from .base import BaseTool


class GroundingTool(BaseTool):
    """Localize spatial regions matching a text query."""
    tool_id = "T3_Ground"
    description = "Text-guided grounding via GeoChat region-grounding capability"

    def run(self, query: str, image_bytes: Optional[bytes] = None,
            modality: str = "optical", **kwargs: Any) -> Dict[str, Any]:
        """
        Execute text-guided grounding.

        Returns:
            Dict conforming to schemas.models.ToolResult with localized bounding boxes.
        """
        target = "target entity"
        for kw in ["water", "building", "road", "tank", "storage", "forest", "field"]:
            if kw in query.lower():
                target = kw
                break

        boxes = [
            [120, 140, 260, 310],
            [340, 420, 480, 590],
            [510, 180, 620, 290]
        ]
        answer = f"Grounded 3 regions matching '{target}' across the scene with 8.4% aggregate coverage."
        confidence = 0.82

        evidence = [
            {
                "tool_id": self.tool_id,
                "label": f"grounded_{target}_{i+1}",
                "coverage_pct": 2.8,
                "bbox_pixels": box,
                "geojson_feature": None
            }
            for i, box in enumerate(boxes)
        ]

        return {
            "tool_id": self.tool_id,
            "answer": answer,
            "confidence": confidence,
            "evidence": evidence,
            "evidence_image_b64": None,
            "metadata": {
                "mock": True,
                "query": query,
                "target": target,
                "boxes_pixel": boxes,
                "count": len(boxes)
            }
        }
