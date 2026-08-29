"""
VQATool — Visual Question Answering over satellite imagery.
Wraps GeoChat visual question-answering head.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from .base import BaseTool


class VQATool(BaseTool):
    """Answer natural-language questions about satellite imagery."""
    tool_id = "T1_VQA"
    description = "Visual Question Answering via GeoChat remote-sensing VQA head"

    def run(self, query: str, image_bytes: Optional[List[bytes]] = None,
            modalities: Optional[List[str]] = None, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute VQA over satellite imagery.

        Returns:
            Dict conforming to schemas.models.ToolResult.
        """
        # Deterministic mock analysis
        answer = f"Scene analysis for '{query}': Dominant land cover includes Arable land (42.5%), Dense Vegetation (35.0%), and Built-up Infrastructure (14.2%)."
        confidence = 0.85

        evidence = [
            {
                "tool_id": self.tool_id,
                "label": "Arable land",
                "coverage_pct": 42.5,
                "bbox_pixels": [0, 0, 300, 400],
                "geojson_feature": None
            },
            {
                "tool_id": self.tool_id,
                "label": "Vegetation",
                "coverage_pct": 35.0,
                "bbox_pixels": [100, 200, 500, 600],
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
                "query": query,
                "class_scores": {"Arable land": 0.425, "Vegetation": 0.35, "Built-up": 0.142}
            }
        }
