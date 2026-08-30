"""
GroundingTool — Text-guided spatial localization in satellite imagery (T3_Ground).
Backends: Qwen2.5-VL via OpenRouter (primary) and GeoChatAdapter (optional specialist).
"""
from __future__ import annotations
import io
import os
from typing import Any, Dict, List, Optional, Union, Literal
import numpy as np
from PIL import Image

from .base import BaseTool, ToolExecutionError
from ai.vision.base import VisionProvider
from ai.vision import get_vision_provider
from models.geochat.adapter import GeoChatAdapter
from schemas.models import EvidenceItem


def _get_image_dimensions(image_input: Union[bytes, np.ndarray, Image.Image]) -> tuple[int, int]:
    """Extract (width, height) from various image types."""
    if isinstance(image_input, Image.Image):
        return image_input.size
    elif isinstance(image_input, np.ndarray):
        h, w = image_input.shape[:2]
        return (w, h)
    elif isinstance(image_input, bytes):
        try:
            with Image.open(io.BytesIO(image_input)) as img:
                return img.size
        except Exception:
            return (512, 512)
    return (512, 512)


class GroundingTool(BaseTool):
    """Localize spatial regions matching a text query via Qwen2.5-VL / GeoChat."""
    tool_id = "T3_Ground"
    description = "Text-guided grounding via multimodal vision provider (Qwen2.5-VL / GeoChat)"

    def __init__(
        self,
        mode: Literal["real", "mock"] = "mock",
        checkpoint_path: Optional[str] = None,
        vision_provider: Optional[VisionProvider] = None,
    ):
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self.vision_provider = vision_provider
        self._geochat_adapter: Optional[GeoChatAdapter] = None

    def _get_geochat_adapter(self, mode: str) -> GeoChatAdapter:
        if self._geochat_adapter is None or self._geochat_adapter.mode != mode:
            self._geochat_adapter = GeoChatAdapter(checkpoint_path=self.checkpoint_path, mode=mode)
            self._geochat_adapter.load()
        return self._geochat_adapter

    def run(
        self,
        query: str,
        image_bytes: Optional[Union[bytes, List[bytes], np.ndarray, Image.Image]] = None,
        modality: str = "optical",
        mode: Optional[Literal["real", "mock"]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Execute text-guided grounding over satellite imagery.

        Returns:
            Dict strictly conforming to schemas.models.ToolResult with localized bounding boxes.
        """
        if not query or not query.strip():
            raise ToolExecutionError("GroundingTool requires a non-empty query string.")

        active_mode = mode or kwargs.get("mode") or self.mode

        if isinstance(image_bytes, list):
            img_input = image_bytes[0] if len(image_bytes) > 0 else None
        else:
            img_input = image_bytes

        if img_input is None:
            raise ToolExecutionError("GroundingTool requires a valid input image.")

        # 1. Mock execution mode
        if active_mode == "mock":
            adapter = self._get_geochat_adapter(mode="mock")
            return adapter.ground(image=img_input, target_query=query, mode="mock")

        # 2. Real execution mode
        provider_name = os.environ.get("VISION_PROVIDER", "qwen_openrouter").lower()

        if provider_name == "geochat":
            try:
                adapter = self._get_geochat_adapter(mode="real")
                return adapter.ground(image=img_input, target_query=query, mode="real")
            except Exception as e:
                raise ToolExecutionError(f"GeoChat grounding execution failed: {e}") from e

        # Default to VisionProvider (OpenRouter)
        try:
            provider = self.vision_provider or get_vision_provider()
            resp = provider.analyze_image_sync(
                image_input=img_input,
                prompt=query,
                task="ground",
                **kwargs,
            )

            img_w, img_h = _get_image_dimensions(img_input)
            total_pixels = max(1, img_w * img_h)

            evidence_items: List[Dict[str, Any]] = []

            if resp.grounding and resp.grounding.objects:
                for obj in resp.grounding.objects:
                    # Convert normalized/raw coordinates to integer [ymin, xmin, ymax, xmax]
                    pixel_box = obj.to_pixel_box(width=img_w, height=img_h)
                    ymin, xmin, ymax, xmax = pixel_box
                    box_area = max(0, (xmax - xmin) * (ymax - ymin))
                    cov_pct = round((box_area / total_pixels) * 100.0, 2)

                    item = EvidenceItem(
                        tool_id=self.tool_id,
                        label=obj.label or query,
                        coverage_pct=cov_pct,
                        bbox_pixels=pixel_box,
                    )
                    evidence_items.append(item.model_dump())

                n_found = len(evidence_items)
                answer = f"Detected {n_found} spatial region(s) matching '{query}' in satellite scene."
            else:
                n_found = 0
                answer = f"No distinct spatial regions matching '{query}' could be grounded."

            return {
                "tool_id": self.tool_id,
                "answer": answer,
                "confidence": None,
                "confidence_status": "uncalibrated",
                "evidence": evidence_items,
                "evidence_image_b64": None,
                "metadata": {
                    "provider": resp.provider,
                    "model": resp.model,
                    "selected_model": resp.selected_model or resp.model,
                    "attempted_models": resp.attempted_models or [resp.model],
                    "fallback_used": resp.fallback_used,
                    "fallback_reason": resp.fallback_reason,
                    "latency_ms": resp.latency_ms,
                    "object_count": n_found,
                    "mode": "remote",
                },
            }

        except Exception as e:
            raise ToolExecutionError(f"Vision grounding execution failed: {e}") from e
