"""
VQATool — Visual Question Answering over satellite imagery (T1_VQA).
Backends: Qwen2.5-VL via OpenRouter (primary) and GeoChatAdapter (optional specialist).
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Union, Literal
import numpy as np
from PIL import Image

from .base import BaseTool, ToolExecutionError
from ai.vision.base import VisionProvider
from ai.vision import get_vision_provider
from models.geochat.adapter import GeoChatAdapter


class VQATool(BaseTool):
    """Answer natural-language questions about satellite imagery via Qwen2.5-VL / GeoChat."""
    tool_id = "T1_VQA"
    description = "Visual Question Answering via multimodal vision provider (Qwen2.5-VL / GeoChat)"

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
        modalities: Optional[List[str]] = None,
        mode: Optional[Literal["real", "mock"]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Execute VQA over satellite imagery.

        Returns:
            Dict strictly conforming to schemas.models.ToolResult.
        """
        if not query or not query.strip():
            raise ToolExecutionError("VQATool requires a non-empty query string.")

        active_mode = mode or kwargs.get("mode") or self.mode

        # Extract primary single image
        if isinstance(image_bytes, list):
            img_input = image_bytes[0] if len(image_bytes) > 0 else None
        else:
            img_input = image_bytes

        if img_input is None:
            raise ToolExecutionError("VQATool requires a valid input image.")

        # 1. Mock execution mode (preserves fast offline test suite)
        if active_mode == "mock":
            adapter = self._get_geochat_adapter(mode="mock")
            return adapter.vqa(image=img_input, question=query, mode="mock")

        # 2. Real execution mode: Check provider routing
        provider_name = os.environ.get("VISION_PROVIDER", "qwen_openrouter").lower()

        if provider_name == "geochat":
            try:
                adapter = self._get_geochat_adapter(mode="real")
                return adapter.vqa(image=img_input, question=query, mode="real")
            except Exception as e:
                raise ToolExecutionError(f"GeoChat VQA execution failed: {e}") from e

        # Default to Qwen VisionProvider (OpenRouter)
        try:
            provider = self.vision_provider or get_vision_provider()
            resp = provider.analyze_image_sync(
                image_input=img_input,
                prompt=query,
                task="vqa",
                **kwargs,
            )
            return {
                "tool_id": self.tool_id,
                "answer": resp.text,
                "confidence": None,
                "confidence_status": "uncalibrated",
                "evidence": [],
                "evidence_image_b64": None,
                "metadata": {
                    "provider": resp.provider,
                    "model": resp.model,
                    "latency_ms": resp.latency_ms,
                    "mode": "remote",
                },
            }
        except Exception as e:
            raise ToolExecutionError(f"Qwen VQA execution failed: {e}") from e
