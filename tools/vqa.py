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

        # 2. Real execution mode: Route through configured VisionProvider (GeoChat or OpenRouter)
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
                    "selected_model": resp.selected_model or resp.model,
                    "attempted_models": resp.attempted_models or [resp.model],
                    "fallback_used": resp.fallback_used,
                    "fallback_reason": resp.fallback_reason,
                    "latency_ms": resp.latency_ms,
                    "mode": "remote",
                },
            }
        except Exception as e:
            # 1. If GeoChat / primary provider fails, attempt secondary live vision models (OpenRouter VLM)
            try:
                from ai.vision.openrouter_qwen import OpenRouterVisionProvider
                sec_provider = OpenRouterVisionProvider()
                sec_resp = sec_provider.analyze_image_sync(
                    image_input=img_input,
                    prompt=query,
                    task="vqa",
                    **kwargs,
                )
                return {
                    "tool_id": self.tool_id,
                    "answer": sec_resp.text,
                    "confidence": None,
                    "confidence_status": "uncalibrated",
                    "evidence": [],
                    "evidence_image_b64": None,
                    "metadata": {
                        "provider": "openrouter",
                        "model": sec_resp.selected_model or sec_resp.model or "Gemma-4-26B",
                        "selected_model": sec_resp.selected_model or sec_resp.model,
                        "attempted_models": ["geochat", sec_resp.model],
                        "fallback_used": True,
                        "fallback_reason": f"GeoChat unavailable; automatically switched to VLM model: {sec_resp.selected_model or sec_resp.model}",
                        "latency_ms": sec_resp.latency_ms,
                        "mode": "remote",
                    },
                }
            except Exception as sec_e:
                # 2. Deterministic local fallback if all remote vision services fail
                adapter = self._get_geochat_adapter(mode="mock")
                fallback_res = adapter.vqa(image=img_input, question=query, mode="mock")

                clean_sec_err = str(sec_e)
                if "rate limit" in clean_sec_err.lower() or "429" in clean_sec_err:
                    vlm_reason = "OpenRouter free quota exhausted (429 Rate Limit)"
                else:
                    vlm_reason = f"VLM failure ({type(sec_e).__name__})"

                fallback_reason = f"GeoChat ({type(e).__name__}) and {vlm_reason}"

                fallback_res["metadata"].update({
                    "provider": "geochat",
                    "model": "GeoChat-7B",
                    "http_status": http_status,
                    "exception_type": type(e).__name__,
                    "exception_message": clean_err,
                    "vlm_fallback_error": vlm_reason,
                    "fallback_used": True,
                    "fallback_reason": fallback_reason,
                    "status": "fallback",
                })
                return fallback_res
