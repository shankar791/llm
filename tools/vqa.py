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
            mock_res = adapter.vqa(image=img_input, question=query, mode="mock")
            mock_res["metadata"].update({
                "provider": "synthetic",
                "model": "Deterministic Mock / Simulator",
                "active_tier": "synthetic",
                "attempted_tiers": ["synthetic"],
                "tier_journey": [
                    {"tier": 3, "provider": "synthetic", "model": "Deterministic Mock", "status": "success", "detail": "Direct mock execution"}
                ],
                "fallback_used": False,
                "fallback_reason": None,
            })
            return mock_res

        # 2. Real execution mode: Multi-tier cascade GeoChat -> OpenRouter -> Synthetic
        tier_journey: List[Dict[str, Any]] = []

        # If custom vision provider is injected (e.g. in test suite), prioritize it directly
        if self.vision_provider is not None:
            try:
                resp = self.vision_provider.analyze_image_sync(
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
                        "active_tier": resp.provider,
                        "attempted_tiers": [resp.provider],
                        "tier_journey": [
                            {"tier": 1, "provider": resp.provider, "model": resp.model, "status": "success", "detail": "Custom injected vision provider"}
                        ],
                        "fallback_used": resp.fallback_used,
                        "fallback_reason": resp.fallback_reason,
                        "latency_ms": resp.latency_ms,
                        "mode": "remote",
                    },
                }
            except Exception as custom_err:
                tier_journey.append({"tier": 1, "provider": "custom_provider", "status": "failed", "detail": str(custom_err)})

        # TIER 1: GEOCHAT VLM
        geochat_err: Optional[Exception] = None
        try:
            from ai.vision.geochat import GeoChatVisionProvider
            gc_provider = GeoChatVisionProvider()
            gc_resp = gc_provider.analyze_image_sync(
                image_input=img_input,
                prompt=query,
                task="vqa",
                **kwargs,
            )
            tier_journey.append({"tier": 1, "provider": "geochat", "model": "GeoChat-7B", "status": "success", "detail": "Live GeoChat VLM inference"})
            return {
                "tool_id": self.tool_id,
                "answer": gc_resp.text,
                "confidence": None,
                "confidence_status": "uncalibrated",
                "evidence": [],
                "evidence_image_b64": None,
                "metadata": {
                    "provider": "geochat",
                    "model": "GeoChat-7B",
                    "selected_model": "GeoChat-7B",
                    "active_tier": "geochat",
                    "attempted_tiers": ["geochat"],
                    "tier_journey": tier_journey,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "latency_ms": gc_resp.latency_ms,
                    "mode": "remote",
                },
            }
        except Exception as e:
            geochat_err = e
            clean_gc_err = f"GeoChat down/unavailable ({type(e).__name__}: {e})"
            tier_journey.append({"tier": 1, "provider": "geochat", "model": "GeoChat-7B", "status": "failed", "detail": clean_gc_err})

        # TIER 2: OPENROUTER VLM FALLBACK
        openrouter_err: Optional[Exception] = None
        try:
            from ai.vision.openrouter_qwen import OpenRouterVisionProvider
            or_provider = OpenRouterVisionProvider()
            or_resp = or_provider.analyze_image_sync(
                image_input=img_input,
                prompt=query,
                task="vqa",
                **kwargs,
            )
            selected_model = or_resp.selected_model or or_resp.model or "OpenRouter-VLM"
            tier_journey.append({"tier": 2, "provider": "openrouter", "model": selected_model, "status": "success", "detail": f"Switched to OpenRouter model: {selected_model}"})
            return {
                "tool_id": self.tool_id,
                "answer": or_resp.text,
                "confidence": None,
                "confidence_status": "uncalibrated",
                "evidence": [],
                "evidence_image_b64": None,
                "metadata": {
                    "provider": "openrouter",
                    "model": selected_model,
                    "selected_model": selected_model,
                    "attempted_models": ["GeoChat-7B", selected_model],
                    "active_tier": "openrouter",
                    "attempted_tiers": ["geochat", "openrouter"],
                    "tier_journey": tier_journey,
                    "fallback_used": True,
                    "fallback_reason": f"Tier 1 (GeoChat) was down ({type(geochat_err).__name__ if geochat_err else 'unavailable'}); automatically switched to Tier 2 (OpenRouter: {selected_model})",
                    "latency_ms": or_resp.latency_ms,
                    "mode": "remote",
                },
            }
        except Exception as e:
            openrouter_err = e
            clean_or_err = f"OpenRouter down/unavailable ({type(e).__name__}: {e})"
            tier_journey.append({"tier": 2, "provider": "openrouter", "model": "OpenRouter-VLM", "status": "failed", "detail": clean_or_err})

        # TIER 3: SYNTHETIC OUTPUT (Deterministic local spectral baseline)
        adapter = self._get_geochat_adapter(mode="mock")
        fallback_res = adapter.vqa(image=img_input, question=query, mode="mock")

        tier_journey.append({"tier": 3, "provider": "synthetic", "model": "Synthetic Spectral Baseline", "status": "success", "detail": "Local deterministic rule-based spectral analysis"})

        gc_desc = f"GeoChat down ({type(geochat_err).__name__ if geochat_err else 'unavailable'})"
        or_desc = f"OpenRouter down ({type(openrouter_err).__name__ if openrouter_err else 'unavailable'})"
        fallback_reason = f"Step 1: {gc_desc} -> Step 2: {or_desc} -> Step 3: Switched to Synthetic Output"

        fallback_res["metadata"].update({
            "provider": "synthetic",
            "model": "Synthetic Spectral Baseline",
            "selected_model": "Synthetic Spectral Baseline",
            "active_tier": "synthetic",
            "attempted_tiers": ["geochat", "openrouter", "synthetic"],
            "tier_journey": tier_journey,
            "geochat_error": str(geochat_err),
            "openrouter_error": str(openrouter_err),
            "fallback_used": True,
            "fallback_reason": fallback_reason,
            "status": "synthetic_fallback",
        })
        return fallback_res
