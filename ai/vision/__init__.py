"""
ai.vision — Multimodal Remote Sensing Vision subsystem for SatQuery AI.

Exports:
  VisionProvider (Protocol), VisionResponse, GroundingBox, GroundingResult
  VisionConfig, get_vision_provider
  OpenRouterQwenVisionProvider, MockVisionProvider
  VisionError and subclasses
"""
from __future__ import annotations
from typing import Optional

from .base import (
    GroundingBox,
    GroundingResult,
    TaskType,
    VisionProvider,
    VisionResponse,
)
from .config import VisionConfig
from .errors import (
    GroundingParseError,
    VisionAuthenticationError,
    VisionConfigurationError,
    VisionError,
    VisionNetworkError,
    VisionRateLimitError,
    VisionResponseError,
    VisionTimeoutError,
)
from .mock import MockVisionProvider
from .openrouter_qwen import OpenRouterQwenVisionProvider


def get_vision_provider(config: Optional[VisionConfig] = None) -> VisionProvider:
    """
    Factory to instantiate the configured VisionProvider.
    Defaults to OpenRouterQwenVisionProvider unless overridden.
    """
    cfg = config or VisionConfig.from_env()
    provider_key = cfg.provider.lower().strip()

    if provider_key in {"mock", "test"}:
        return MockVisionProvider()
    elif provider_key in {"openrouter", "qwen_openrouter", "openrouter_qwen", "qwen"}:
        return OpenRouterQwenVisionProvider(config=cfg)
    else:
        # Default to OpenRouter Qwen
        return OpenRouterQwenVisionProvider(config=cfg)


__all__ = [
    "GroundingBox",
    "GroundingResult",
    "TaskType",
    "VisionProvider",
    "VisionResponse",
    "VisionConfig",
    "get_vision_provider",
    "OpenRouterQwenVisionProvider",
    "MockVisionProvider",
    "VisionError",
    "VisionConfigurationError",
    "VisionAuthenticationError",
    "VisionRateLimitError",
    "VisionTimeoutError",
    "VisionNetworkError",
    "VisionResponseError",
    "GroundingParseError",
]
