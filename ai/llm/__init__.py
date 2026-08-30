"""
ai.llm — Provider-agnostic LLM foundation for SatQuery AI.
"""
from __future__ import annotations

from .base import LLMProvider, LLMResponse
from .config import LLMConfig
from .errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from .provider import MockLLMProvider, OpenAICompatibleProvider, get_llm_provider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "LLMConfig",
    "LLMError",
    "LLMConfigurationError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMNetworkError",
    "LLMResponseError",
    "OpenAICompatibleProvider",
    "MockLLMProvider",
    "get_llm_provider",
]
