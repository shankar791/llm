"""
Configuration management for provider-agnostic LLM foundation.
Reads from environment variables with safe defaults and guarantees API key secrecy.
Default primary model: qwen/qwen3-30b-a3b:free via OpenRouter.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional


DEFAULT_LLM_MODEL = "qwen/qwen3-14b:free"
DEFAULT_LLM_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class LLMConfig:
    """Configuration container for LLM provider instances."""
    provider: str = "openai_compatible"
    model: str = DEFAULT_LLM_MODEL
    base_url: str = DEFAULT_LLM_BASE_URL
    api_key: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Create LLMConfig by parsing standard environment variables."""
        provider = os.environ.get("LLM_PROVIDER", "openai_compatible").strip().lower()
        model = os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL).strip()
        base_url = os.environ.get("LLM_BASE_URL", DEFAULT_LLM_BASE_URL).rstrip("/")
        
        # Check LLM_API_KEY, fallback to OPENROUTER_API_KEY
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")

        timeout_str = os.environ.get("LLM_TIMEOUT", "30.0")
        try:
            timeout = float(timeout_str)
        except ValueError:
            timeout = 30.0

        retries_str = os.environ.get("LLM_MAX_RETRIES", "3")
        try:
            max_retries = int(retries_str)
        except ValueError:
            max_retries = 3

        return cls(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def __repr__(self) -> str:
        """Safe string representation that strictly masks API keys."""
        masked_key = "***" if self.api_key else "None"
        return (
            f"LLMConfig(provider='{self.provider}', model='{self.model}', "
            f"base_url='{self.base_url}', api_key={masked_key}, "
            f"timeout={self.timeout}s, max_retries={self.max_retries})"
        )
