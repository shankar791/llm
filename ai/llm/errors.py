"""
Explicit error hierarchy for LLM providers.
Ensures unambiguous error classification without masking failure reasons or leaking secrets.
"""
from __future__ import annotations
from typing import Optional


class LLMError(Exception):
    """Base class for all LLM-related exceptions."""
    def __init__(self, message: str, provider: str = "", status_code: Optional[int] = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class LLMConfigurationError(LLMError):
    """Raised when LLM configuration is missing, invalid, or unsupported."""
    pass


class LLMAuthenticationError(LLMError):
    """Raised when authentication fails (e.g. invalid or missing API key, HTTP 401/403)."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limit or quota is exceeded (HTTP 429)."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when an LLM API call times out."""
    pass


class LLMNetworkError(LLMError):
    """Raised when low-level network connectivity or DNS resolution fails."""
    pass


class LLMResponseError(LLMError):
    """Raised when the LLM returns an HTTP 5xx error or malformed payload."""
    pass
