"""
Typed exception hierarchy for the SatQuery AI Vision subsystem.
"""
from __future__ import annotations
from typing import Optional


class VisionError(Exception):
    """Base exception for all vision provider errors."""
    def __init__(self, message: str, provider: Optional[str] = None, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code


class VisionConfigurationError(VisionError):
    """Raised when provider configuration, endpoint URL, or required credentials are missing."""
    pass


class VisionAuthenticationError(VisionError):
    """Raised on HTTP 401/403 or invalid API keys."""
    pass


class VisionRateLimitError(VisionError):
    """Raised on HTTP 429 rate-limiting from remote providers."""
    pass


class VisionTimeoutError(VisionError):
    """Raised when an HTTP request or model inference times out."""
    pass


class VisionNetworkError(VisionError):
    """Raised on connection failures, DNS errors, or socket errors."""
    pass


class VisionResponseError(VisionError):
    """Raised when the provider returns a non-200 status or unexpected payload format."""
    pass


class GroundingParseError(VisionError):
    """Raised when structured bounding box extraction fails or emits invalid coordinates."""
    pass
