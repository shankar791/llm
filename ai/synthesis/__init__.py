"""
ai.synthesis — Evidence-grounded natural language synthesis for SatQuery AI.

Exports:
  LLMSynthesizer, SynthesisResult, SynthesisClaim, SynthesisPayload, PostValidationResult
  SynthesisValidator, DeterministicFallbackFormatter
"""
from __future__ import annotations

from .schema import (
    SynthesisClaim,
    SynthesisPayload,
    PostValidationResult,
    SynthesisResult,
)
from .validator import SynthesisValidator
from .fallback import DeterministicFallbackFormatter
from .llm import LLMSynthesizer
from .formatter import format_vlm_presentation

__all__ = [
    "LLMSynthesizer",
    "SynthesisResult",
    "SynthesisClaim",
    "SynthesisPayload",
    "PostValidationResult",
    "SynthesisValidator",
    "DeterministicFallbackFormatter",
    "format_vlm_presentation",
]
