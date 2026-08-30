"""
ai.intent — Query intent classification for SatQuery AI.

Classifies a natural-language query into one of the supported satellite-analysis
task types: vqa, caption, ground, change, fusion.

Exports:
  IntentResult, LLMIntentPayload, TaskType
  BaseIntentClassifier, RuleBasedIntentClassifier, LLMIntentClassifier
"""
from __future__ import annotations

from .schema import IntentResult, LLMIntentPayload, TaskType
from .classifier import (
    BaseIntentClassifier,
    RuleBasedIntentClassifier,
    LLMIntentClassifier,
)

__all__ = [
    "IntentResult",
    "LLMIntentPayload",
    "TaskType",
    "BaseIntentClassifier",
    "RuleBasedIntentClassifier",
    "LLMIntentClassifier",
]
