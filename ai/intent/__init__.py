"""
ai.intent — Query intent classification for SatQuery AI.

Classifies a natural-language query into one of the supported satellite-analysis
task types: vqa, caption, ground, change, fusion.

Modules:
  schema      — IntentResult dataclass and TaskType literal
  classifier  — BaseIntentClassifier ABC + RuleBasedIntentClassifier implementation
"""
from __future__ import annotations
