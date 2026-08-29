"""
IntentResult — Structured data schema for query intent classification.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict
from schemas.models import IntentSchema

TaskType = Literal["vqa", "caption", "ground", "change", "fusion", "impact", "scenario"]


@dataclass
class IntentResult:
    """Result of classifying a user natural-language query."""
    primary_task: TaskType
    workflow: List[str]                  # Ordered list of tool IDs, e.g. ['T4_Change']
    target: Optional[str] = None         # Target feature, e.g. 'building', 'water', 'forest'
    temporal_scope: Optional[Dict[str, str]] = None  # {'start_date': '2020', 'end_date': '2024'}
    spatial_scope: Optional[str] = "entire_scene"
    modality: Optional[str] = "optical"
    scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    reasoning: str = ""

    def to_schema(self) -> IntentSchema:
        """Convert to canonical Pydantic IntentSchema."""
        return IntentSchema(
            task=self.primary_task,
            target=self.target,
            temporal_scope=self.temporal_scope,
            spatial_scope=self.spatial_scope,
            modality=self.modality,
            workflow=self.workflow,
            confidence=self.confidence,
            reasoning=self.reasoning,
        )
