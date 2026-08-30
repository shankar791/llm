"""
IntentResult & LLMIntentPayload — Structured data schemas for query intent classification.
Enforces strict Pydantic validation on LLM JSON output while decoupling intent from tool IDs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from schemas.models import IntentSchema

TaskType = Literal["vqa", "caption", "ground", "change", "fusion"]
ExtendedTaskType = Literal["vqa", "caption", "ground", "change", "fusion", "impact", "scenario"]


class LLMIntentPayload(BaseModel):
    """
    Strict validation schema for raw JSON returned by the LLM Intent Classifier.
    Note: tool_id and workflow are deliberately EXCLUDED from LLM outputs.
    """
    task: TaskType = Field(..., description="Primary remote-sensing task: 'vqa', 'caption', 'ground', 'change', 'fusion'")
    target: Optional[str] = Field(default=None, description="Identified entity or feature (e.g. 'building', 'water', 'forest')")
    modality: Optional[str] = Field(default="optical", description="Identified required modality: 'optical', 'sar', 'multimodal'")
    temporal_scope: Optional[Dict[str, str]] = Field(default=None, description="Temporal bounds (e.g. {'start_date': '2020', 'end_date': '2024'})")
    spatial_scope: Optional[str] = Field(default="entire_scene", description="Spatial scope or ROI name")
    requires_temporal_pair: bool = Field(default=False, description="Whether query conceptually requires images from multiple dates")
    requires_cross_modal_pair: bool = Field(default=False, description="Whether query conceptually requires optical+SAR cross-modal pair")
    ambiguous: bool = Field(default=False, description="Whether the user request is ambiguous or underspecified")
    clarification_needed: bool = Field(default=False, description="Whether user clarification should be requested")
    reasoning: str = Field(default="", description="Chain-of-thought rationale explaining classification")


@dataclass
class IntentResult:
    """
    Internal container passed across the Master Agent state machine.
    Workflow is derived deterministically downstream — NEVER dictated by the LLM.
    """
    primary_task: ExtendedTaskType
    workflow: List[str]                  # Derived deterministically downstream
    target: Optional[str] = None
    temporal_scope: Optional[Dict[str, str]] = None
    spatial_scope: Optional[str] = "entire_scene"
    modality: Optional[str] = "optical"
    requires_temporal_pair: bool = False
    requires_cross_modal_pair: bool = False
    ambiguous: bool = False
    clarification_needed: bool = False
    scores: Dict[str, float] = field(default_factory=dict)
    confidence: Optional[float] = None
    confidence_status: str = "uncalibrated"
    reasoning: str = ""
    classifier_source: Literal["llm", "rule_fallback", "rule_primary"] = "rule_primary"
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    raw_llm_response: Optional[Dict[str, Any]] = None

    def to_schema(self) -> IntentSchema:
        """Convert to canonical Pydantic IntentSchema."""
        return IntentSchema(
            task=self.primary_task,
            target=self.target,
            temporal_scope=self.temporal_scope,
            spatial_scope=self.spatial_scope,
            modality=self.modality,
            requires_temporal_pair=self.requires_temporal_pair,
            requires_cross_modal_pair=self.requires_cross_modal_pair,
            ambiguous=self.ambiguous,
            clarification_needed=self.clarification_needed,
            classifier_source=self.classifier_source,
            fallback_used=self.fallback_used,
            fallback_reason=self.fallback_reason,
            workflow=self.workflow,
            confidence=self.confidence,
            confidence_status=self.confidence_status,
            reasoning=self.reasoning,
        )

    @classmethod
    def from_schema(cls, schema: IntentSchema) -> IntentResult:
        """Construct IntentResult from canonical IntentSchema."""
        return cls(
            primary_task=schema.task,
            workflow=schema.workflow,
            target=schema.target,
            temporal_scope=schema.temporal_scope,
            spatial_scope=schema.spatial_scope,
            modality=schema.modality,
            requires_temporal_pair=schema.requires_temporal_pair,
            requires_cross_modal_pair=schema.requires_cross_modal_pair,
            ambiguous=schema.ambiguous,
            clarification_needed=schema.clarification_needed,
            confidence=schema.confidence,
            confidence_status=schema.confidence_status,
            reasoning=schema.reasoning,
            classifier_source=schema.classifier_source,  # type: ignore[arg-type]
            fallback_used=schema.fallback_used,
            fallback_reason=schema.fallback_reason,
        )
