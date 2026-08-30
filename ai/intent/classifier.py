"""
Intent classification engine for SatQuery AI.
Provides LLMIntentClassifier with Pydantic validation and automatic fallback to RuleBasedIntentClassifier.
Strictly separates natural language intent understanding from authoritative tool routing decisions.
"""
from __future__ import annotations
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from ai.llm.base import LLMProvider
from ai.llm.errors import LLMError
from ai.llm.provider import get_llm_provider
from .schema import IntentResult, LLMIntentPayload, TaskType

logger = logging.getLogger("satquery.intent")


class BaseIntentClassifier(ABC):
    """Abstract interface for all intent classifiers."""

    @abstractmethod
    def classify(
        self,
        query: str,
        n_images: int = 1,
        modalities: Optional[List[str]] = None,
        timestamps: Optional[List[str]] = None,
        crs: Optional[str] = None,
    ) -> IntentResult:
        """Classify a query into an IntentResult."""
        ...


class RuleBasedIntentClassifier(BaseIntentClassifier):
    """
    Deterministic rule-based and keyword-driven intent classifier.
    Fast, reliable, and serves as the baseline and robust fallback engine.
    """

    KEYWORDS: Dict[str, List[str]] = {
        "ground":  ["highlight", "where is", "where are", "locate", "find", "show me", "bounding", "mark", "ground"],
        "change":  ["change", "difference", "what changed", "before and after", "compare", "increased", "decreased", "between", "construction"],
        "caption": ["describe", "caption", "summarize", "overview"],
        "fusion":  ["sar", "optical and sar", "cross-modal", "radar", "fused", "coherence", "optical + sar"],
        "vqa":     ["what is", "what are", "is there", "are there", "how many", "what type", "classify", "what is visible", "dominant"],
    }

    TARGET_PATTERNS: Dict[str, List[str]] = {
        "building": ["building", "construction", "structure", "built-up", "urban", "house", "settlement"],
        "water":    ["water", "lake", "river", "flood", "reservoir", "waterbody", "pond"],
        "road":     ["road", "highway", "track", "street", "path", "transport"],
        "forest":   ["forest", "deforestation", "trees", "woodland"],
        "agriculture": ["crop", "agriculture", "field", "farm", "arable", "farming"],
        "vegetation": ["vegetation", "greenery", "grass", "scrub"],
    }

    DEFAULT_WORKFLOWS: Dict[str, List[str]] = {
        "ground":  ["T3_Ground"],
        "change":  ["T4_Change"],
        "caption": ["T2_Caption"],
        "fusion":  ["T5_OpticalSAR"],
        "vqa":     ["T1_VQA"],
    }

    def classify(
        self,
        query: str,
        n_images: int = 1,
        modalities: Optional[List[str]] = None,
        timestamps: Optional[List[str]] = None,
        crs: Optional[str] = None,
    ) -> IntentResult:
        q = query.lower().strip()
        mods = modalities or ["optical"]
        scores: Dict[str, float] = {
            "vqa": 0.0,
            "caption": 0.0,
            "ground": 0.0,
            "change": 0.0,
            "fusion": 0.0,
        }

        # 1. Keyword scoring
        for task, kws in self.KEYWORDS.items():
            for kw in kws:
                if kw in q:
                    scores[task] += 1.0

        # 2. Structural / Modality signals
        if n_images == 2:
            mset = set(mods)
            if mset == {"optical", "sar"} or "sar" in q:
                scores["fusion"] += 3.5
            else:
                scores["change"] += 3.0

        # If query has no hits, fallback to VQA as baseline
        if all(v == 0.0 for v in scores.values()):
            scores["vqa"] = 1.0

        primary: TaskType = max(scores, key=scores.get)  # type: ignore[arg-type]
        workflow = self.DEFAULT_WORKFLOWS.get(primary, ["T1_VQA"])

        # 3. Target extraction
        target = None
        for t_name, t_kws in self.TARGET_PATTERNS.items():
            if any(k in q for k in t_kws):
                target = t_name
                break

        # 4. Temporal extraction
        temporal = None
        match_years = re.findall(r"\b(19\d\d|20\d\d)\b", query)
        if len(match_years) >= 2:
            temporal = {"start_date": match_years[0], "end_date": match_years[1]}
        elif len(match_years) == 1:
            temporal = {"date": match_years[0]}

        confidence = min(1.0, 0.6 + (scores[primary] * 0.1))
        reasoning = (
            f"Rule-based classification: task='{primary}' (score={scores[primary]:.1f}) "
            f"with target='{target or 'unspecified'}'. Active inputs: {n_images} image(s), modalities={mods}."
        )

        return IntentResult(
            primary_task=primary,
            workflow=workflow,
            target=target,
            temporal_scope=temporal,
            spatial_scope="entire_scene",
            modality="multimodal" if set(mods) == {"optical", "sar"} else mods[0],
            requires_temporal_pair=(primary == "change" or bool(temporal and len(temporal) > 1)),
            requires_cross_modal_pair=(primary == "fusion"),
            ambiguous=False,
            clarification_needed=False,
            scores=scores,
            confidence=confidence,
            confidence_status="calibrated",
            reasoning=reasoning,
            classifier_source="rule_primary",
            fallback_used=False,
            fallback_reason=None,
        )


class LLMIntentClassifier(BaseIntentClassifier):
    """
    LLM-powered Intent Classifier for SatQuery AI.
    Interprets natural language queries, extracts remote-sensing concepts, validates JSON output,
    and seamlessly falls back to RuleBasedIntentClassifier on transient or schema failures.
    """

    SYSTEM_PROMPT = """You are the SatQuery AI intent classifier specializing in remote-sensing satellite imagery analysis.
Your job is to analyze the user's natural language query and context metadata, and classify the remote-sensing task.

Allowed 'task' values (CHOOSE EXACTLY ONE):
- 'vqa': Questions asking about visible objects, land cover classes, scene properties, presence of features, or scene facts.
- 'caption': Requests to describe, summarize, or provide an overview of the satellite scene.
- 'ground': Requests to locate, pinpoint, mark, find, or highlight specific objects or spatial features.
- 'change': Requests involving temporal before/after comparison, differences over time, new construction, deforestation, or urban growth.
- 'fusion': Requests explicitly requiring cross-modal analysis combining Optical and SAR (radar) imagery.

CRITICAL RULES:
1. Return JSON ONLY conforming strictly to the requested schema.
2. DO NOT output tool names or tool IDs (e.g. do NOT output 'T1_VQA', 'T4_Change', etc.). Authoritative tool selection is handled by downstream compatibility rules.
3. DO NOT invent measurements, geographic coordinates, bounding box numbers, or raster metadata.
4. Set 'requires_temporal_pair': true if the query conceptually demands multi-date imagery (e.g. change detection).
5. Set 'requires_cross_modal_pair': true if the query demands simultaneous optical + SAR data.
6. Set 'ambiguous': true and 'clarification_needed': true if the query is fundamentally ambiguous, empty, or lacks clear remote-sensing intent.
7. Set 'target' to the primary object or entity mentioned (e.g. 'building', 'water', 'forest', 'road', 'agriculture', 'aircraft', 'ship').
8. Extract temporal years or dates if mentioned (e.g. temporal_scope: {'start_date': '2020', 'end_date': '2024'})."""

    DEFAULT_WORKFLOWS: Dict[str, List[str]] = {
        "vqa": ["T1_VQA"],
        "caption": ["T2_Caption"],
        "ground": ["T3_Ground"],
        "change": ["T4_Change"],
        "fusion": ["T5_OpticalSAR"],
    }

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        fallback_classifier: Optional[BaseIntentClassifier] = None,
    ):
        self.provider = provider or get_llm_provider()
        self.fallback_classifier = fallback_classifier or RuleBasedIntentClassifier()

    def classify(
        self,
        query: str,
        n_images: int = 1,
        modalities: Optional[List[str]] = None,
        timestamps: Optional[List[str]] = None,
        crs: Optional[str] = None,
    ) -> IntentResult:
        """
        Classify user query using the LLM provider with structured output and fallback.
        """
        mods = modalities or ["optical"]

        # Build clean, minimal metadata payload
        context_payload = {
            "query": query,
            "image_count": n_images,
            "modalities": mods,
            "timestamps": timestamps or [],
            "crs_available": bool(crs),
            "cross_modal_available": set(mods) == {"optical", "sar"},
        }

        user_content = (
            f"Analyze the following remote sensing query and metadata:\n"
            f"```json\n{json.dumps(context_payload, indent=2)}\n```\n\n"
            f"Respond with a JSON object containing: task, target, modality, temporal_scope, "
            f"spatial_scope, requires_temporal_pair, requires_cross_modal_pair, ambiguous, "
            f"clarification_needed, reasoning."
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            # 1. Execute LLM call with structured JSON request
            resp = self.provider.generate_sync(
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            # 2. Parse JSON
            raw_data = resp.json()

            # 3. Validate with Pydantic schema
            payload = LLMIntentPayload.model_validate(raw_data)

            # 4. Derive default deterministic workflow downstream (LLM does not choose tool)
            workflow = self.DEFAULT_WORKFLOWS.get(payload.task, ["T1_VQA"])

            return IntentResult(
                primary_task=payload.task,
                workflow=workflow,
                target=payload.target,
                temporal_scope=payload.temporal_scope,
                spatial_scope=payload.spatial_scope or "entire_scene",
                modality=payload.modality or ("multimodal" if payload.requires_cross_modal_pair else mods[0]),
                requires_temporal_pair=payload.requires_temporal_pair,
                requires_cross_modal_pair=payload.requires_cross_modal_pair,
                ambiguous=payload.ambiguous,
                clarification_needed=payload.clarification_needed,
                confidence=None,  # Uncalibrated LLM confidence
                confidence_status="uncalibrated",
                reasoning=payload.reasoning or f"LLM classified task as '{payload.task}'.",
                classifier_source="llm",
                fallback_used=False,
                fallback_reason=None,
                raw_llm_response=raw_data,
            )

        except Exception as e:
            fallback_reason = f"{e.__class__.__name__}: {str(e)}"
            logger.warning(
                f"LLMIntentClassifier encountered failure ({fallback_reason}). "
                f"Explicitly activating RuleBasedIntentClassifier fallback."
            )

            # Execute explicit fallback
            fallback_res = self.fallback_classifier.classify(
                query=query,
                n_images=n_images,
                modalities=mods,
                timestamps=timestamps,
                crs=crs,
            )
            fallback_res.classifier_source = "rule_fallback"
            fallback_res.fallback_used = True
            fallback_res.fallback_reason = fallback_reason
            return fallback_res
