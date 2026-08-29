"""
Intent classification for SatQuery AI queries.
"""
from __future__ import annotations
import re
from abc import ABC, abstractmethod
from typing import List, Optional
from .schema import IntentResult, TaskType


class BaseIntentClassifier(ABC):
    """Abstract interface for all intent classifiers."""

    @abstractmethod
    def classify(self, query: str, n_images: int = 1,
                 modalities: Optional[List[str]] = None) -> IntentResult:
        """Classify a query into an IntentResult."""
        ...


class RuleBasedIntentClassifier(BaseIntentClassifier):
    """
    Deterministic rule-based and keyword-driven intent classifier.
    Fast, reliable, and auditable.
    """

    KEYWORDS: dict[str, list[str]] = {
        "ground":  ["highlight", "where is", "where are", "locate", "find", "show me", "bounding", "mark", "ground"],
        "change":  ["change", "difference", "what changed", "before and after", "compare", "increased", "decreased", "between", "construction"],
        "caption": ["describe", "caption", "summarize", "overview"],
        "fusion":  ["sar", "optical and sar", "cross-modal", "radar", "fused", "coherence", "optical + sar"],
        "vqa":     ["what is", "what are", "is there", "are there", "how many", "what type", "classify", "what is visible", "dominant"],
    }

    TARGET_PATTERNS: dict[str, list[str]] = {
        "building": ["building", "construction", "structure", "built-up", "urban", "house", "settlement"],
        "water":    ["water", "lake", "river", "flood", "reservoir", "waterbody", "pond"],
        "road":     ["road", "highway", "track", "street", "path", "transport"],
        "forest":   ["forest", "deforestation", "trees", "woodland"],
        "agriculture": ["crop", "agriculture", "field", "farm", "arable", "farming"],
        "vegetation": ["vegetation", "greenery", "grass", "scrub"],
    }

    WORKFLOWS: dict[str, list[str]] = {
        "ground":  ["T3_Ground"],
        "change":  ["T4_Change"],
        "caption": ["T2_Caption"],
        "fusion":  ["T5_OpticalSAR"],
        "vqa":     ["T1_VQA"],
    }

    def classify(self, query: str, n_images: int = 1,
                 modalities: Optional[List[str]] = None) -> IntentResult:
        q = query.lower().strip()
        mods = modalities or ["optical"]
        scores: dict[str, float] = {
            "vqa": 0.0,
            "caption": 0.0,
            "ground": 0.0,
            "change": 0.0,
            "fusion": 0.0
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
        workflow = self.WORKFLOWS.get(primary, ["T1_VQA"])

        # 3. Target extraction
        target = None
        for t_name, t_kws in self.TARGET_PATTERNS.items():
            if any(k in q for k in t_kws):
                target = t_name
                break

        # 4. Temporal extraction (e.g. "between 2020 and 2024")
        temporal = None
        match_years = re.findall(r"\b(19\d\d|20\d\d)\b", query)
        if len(match_years) >= 2:
            temporal = {"start_date": match_years[0], "end_date": match_years[1]}
        elif len(match_years) == 1:
            temporal = {"date": match_years[0]}

        confidence = min(1.0, 0.6 + (scores[primary] * 0.1))
        reasoning = (
            f"Classified task as '{primary}' (score={scores[primary]:.1f}) with target '{target or 'unspecified'}'. "
            f"Active inputs: {n_images} image(s), modalities={mods}."
        )

        return IntentResult(
            primary_task=primary,
            workflow=workflow,
            target=target,
            temporal_scope=temporal,
            spatial_scope="entire_scene",
            modality="multimodal" if set(mods) == {"optical", "sar"} else mods[0],
            scores=scores,
            confidence=confidence,
            reasoning=reasoning
        )
