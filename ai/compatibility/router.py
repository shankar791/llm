"""
ToolCompatibilityRouter — Validates query intent against input raster metadata.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set, Any
from ai.intent.schema import IntentResult
from schemas.models import CompatibilityResult


@dataclass
class CompatibilityCheckResult:
    """Outcome of verifying query requirements against uploaded raster data."""
    compatible: bool
    missing_requirements: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    explanation: str = ""
    validated_tool_ids: List[str] = field(default_factory=list)

    def to_schema(self) -> CompatibilityResult:
        """Convert to canonical Pydantic CompatibilityResult."""
        return CompatibilityResult(
            compatible=self.compatible,
            missing_requirements=self.missing_requirements,
            warnings=self.warnings,
            explanation=self.explanation,
            validated_tool_ids=self.validated_tool_ids,
        )


class ToolCompatibilityRouter:
    """Validates workflow requirements against active raster modalities and count."""

    def check_compatibility(self, intent: IntentResult, n_images: int,
                            modalities: List[str]) -> CompatibilityCheckResult:
        """
        Verify whether the available data satisfies the intent requirements.
        """
        missing = []
        warnings = []
        available_mods = set(modalities)

        # 1. Task-specific image count and modality checks
        if intent.primary_task == "change":
            if n_images < 2:
                missing.append(f"Change detection requires 2 images from distinct dates, but only {n_images} was provided.")
            elif len(available_mods) > 1 and available_mods != {"optical"}:
                warnings.append("Change detection is most reliable between matching modalities (optical-optical or SAR-SAR).")

        elif intent.primary_task == "fusion":
            if n_images < 2:
                missing.append("Optical+SAR cross-modal fusion requires both an Optical and a SAR scene.")
            elif available_mods != {"optical", "sar"}:
                missing.append(f"Optical+SAR fusion requires both 'optical' and 'sar' imagery, got {available_mods}.")

        # 2. Tool-specific requirements
        validated_tools = []
        for tool_id in intent.workflow:
            if tool_id in {"T1_VQA", "T3_Ground"}:
                if "optical" not in available_mods:
                    missing.append(f"Tool '{tool_id}' requires optical imagery.")
                else:
                    validated_tools.append(tool_id)

            elif tool_id == "T2_Caption":
                if not (available_mods & {"optical", "sar"}):
                    missing.append("Tool 'T2_Caption' requires an optical or SAR image.")
                else:
                    validated_tools.append(tool_id)

            elif tool_id == "T4_Change":
                if n_images >= 2:
                    validated_tools.append(tool_id)

            elif tool_id == "T5_OpticalSAR":
                if {"optical", "sar"}.issubset(available_mods):
                    validated_tools.append(tool_id)

            else:
                validated_tools.append(tool_id)

        is_compatible = len(missing) == 0

        if is_compatible:
            explanation = f"Input verified: {n_images} raster(s) ({', '.join(modalities)}) fully satisfy task '{intent.primary_task}'."
        else:
            explanation = "Incompatible request: " + " ".join(missing)

        return CompatibilityCheckResult(
            compatible=is_compatible,
            missing_requirements=missing,
            warnings=warnings,
            explanation=explanation,
            validated_tool_ids=validated_tools if is_compatible else []
        )
