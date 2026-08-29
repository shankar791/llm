"""
Centralized Tool Registry for SatQuery AI.
Explicitly defines allowed specialist tools, requirements, and metadata.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Set, Dict, Type, Optional, List
from .base import BaseTool
from .vqa import VQATool
from .captioning import CaptioningTool
from .grounding import GroundingTool
from .change_detection import ChangeDetectionTool
from .optical_sar import OpticalSARTool


@dataclass(frozen=True)
class ToolDefinition:
    """Specification of an approved specialist remote-sensing tool."""
    tool_id: str
    name: str
    description: str
    supported_task: str
    required_modalities: Set[str]
    min_images: int
    max_images: int
    tool_class: Type[BaseTool]


class ToolRegistry:
    """
    Central repository of approved tools with strict allowlisting.
    Prevents arbitrary or unverified tool execution.
    """

    _REGISTRY: Dict[str, ToolDefinition] = {
        "T1_VQA": ToolDefinition(
            tool_id="T1_VQA",
            name="Single-Image VQA",
            description="Visual Question Answering over high-resolution optical imagery",
            supported_task="vqa",
            required_modalities={"optical"},
            min_images=1,
            max_images=2,
            tool_class=VQATool,
        ),
        "T2_Caption": ToolDefinition(
            tool_id="T2_Caption",
            name="Scene Captioning",
            description="Structured natural-language scene summary and land-cover description",
            supported_task="caption",
            required_modalities={"optical", "sar"},
            min_images=1,
            max_images=1,
            tool_class=CaptioningTool,
        ),
        "T3_Ground": ToolDefinition(
            tool_id="T3_Ground",
            name="Region Grounding",
            description="Text-guided spatial localization and bounding box extraction",
            supported_task="ground",
            required_modalities={"optical"},
            min_images=1,
            max_images=1,
            tool_class=GroundingTool,
        ),
        "T4_Change": ToolDefinition(
            tool_id="T4_Change",
            name="Bi-Temporal Change Detection",
            description="Difference mapping, change fraction, and cluster polygon extraction across temporal pairs",
            supported_task="change",
            required_modalities=set(),  # Both images must share same modality
            min_images=2,
            max_images=2,
            tool_class=ChangeDetectionTool,
        ),
        "T5_OpticalSAR": ToolDefinition(
            tool_id="T5_OpticalSAR",
            name="Optical+SAR Cross-Modal Fusion",
            description="Joint multimodal analysis fusing optical spectral bands with SAR backscatter texture",
            supported_task="fusion",
            required_modalities={"optical", "sar"},
            min_images=2,
            max_images=2,
            tool_class=OpticalSARTool,
        ),
    }

    @classmethod
    def get(cls, tool_id: str) -> Optional[ToolDefinition]:
        """Retrieve tool definition by ID if allowed."""
        return cls._REGISTRY.get(tool_id)

    @classmethod
    def is_allowed(cls, tool_id: str) -> bool:
        """Check if tool ID is in the approved allowlist."""
        return tool_id in cls._REGISTRY

    @classmethod
    def list_tools(cls) -> List[ToolDefinition]:
        """Return all registered tool definitions."""
        return list(cls._REGISTRY.values())

    @classmethod
    def get_tool_for_task(cls, task: str) -> Optional[ToolDefinition]:
        """Lookup primary approved tool for a given structured task."""
        for tool_def in cls._REGISTRY.values():
            if tool_def.supported_task == task:
                return tool_def
        return None

    @classmethod
    def instantiate(cls, tool_id: str) -> BaseTool:
        """Instantiate a tool instance if allowed, else raise ValueError."""
        tool_def = cls.get(tool_id)
        if not tool_def:
            raise ValueError(f"Unknown or unapproved tool ID: {tool_id!r}. Allowed: {list(cls._REGISTRY.keys())}")
        return tool_def.tool_class()
