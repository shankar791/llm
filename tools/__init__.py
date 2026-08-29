"""
SatQuery AI specialist tools registry.
"""
from .base import BaseTool, ToolExecutionError
from .vqa import VQATool
from .captioning import CaptioningTool
from .grounding import GroundingTool
from .change_detection import ChangeDetectionTool
from .optical_sar import OpticalSARTool
from .registry import ToolRegistry, ToolDefinition

TOOL_REGISTRY: dict[str, type[BaseTool]] = {
    t.tool_id: t.tool_class for t in ToolRegistry.list_tools()
}

__all__ = [
    "BaseTool",
    "ToolExecutionError",
    "VQATool",
    "CaptioningTool",
    "GroundingTool",
    "ChangeDetectionTool",
    "OpticalSARTool",
    "ToolRegistry",
    "ToolDefinition",
    "TOOL_REGISTRY",
]
