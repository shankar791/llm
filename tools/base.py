"""
Base class for all SatQuery AI specialist tools.

All tools share a single interface: run(**kwargs) -> dict (conforming to ToolResult).
The LangGraph orchestrator invokes tools through this interface — never calling
underlying model adapters directly.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """Abstract base class for all specialist remote-sensing tools."""
    tool_id: str  # Class-level constant, e.g. 'T1_VQA', 'T4_Change'
    description: str  # Human-readable capability summary

    @abstractmethod
    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the tool.

        Returns:
            A dictionary conforming to schemas.models.ToolResult:
            {
                "tool_id": str,
                "answer": str,
                "confidence": float,
                "evidence": list[dict],
                "evidence_image_b64": str | None,
                "metadata": dict
            }

        Raises:
            ToolExecutionError: on unrecoverable tool failure.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} tool_id={self.tool_id!r}>"


class ToolExecutionError(RuntimeError):
    """Raised when a tool encounters an execution failure."""
    pass
