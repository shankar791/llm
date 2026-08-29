"""
Execution trace recording for SatQuery AI.

Every graph node appends a TraceStep to the ExecutionTrace.
The complete trace is serialized into AgentResponse and returned
to the frontend for full auditability.
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceStep:
    """A single recorded action in the execution trace."""
    step: int
    action: str
    detail: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "action": self.action,
            "detail": self.detail,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class ExecutionTrace:
    """Complete audit log for a single agent execution."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: float = field(default_factory=time.time)
    steps: list[TraceStep] = field(default_factory=list)

    def append(self, action: str, detail: dict | None = None) -> None:
        """
        Append a new step to the trace.

        Args:
            action: Short identifier for the action (e.g. 'validate_inputs').
            detail: Arbitrary key-value pairs for the audit record.
        """
        step = TraceStep(
            step=len(self.steps) + 1,
            action=action,
            detail=detail or {},
            duration_ms=round((time.time() - self.started_at) * 1000),
        )
        self.steps.append(step)

    def to_dict(self) -> dict:
        """Serialize the trace to a JSON-serializable dict."""
        return {
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "elapsed_ms": round((time.time() - self.started_at) * 1000),
            "steps": [s.to_dict() for s in self.steps],
        }
