"""
Base abstractions and protocol definitions for provider-agnostic LLM foundation.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class LLMResponse:
    """Standardized response container from any LLM provider."""
    content: str
    model: str
    provider: str
    latency_ms: float
    usage: Optional[Dict[str, int]] = field(default=None)  # {"prompt_tokens": X, "completion_tokens": Y, "total_tokens": Z}
    finish_reason: Optional[str] = field(default=None)
    raw_data: Optional[Dict[str, Any]] = field(default=None)

    def json(self) -> Dict[str, Any]:
        """Parse text content as a structured JSON object."""
        text = self.content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"LLM output is not valid JSON: {e}\nRaw output: {self.content}") from e


@runtime_checkable
class LLMProvider(Protocol):
    """
    Protocol defining the vendor-agnostic interface for LLM completions.
    Supports both asynchronous and synchronous execution with optional structured JSON output.
    """

    async def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        """Asynchronously generate a completion for the given conversation messages."""
        ...

    def generate_sync(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        """Synchronously generate a completion for the given conversation messages."""
        ...
