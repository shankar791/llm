"""
LLM-based final answer synthesis for SatQuery AI.
Synthesizes verified structured evidence into clear natural language.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional


class LLMSynthesizer:
    """
    Synthesizes structured evidence into a clear, grounded narrative.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature

    def synthesize(
        self,
        query: str,
        tool_results: List[Dict[str, Any]],
        confidence: float,
        geojson: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> str:
        """
        Produce the final grounded natural-language answer.
        """
        if error:
            return f"Analysis could not be completed: {error}"

        if not tool_results:
            return "No analysis output was generated. Please verify query and image inputs."

        answers = [r.get("answer", "") for r in tool_results if r.get("answer")]
        if not answers:
            return "Analysis completed, but no substantive findings were returned by specialist tools."

        # Template-based synthesis ensuring full empirical grounding
        return "\n\n".join(answers)
