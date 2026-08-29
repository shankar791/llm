"""
ai.synthesis — LLM-based final answer synthesis for SatQuery AI.

Produces a single coherent grounded answer from all tool outputs,
the aggregated confidence score, and optional GeoJSON spatial context.

Modules:
  llm — LLMSynthesizer (Phase 0: concatenation; Phase 1: LangChain ChatModel)
"""
from __future__ import annotations
