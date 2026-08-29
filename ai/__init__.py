"""
ai — SatQuery AI orchestration layer.

This package contains the LangGraph-based agent graph, intent classification,
tool compatibility routing, session memory, and LLM synthesis.

Sub-packages:
  ai.graph        — LangGraph StateGraph definition (nodes, edges, builder)
  ai.intent       — Query intent classification (schema + classifiers)
  ai.compatibility — Tool–modality compatibility routing
  ai.memory       — Per-session conversation memory
  ai.synthesis    — LLM-based final answer synthesis
"""
from __future__ import annotations
