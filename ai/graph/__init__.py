"""
ai.graph — LangGraph StateGraph for SatQuery AI.

This sub-package assembles the agent execution graph that orchestrates
all specialist tools in response to a user query.

Modules:
  state   — AgentState TypedDict threaded through every node
  nodes   — Node functions (validate, classify, route, run tools, synthesize)
  edges   — Conditional edge predicate functions
  builder — build_graph() factory that compiles the StateGraph
"""
from __future__ import annotations
