"""
Conditional edge routing predicates for the Master Agent LangGraph.
"""
from __future__ import annotations
from .state import AgentState


def route_after_compatibility(state: AgentState) -> str:
    """
    Inspect compatibility result.
    If incompatible or error occurred, bypass tool execution directly to synthesis.
    """
    if not state.get("is_compatible", True) or state.get("error"):
        return "llm_synthesis"
    return "master_router"


def route_after_master_router(state: AgentState) -> str:
    """
    Inspect Master Agent routing decision.
    If routing succeeded, execute specialist tool; otherwise exit to synthesis.
    """
    if state.get("selected_tool") and not state.get("error"):
        return "execute_specialist_tool"
    return "llm_synthesis"
