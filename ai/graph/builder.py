"""
LangGraph StateGraph assembly for the SatQuery AI Master Agent Core.
"""
from __future__ import annotations
from langgraph.graph import StateGraph, END
from .state import AgentState
from . import nodes, edges


def build_graph():
    """
    Assemble and compile the SatQuery AI Master Agent StateGraph.

    Canonical Topology:
        START
          ↓
        validate_inputs
          ↓
        classify_intent
          ↓
        compatibility_check ──(incompatible)──→ llm_synthesis ──→ END
          ↓ (compatible)
        master_router ───────(routing error)──→ llm_synthesis ──→ END
          ↓ (routed)
        execute_specialist_tool
          ↓
        standardize_results
          ↓
        gis_processor
          ↓
        evidence_confidence
          ↓
        llm_synthesis
          ↓
         END

    Returns:
        CompiledStateGraph ready for invocation via .invoke()
    """
    graph = StateGraph(AgentState)

    # --- Register Nodes ---
    graph.add_node("validate_inputs",          nodes.validate_inputs_node)
    graph.add_node("classify_intent",          nodes.classify_intent_node)
    graph.add_node("compatibility_check",      nodes.compatibility_check_node)
    graph.add_node("master_router",            nodes.master_router_node)
    graph.add_node("execute_specialist_tool",  nodes.execute_specialist_tool_node)
    graph.add_node("standardize_results",      nodes.standardize_results_node)
    graph.add_node("gis_processor",            nodes.gis_processor_node)
    graph.add_node("evidence_confidence",      nodes.evidence_confidence_node)
    graph.add_node("llm_synthesis",            nodes.llm_synthesis_node)

    # --- Linear Entry Sequence ---
    graph.set_entry_point("validate_inputs")
    graph.add_edge("validate_inputs", "classify_intent")
    graph.add_edge("classify_intent", "compatibility_check")

    # --- Compatibility Gate (Conditional Branch) ---
    graph.add_conditional_edges(
        "compatibility_check",
        edges.route_after_compatibility,
        {
            "master_router": "master_router",
            "llm_synthesis": "llm_synthesis",
        }
    )

    # --- Master Router (Conditional Branch) ---
    graph.add_conditional_edges(
        "master_router",
        edges.route_after_master_router,
        {
            "execute_specialist_tool": "execute_specialist_tool",
            "llm_synthesis": "llm_synthesis",
        }
    )

    # --- Post-Execution Standardization & Synthesis Chain ---
    graph.add_edge("execute_specialist_tool",  "standardize_results")
    graph.add_edge("standardize_results",      "gis_processor")
    graph.add_edge("gis_processor",            "evidence_confidence")
    graph.add_edge("evidence_confidence",      "llm_synthesis")
    graph.add_edge("llm_synthesis",            END)

    return graph.compile()
