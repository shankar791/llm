"""
AgentState — The canonical mutable state object threaded through every LangGraph node.
"""
from __future__ import annotations
from typing import Any, Optional, List, Dict
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """Complete mutable state for a single SatQuery AI query execution."""
    # --- Inbound Input Fields ---
    session_id: str
    query: str
    rasters: List[Dict[str, Any]]         # RasterReference dictionaries
    image_bytes: List[bytes]              # Raw upload bytes, one per image
    image_modalities: List[str]           # 'optical' | 'sar' per image
    image_filenames: List[str]
    metadata: Dict[str, Any]

    # --- Set by validate_inputs_node ---
    scenario: str                         # 'single_image' | 'bi_temporal_pair' | 'cross_modal_pair'
    n_images: int

    # --- Set by classify_intent_node ---
    intent: Optional[str]                 # 'vqa' | 'caption' | 'ground' | 'change' | 'fusion'
    intent_target: Optional[str]          # Target entity: 'building', 'water', 'forest', etc.
    intent_schema: Optional[Dict[str, Any]] # Serialized IntentSchema
    intent_scores: Dict[str, float]
    workflow: List[str]                   # Candidate tool IDs from intent

    # --- Set by compatibility_check_node ---
    compatibility: Optional[Dict[str, Any]] # Serialized CompatibilityResult
    is_compatible: bool

    # --- Set by master_router_node ---
    selected_tool: Optional[str]          # Approved tool ID selected by Master Agent (e.g. 'T4_Change')
    tool_request: Optional[Dict[str, Any]] # Structured ToolRequest payload
    decision_log: Optional[Dict[str, Any]] # Master Agent structured routing decision

    # --- Accumulated by specialist execution nodes ---
    tool_results: List[Dict[str, Any]]     # Standard ToolResult dictionaries

    # --- Set by gis_processor_node ---
    geojson: Optional[Dict[str, Any]]     # Canonical GeoJSON FeatureCollection

    # --- Set by evidence_confidence_node ---
    confidence: float

    # --- Set by llm_synthesis_node ---
    final_answer: Optional[str]

    # --- Audit Log (append-only) ---
    trace: List[Dict[str, Any]]

    # --- Error and Status Information ---
    error: Optional[str]
