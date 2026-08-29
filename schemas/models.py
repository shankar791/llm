"""
Pydantic data-contract models shared across the entire SatQuery AI stack.
All tool inputs, outputs, API responses, and inter-node state payloads conform to these schemas.
"""
from __future__ import annotations
from typing import Any, Optional, List, Dict, Literal
from pydantic import BaseModel, Field
import uuid


class RasterReference(BaseModel):
    """Reference to an uploaded or catalog satellite raster."""
    filename: str = Field(..., description="File name or catalog identifier")
    modality: str = Field(default="optical", description="Sensor modality: 'optical', 'sar', 'multispectral'")
    crs: Optional[str] = Field(default="EPSG:4326", description="Coordinate reference system")
    bounds: Optional[List[float]] = Field(default=None, description="Bounding box [minx, miny, maxx, maxy]")
    acquisition_date: Optional[str] = Field(default=None, description="Acquisition date (YYYY-MM-DD)")
    resolution_m: Optional[float] = Field(default=None, description="Ground sample distance in meters")


class QueryRequest(BaseModel):
    """Inbound query from the frontend or API client."""
    query: str = Field(..., description="Natural-language question about the imagery")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()),
                            description="Client session ID for memory continuity")
    rasters: List[RasterReference] = Field(default_factory=list,
                                           description="List of associated raster image references")
    metadata: Dict[str, Any] = Field(default_factory=dict,
                                     description="Optional metadata or client parameters")


class IntentSchema(BaseModel):
    """Structured representation of parsed user intent."""
    task: Literal["vqa", "caption", "ground", "change", "fusion", "impact", "scenario"] = Field(
        ..., description="Primary remote-sensing task type"
    )
    target: Optional[str] = Field(default=None, description="Target entity, e.g. 'building', 'water', 'forest'")
    temporal_scope: Optional[Dict[str, str]] = Field(
        default=None, description="Temporal bounds, e.g. {'start_date': '2020', 'end_date': '2024'}"
    )
    spatial_scope: Optional[str] = Field(default="entire_scene", description="Spatial scope or ROI name")
    modality: Optional[str] = Field(default="optical", description="Required sensor modality")
    workflow: List[str] = Field(default_factory=list, description="Ordered list of specialist tool IDs to execute")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Intent classification confidence")
    reasoning: str = Field(default="", description="Human-readable classification rationale")


class CompatibilityResult(BaseModel):
    """Outcome of verifying query requirements against available raster data."""
    compatible: bool = Field(..., description="Whether data satisfies task requirements")
    missing_requirements: List[str] = Field(default_factory=list, description="List of missing requirements if any")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings (e.g. minor GSD mismatch)")
    explanation: str = Field(default="", description="User-readable compatibility summary")
    validated_tool_ids: List[str] = Field(default_factory=list, description="Approved tool IDs for execution")


class ToolRequest(BaseModel):
    """Standard input payload dispatched to any specialist tool."""
    tool_id: str = Field(..., description="Target specialist tool identifier (e.g. 'T4_Change')")
    query: str = Field(default="", description="Natural-language prompt or entity description")
    rasters: List[RasterReference] = Field(default_factory=list, description="Input raster references")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool-specific operational parameters")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session and context metadata")


class EvidenceItem(BaseModel):
    """A single piece of spatial evidence produced by a specialist tool."""
    tool_id: str = Field(..., description="Specialist tool identifier (e.g. 'T4_Change')")
    label: str = Field(..., description="Semantic entity class or finding tag")
    coverage_pct: float = Field(..., ge=0.0, le=100.0, description="Spatial coverage percentage")
    bbox_pixels: Optional[List[int]] = Field(default=None, description="[ymin, xmin, ymax, xmax] pixel bounding box")
    geojson_feature: Optional[Dict[str, Any]] = Field(default=None, description="Standard GeoJSON Feature object")


class ToolResult(BaseModel):
    """Standardised output from any specialist tool — the common currency of the agent."""
    tool_id: str = Field(..., description="Identifier of the executing tool")
    answer: str = Field(..., description="Tool-level textual summary of findings")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence score [0.0, 1.0]")
    evidence: List[EvidenceItem] = Field(default_factory=list, description="Verified spatial evidence items")
    evidence_image_b64: Optional[str] = Field(default=None, description="Base64 encoded PNG overlay data URI")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Tool-specific quantitative metrics")


class TraceStepSchema(BaseModel):
    """Single step in the execution trace."""
    step: int
    node: str
    status: str
    duration_ms: int
    detail: Optional[Dict[str, Any]] = None


class AgentResponse(BaseModel):
    """Complete response returned to the frontend after all tool execution."""
    session_id: str
    query: str
    final_answer: str
    confidence: float
    tool_results: List[ToolResult]
    geojson: Optional[Dict[str, Any]] = Field(default=None, description="Canonical GeoJSON FeatureCollection")
    trace_id: str
    elapsed_ms: int
