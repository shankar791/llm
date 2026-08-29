# API Specification & Data Contracts

This document specifies the REST API endpoints and Pydantic data contracts connecting the Frontend, Backend API Gateway, AI Orchestrator, and Specialist Tools.

---

## REST Endpoints

### 1. Execute Query (`POST /api/query`)

Submits a natural-language query and associated satellite raster files for analysis.

- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `query` (string, required): Natural-language question.
  - `files` (array of File, required): 1 to 2 uploaded raster files (GeoTIFF, PNG, JPG).
  - `session_id` (string, optional): Client session UUID for memory continuation.

#### Successful Response (`200 OK`)
Conforms to `schemas/models.py::AgentResponse`:
```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "query": "Identify new construction between 2020 and 2024.",
  "final_answer": "Analysis between 2020 and 2024 reveals approximately 14.25 hectares of new construction activity across 14 distinct clusters in the scene.",
  "confidence": 0.88,
  "tool_results": [
    {
      "tool_id": "T4_Change",
      "answer": "14.25 ha new construction detected across 14 clusters",
      "confidence": 0.88,
      "evidence": [
        {
          "tool_id": "T4_Change",
          "label": "construction_change",
          "coverage_pct": 7.2,
          "bbox_pixels": [120, 340, 480, 810],
          "geojson_feature": {
            "type": "Feature",
            "geometry": {
              "type": "Polygon",
              "coordinates": [[[77.58, 12.97], [77.59, 12.97], [77.59, 12.98], [77.58, 12.98], [77.58, 12.97]]]
            },
            "properties": {
              "cluster_id": 1,
              "area_sqm": 42100.0,
              "severity": "MODERATE"
            }
          }
        }
      ],
      "evidence_image_b64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEU...",
      "metadata": {
        "change_fraction": 0.072,
        "n_clusters": 14,
        "algorithm": "ChangeFormer-INT8"
      }
    }
  ],
  "geojson": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "MultiPolygon",
          "coordinates": [...]
        },
        "properties": {
          "total_area_ha": 14.25,
          "task": "T4_Change"
        }
      }
    ]
  },
  "trace_id": "tr-20260829-9a8f2",
  "elapsed_ms": 238
}
```

---

### 2. Service Health Check (`GET /health`)

Returns API service availability and initialized hardware acceleration backend.

#### Response (`200 OK`)
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "device": "CPU (Intel Core Ultra 5 / OpenVINO)",
  "active_tools": ["T1_VQA", "T2_Caption", "T3_Ground", "T4_Change", "T5_OpticalSAR"]
}
```

---

### 3. Session State Query (`GET /api/session/{session_id}`) *(Proposed)*

Retrieves historical queries, active bounding boxes, and cached feature collections for follow-up refinement.

---

## Core Pydantic Schemas (`schemas/models.py`)

```python
from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict
import uuid

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural-language prompt about the imagery")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Session identifier")

class EvidenceItem(BaseModel):
    tool_id: str = Field(..., description="Specialist tool identifier (e.g. T4_Change)")
    label: str = Field(..., description="Semantic entity class or finding tag")
    coverage_pct: float = Field(..., ge=0.0, le=100.0, description="Spatial coverage percentage")
    bbox_pixels: Optional[List[int]] = Field(None, description="[ymin, xmin, ymax, xmax] pixel box")
    geojson_feature: Optional[Dict[str, Any]] = Field(None, description="Standard GeoJSON Feature object")

class ToolResult(BaseModel):
    tool_id: str = Field(..., description="Identifier of the executing tool")
    answer: str = Field(..., description="Tool-level textual summary of findings")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    evidence: List[EvidenceItem] = Field(default_factory=list, description="Verified spatial evidence items")
    evidence_image_b64: Optional[str] = Field(None, description="Base64 encoded PNG overlay data URI")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Raw quantitative metrics")

class AgentResponse(BaseModel):
    session_id: str
    query: str
    final_answer: str
    confidence: float
    tool_results: List[ToolResult]
    geojson: Optional[Dict[str, Any]] = Field(None, description="Canonical GeoJSON FeatureCollection")
    trace_id: str
    elapsed_ms: int
```

---

## Internal Orchestration Schemas (Phase 1 Expansions)

### Intent Schema (`IntentSchema`)
```json
{
  "task": "CHANGE_DETECTION",
  "target": "BUILDING",
  "temporal_scope": {
    "start_date": "2020",
    "end_date": "2024"
  },
  "spatial_scope": "entire_scene",
  "modality": "optical",
  "confidence": 0.95
}
```

### Compatibility Result (`CompatibilityResult`)
```json
{
  "compatible": true,
  "missing_requirements": [],
  "warnings": ["Minor GSD difference between acquisitions (0.65m vs 0.70m)"],
  "explanation": "Input verified: 2 optical scenes with 98.4% spatial overlap."
}
```
