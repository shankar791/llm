# System Architecture

This document defines the high-level technical architecture, component boundaries, and data flows for SatQuery AI.

## Architectural Diagram

```
+-------------------------------------------------------------------------------+
|                                  USER LAYER                                   |
|   - Browser / Web Client (React + TypeScript + Tailwind CSS)                  |
|   - Interactive Geospatial Mapping Canvas (Leaflet / react-leaflet)           |
+---------------------------------------+---------------------------------------+
                                        | HTTP / Multipart REST
                                        v
+---------------------------------------+---------------------------------------+
|                               API & DATA LAYER                                |
|   - FastAPI Web Server (backend/server.py)                                    |
|   - Rasterio Metadata Extraction & CRS Parsing (backend/rasterio_utils.py)    |
|   - Request Validation (schemas/models.py::QueryRequest)                      |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+---------------------------------------+---------------------------------------+
|                     MASTER AGENT / AI ORCHESTRATOR                            |
|             (LangGraph State Machine: ai/graph/state.py)                      |
|                                                                               |
|   [validate_inputs_node]                                                      |
|          |                                                                    |
|   [classify_intent_node]  --> (Rule-based or Function-Calling LLM)            |
|          |                                                                    |
|   [compatibility_check_node]                                                  |
|          |                                                                    |
|   [route_tools_node]                                                          |
|          |                                                                    |
|          +--------------------+--------------------+                          |
|          |                    |                    |                          |
|          v                    v                    v                          |
|   +--------------+     +--------------+     +--------------+                  |
|   |  T1 / T2 / T3|     |      T4      |     |      T5      |                  |
|   | (GeoChat)    |     |(ChangeFormer)|     |  (EarthGPT)  |                  |
|   +-------+------+     +------+-------+     +------+-------+                  |
|           |                   |                    |                          |
|           +-------------------+--------------------+                          |
|                               |                                               |
|                               v                                               |
|   [gis_processor_node] (Deterministic Rasterio / Shapely / GeoPandas)         |
|          |                                                                    |
|   [evidence_confidence_node]                                                  |
|          |                                                                    |
|   [llm_synthesis_node] (Grounded Natural-Language Answer Synthesis)           |
+---------------------------------------+---------------------------------------+
                                        | AgentResponse JSON (with GeoJSON)
                                        v
+---------------------------------------+---------------------------------------+
|                              PRESENTATION                             |
|   - Interactive GeoJSON polygon overlays on Leaflet Map                       |
|   - Synthesized natural-language answers + confidence metrics                 |
|   - Multi-step execution audit trace with millisecond timings                 |
+-------------------------------------------------------------------------------+
```

## System Components

### 1. Frontend (React + TypeScript + Leaflet)
- **Role**: Presentation and interaction layer.
- **Responsibilities**:
  - Accept user file uploads (GeoTIFF, PNG, JPG) and natural language queries.
  - Render base satellite imagery tiles and vector GeoJSON FeatureCollections.
  - Display synthesized answers, evidence badges, confidence bars, and execution traces.
  - Support follow-up query submission referencing active session IDs.
- **Key Principle**: The frontend consumes *only* standardized `AgentResponse` structures and never parses raw model logits or model-specific formats.

### 2. Backend API Layer (FastAPI)
- **Role**: Ingestion, validation, file management, and API gateway.
- **Responsibilities**:
  - Receive multipart form requests (`POST /api/query`).
  - Read raster bytes, determine MIME types, and extract geospatial headers (CRS, spatial bounds, transform matrix, resolution) via `rasterio`.
  - Package inputs into `AgentState` and invoke the Master Agent.
  - Return formatted `AgentResponse` JSON.
- **Key Principle**: The backend API does *not* make model routing or AI reasoning decisions; it acts as a reliable gateway.

### 3. Master Agent / Orchestrator (LangGraph)
- **Role**: Central decision-making and workflow execution engine.
- **Framework**: LangGraph (`ai/graph/state.py` defines `AgentState`).
- **Core Principle**: **There is exactly ONE Master Agent.** Specialist models are tools invoked by the Master Agent.
- **State Machine Nodes**:
  1. `validate_inputs_node`: Detects image count, modality tags, and data validity.
  2. `classify_intent_node`: Parses query into structured intent (task, target, spatial/temporal scope, modality).
  3. `compatibility_check_node`: Evaluates whether uploaded imagery satisfies task requirements. If incompatible, branches directly to synthesis with an explanatory error.
  4. `route_tools_node`: Dynamically selects tool sequence based on intent and input scenario.
  5. `tool_execution_nodes`: Invokes specialist tool instances (`tools/base.py::BaseTool`).
  6. `gis_processor_node`: Executes deterministic spatial computations.
  7. `evidence_confidence_node`: Aggregates multi-source confidence metrics and creates `EvidenceItem` records.
  8. `llm_synthesis_node`: Translates verified evidence into a coherent natural language narrative.

### 4. Specialist Tools & Isolated Adapters
Specialist models are wrapped inside standard tool classes deriving from `tools/base.py::BaseTool`:
- **`VQATool` (`T1_VQA`)**: Single-image visual question answering via GeoChat adapter.
- **`CaptionTool` (`T2_Caption`)**: Structured thematic scene description via GeoChat adapter.
- **`GroundingTool` (`T3_Ground`)**: Region localization and bounding box extraction via GeoChat adapter.
- **`ChangeDetectionTool` (`T4_Change`)**: Bi-temporal difference mapping via ChangeFormer adapter.
- **`OpticalSARTool` (`T5_OpticalSAR`)**: Optical + SAR cross-modal fusion via EarthGPT adapter.
- **`FallbackTool` (`Fallback`)**: Zero-shot similarity search via RemoteCLIP adapter.

**Adapter Isolation**:
External model libraries reside under `models/<model_name>/` and are instantiated strictly inside tool adapter classes. No external model code is imported directly into orchestrator or API layers.

### 5. Deterministic GIS Engine
- **Role**: Precise, reproducible geospatial calculations.
- **Rule**: **LLMs MUST NEVER perform deterministic GIS calculations.**
- **Key Operations**:
  - Mask thresholding & polygonization (via `rasterio.features.shapes` and `shapely`).
  - Coordinate reprojection (via `pyproj` / `rasterio.warp`).
  - Quantitative metrics: exact area in $m^2$ and hectares, polygon counts, spatial density, overlap fractions.
  - Post-processing seasonal filters (e.g., monsoon greening baseline comparison).

### 6. Evidence Engine & Grounded Synthesis
- **Evidence Contract**: Every factual claim produced by the system is linked to an `EvidenceItem` with geometric boundaries, sensor metadata, tool provenance, and confidence values.
- **Grounded LLM Synthesis**: The synthesis LLM is strictly constrained to explain and contextualize verified `EvidenceItem` data, eliminating hallucinated statistics, dates, or coordinates.
