# System Requirements

This document details the functional and non-functional requirements for SatQuery AI (ISRO / SAC · SIH 2026 · PS #26167).

## Functional Requirements

### 1. Input Processing & Data Sources
- **FR-1.1 GeoTIFF Ingestion**: Ingest single-band, multiband, and composite GeoTIFF (`.tif`, `.tiff`) imagery. Extract geospatial metadata (Coordinate Reference System, spatial bounds, pixel dimensions, ground sample distance / resolution).
- **FR-1.2 Standard Format Support**: Accept standard raster formats (`.png`, `.jpg`, `.jpeg`) for rapid testing and demonstrations.
- **FR-1.3 Multimodal Ingestion**: Support independent or paired ingestion of:
  - Single optical satellite scenes.
  - Bi-temporal optical or SAR image pairs for change detection.
  - Paired Optical + SAR scenes covering overlapping geographic extents.
- **FR-1.4 Curated Satellite Data Repository**: Allow querying pre-indexed satellite data archives when user imagery is not directly uploaded.

### 2. Natural-Language Query Understanding
- **FR-2.1 Structured Intent Parsing**: Extract and classify the following entities from unstructured natural-language prompts:
  - **Task Type**: `SINGLE_IMAGE_ANALYSIS` (VQA, Captioning, Grounding), `CHANGE_DETECTION`, `OPTICAL_SAR_FUSION`, `IMPACT_ANALYSIS`, `SCENARIO_ANALYSIS`.
  - **Target Feature**: `building`, `road`, `agriculture`, `forest`, `water`, `vegetation`, `infrastructure`, `land_cover`.
  - **Temporal Scope**: Start date, end date, baseline date, comparison window.
  - **Spatial Scope**: Scene-wide, bounded bounding box, named region of interest (ROI).
  - **Required Modality**: `optical`, `sar`, or `multimodal_fused`.
  - **Requested Outputs**: Numerical metrics, binary masks, bounding boxes, GeoJSON vectors, textual summaries.

### 3. Query-Data Compatibility Gate
- **FR-3.1 Pre-Flight Validation**: Prior to initiating model inference, verify:
  - Adequate image count matching the requested task (e.g., exactly 2 images for change detection).
  - Sensor modality compatibility (e.g., optical + SAR for fusion queries).
  - Temporal metadata availability for multi-date queries.
  - Geographic bounding box overlap (>80% overlap required for paired analysis).
  - Resolution and Ground Sample Distance (GSD) compatibility (flag mismatches exceeding 10x).
  - CRS alignment (reproject dynamically if transforms are defined).
- **FR-3.2 Explanatory Failure Reporting**: If compatibility fails, return clear, user-friendly explanations indicating the missing or invalid requirement rather than failing during model execution.

### 4. Specialist Vision-Language & Remote-Sensing Capabilities
- **FR-4.1 Single-Image VQA (T1_VQA)**: Multi-label classification and natural-language QA over high-resolution optical imagery.
- **FR-4.2 Structured Captioning (T2_Caption)**: Concise thematic descriptions covering terrain, land use, density, and spatial distribution.
- **FR-4.3 Text-Guided Grounding (T3_Ground)**: Localization of described entities returning pixel bounding boxes and geographic extents.
- **FR-4.4 Bi-temporal Change Detection (T4_Change)**: Difference extraction producing binary change masks, change severity indices, and cluster counts.
- **FR-4.5 Optical + SAR Cross-Modal Fusion (T5_OpticalSAR)**: Co-registered multi-modal feature fusion for land-cover classification and texture-sensitive terrain analysis.
- **FR-4.6 General Fallback**: Zero-shot retrieval and visual similarity fallback (RemoteCLIP) when query ambiguity is high.

### 5. Deterministic Geospatial & GIS Analysis
- **FR-5.1 Vectorization**: Convert raster probability and binary masks to simplified GeoJSON polygon FeatureCollections.
- **FR-5.2 Quantitative Spatial Calculations**:
  - Total polygon area in square meters and hectares.
  - Feature counts and spatial density per square kilometer.
  - Nearest-neighbor proximity and spatial clustering.
  - Percentage change relative to total scene area.
- **FR-5.3 Seasonal Filtering**: Post-process change detection results against seasonal baseline indices (e.g., monsoon greening adjustments for South Asia).

### 6. Evidence, Confidence & Explainability
- **FR-6.1 Evidence Items (`EvidenceItem`)**: Associate every quantitative finding with an explicit evidence record (tool ID, geographic coordinates, label, coverage percentage, bounding box).
- **FR-6.2 Multi-Level Confidence Scoring**: Compute calibrated confidence based on model prediction certainty, GIS consistency, and sensor alignment.
- **FR-6.3 Execution Trace**: Return an auditable step-by-step trace (`trace_id`, node name, execution duration in milliseconds, input parameters, status).
- **FR-6.4 Factually Grounded Synthesis**: Synthesize natural-language explanations strictly from verified `EvidenceItem` records (preventing hallucinated coordinates or measurements).

### 7. User Interface (React + Leaflet)
- **FR-7.1 Ingestion Controls**: Drag-and-drop file upload, modality selector, and satellite catalog picker.
- **FR-7.2 Geospatial Map Canvas**: Interactive Leaflet map displaying basemaps, raster overlays, and interactive GeoJSON vector layers.
- **FR-7.3 Response Panel**: Natural-language narrative, confidence indicators, and primary quantitative metrics.
- **FR-7.4 Evidence & Trace Viewers**: Expandable panels for inspecting raw evidence images and step-by-step execution metrics.
- **FR-7.5 Export Options**: Download analysis reports (PDF/JSON) and GeoJSON vector files.
- **FR-7.6 Session Refinement**: Support iterative follow-up queries referencing previous session context.

---

## Non-Functional Requirements

### 1. Performance & Hardware Constraints
- **NFR-1.1 CPU/NPU Deployment Target**: All runtime inference must run efficiently on local CPU / Intel NPU hardware without mandatory GPU dependencies.
- **NFR-1.2 Latency Budget**: Heuristic baseline <250ms; quantized ONNX/OpenVINO models target <1.5s on target hardware (Intel Core Ultra 5).
- **NFR-1.3 Quantization Standard**: Export specialist models to ONNX (opset 17) and apply OpenVINO INT8 quantization for edge inference.

### 2. System Reliability & Security
- **NFR-2.1 Isolated Model Execution**: Failures in individual specialist models must be trapped gracefully without crashing the master orchestration server.
- **NFR-2.2 Zero Credential Leaks**: Strict protection of environment variables and API keys.
- **NFR-2.3 Memory Footprint**: Maximum runtime memory footprint must remain under 4GB RAM during multi-image analysis.

### 3. Maintainability & Modularity
- **NFR-3.1 Tool Swappability**: Any specialist model must be replaceable via its `BaseTool` adapter without altering the LangGraph graph, API schemas, or frontend components.
- **NFR-3.2 Determinism**: Geospatial GIS calculations must yield identical results across multiple executions given the same input raster.
