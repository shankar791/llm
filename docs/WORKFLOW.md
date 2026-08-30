# System Workflow & Execution Pipeline

This document details the step-by-step end-to-end execution workflow for SatQuery AI, including decision branching and detailed execution traces.

## End-to-End Master Pipeline Sequence

```
1. USER INTERACTION
   Upload 1-2 GeoTIFF/image files + enter natural-language query
   ↓
2. INPUT VALIDATION (validate_inputs_node)
   Inspect format, byte integrity, dimensions, band counts, and sensor modality tags
   ↓
3. GEO-METADATA EXTRACTION
   Extract CRS, Affine transform matrix, geographic bounding box, spatial resolution
   ↓
4. INTENT UNDERSTANDING (classify_intent_node)
   LLM Intent Classifier (or Rule fallback) parses query into:
   - task: ('vqa' | 'caption' | 'ground' | 'change' | 'fusion')
   - target, temporal scope, spatial scope, requires_pair, ambiguity
   - Strict Pydantic validation on structured JSON
   - NO tool IDs output by LLM
   ↓
5. QUERY-DATA COMPATIBILITY CHECK (compatibility_check_node)
   Deterministic verification of actual data against requirements:
   - Image count (e.g. change requires ≥ 2 images)
   - Modalities (e.g. fusion requires optical + SAR)
   - Spatial overlap & CRS compatibility
   ↓
   [Branch: If Incompatible → Generate explanation & exit to synthesis]
   ↓
6. MASTER AGENT ROUTING (master_router_node)
   Authoritative assignment of specialist tool IDs (T1..T5) from ToolRegistry
   ↓
7. SPECIALIST MODEL INFERENCE
   Execute selected tool adapter (GeoChat, ChangeFormer, EarthGPT, or RemoteCLIP)
   ↓
8. RESULT INTEGRATION & STANDARDIZATION
   Convert raw tensor/mask outputs into standard ToolResult structure
   ↓
9. DETERMINISTIC GIS PROCESSING (gis_processor_node)
   Polygonize binary masks, compute area (m²/ha), apply seasonal false-positive filter
   ↓
10. EVIDENCE & CONFIDENCE MAPPING (evidence_confidence_node)
    Construct EvidenceItems linking geometries, timestamps, metrics, and confidence
   ↓
11. IMPACT / TREND ANALYSIS
    Categorize change magnitude, direction (expansion/reduction), and spatial density
   ↓
12. SCENARIO ANALYSIS
    Evaluate baseline, sustainable, and high-impact trajectories with explicit assumptions
   ↓
13. EVIDENCE-GROUNDED SYNTHESIS (llm_synthesis_node)
    - LLMSynthesizer converts ToolResults, EvidenceItems, and GIS metrics into grounded response
    - Claims mapped to valid evidence IDs (e.g. E1, E2)
    - Post-validator verifies numeric consistency against GIS measurements
    - Automatic switch to DeterministicFallbackFormatter if LLM is unavailable or hallucinates
    ↓
14. FRONTEND PRESENTATION & VISUALIZATION
    Render GeoJSON layers on Leaflet map, display answer, evidence badges, and execution trace
```

---

## Detailed Task Workflows

### 1. Bi-Temporal Change Detection Workflow

**Example Query**: *"Identify new construction between 2020 and 2024."*

```
[User Input]: 2 GeoTIFFs (t0_2020.tif, t1_2024.tif) + Query string
  ↓
[Input Validation]: Detects 2 optical rasters; extracts CRS=EPSG:32643, bounds, GSD=0.65m
  ↓
[Intent Classification]:
  - Task: CHANGE_DETECTION
  - Target: BUILDING
  - Temporal Scope: 2020 → 2024
  - Required Modality: Optical
  ↓
[Compatibility Check]:
  - Check 1: Image count == 2? (Pass)
  - Check 2: Modalities match? (Pass)
  - Check 3: Spatial overlap > 80%? (Pass, overlap = 98.4%)
  - Check 4: CRS compatible or reprojectable? (Pass)
  ↓
[Master Agent Routing]:
  - Invokes: ChangeDetectionTool (T4_Change)
  ↓
[Model Execution]:
  - ChangeFormerAdapter.detect(image_t0, image_t1)
  - Output: Binary change mask (numpy 2D array, 1024x1024)
  ↓
[Deterministic GIS Engine]:
  - Vectorizes mask to Shapely MultiPolygons
  - Filters small noise components (< 50 m²)
  - Applies seasonal monsoon filter (no false greening detected)
  - Computes total change area = 142,500 m² (14.25 ha) across 14 discrete clusters
  - Converts geometry to GeoJSON FeatureCollection
  ↓
[Evidence & Confidence]:
  - Creates EvidenceItem (tool_id="T4_Change", label="new_construction", coverage_pct=7.2, confidence=0.88)
  - Generates base64 overlay image highlighting change polygons
  ↓
[Language Synthesis]:
  - Synthesizes: "Analysis between 2020 and 2024 indicates 14.25 hectares (7.2% of the surveyed area) of new construction activity across 14 distinct clusters, primarily in the northeast sector."
  ↓
[Leaflet Display]:
  - Highlights GeoJSON polygons in red/yellow with interactive hover tooltips showing cluster area.
```

---

### 2. Single-Image VQA & Grounding Workflow

**Example Query**: *"Where are the industrial storage tanks in this port image?"*

```
[User Input]: 1 GeoTIFF (port_optical.tif) + Query string
  ↓
[Intent Classification]:
  - Task: GROUNDING (T3_Ground) + VQA (T1_VQA)
  - Target: INDUSTRIAL_STORAGE_TANKS
  - Modality: Optical
  ↓
[Compatibility Check]:
  - Single optical image available? (Pass)
  ↓
[Master Agent Routing]:
  - Invokes: GroundingTool (T3_Ground) via GeoChatAdapter
  ↓
[Model Execution]:
  - GeoChatAdapter.ground(image, "industrial storage tanks")
  - Output: List of normalized pixel bounding boxes [[ymin, xmin, ymax, xmax], ...]
  ↓
[Deterministic GIS Engine]:
  - Maps pixel coordinates to geographic coordinates using raster Affine transform
  - Converts bounding boxes to GeoJSON Polygons with EPSG:4326 coordinates
  - Calculates count = 8 tanks, total footprint = 18,200 m²
  ↓
[Evidence & Synthesis]:
  - Generates EvidenceItems for each localized tank
  - Final answer: "Detected 8 industrial storage tanks clustered in the southern dockyard with a combined footprint of ~1.82 hectares."
```

---

### 3. Optical + SAR Multimodal Fusion Workflow

**Example Query**: *"Use optical and SAR data to classify flood extent and waterlogged soil."*

```
[User Input]: 1 Optical GeoTIFF + 1 SAR GeoTIFF (e.g. RISAT/Sentinel-1)
  ↓
[Intent Classification]:
  - Task: OPTICAL_SAR_FUSION (T5_OpticalSAR)
  - Target: WATER_SOIL_MOISTURE
  - Modality: Multimodal (Optical + SAR)
  ↓
[Compatibility Check]:
  - 1 Optical and 1 SAR scene present? (Pass)
  - Spatial bounding boxes align? (Pass)
  ↓
[Master Agent Routing]:
  - Invokes: OpticalSARTool (T5_OpticalSAR) via EarthGPTAdapter
  ↓
[Model Execution]:
  - EarthGPTAdapter.fuse(optical_raster, sar_raster)
  - Fuses optical spectral bands with SAR backscatter intensity/texture
  ↓
[Deterministic GIS Engine]:
  - Differentiates standing open water (low SAR backscatter + low NDVI) from waterlogged vegetation (high SAR cross-section + high NDVI)
  - Computes standing water area = 45.3 ha; waterlogged soil area = 28.1 ha
  ↓
[Evidence & Synthesis]:
  - Synthesizes multimodal findings citing complementary optical spectral and SAR dielectric evidence.
```

---

### 4. Incompatibility & Fallback Routing

```
[Query]: "Detect deforestation between 2021 and 2024."
[Input]: Only 1 single 2024 image uploaded.
  ↓
[Compatibility Check]:
  - Required image count: 2
  - Provided image count: 1
  - Outcome: FAIL
  ↓
[Early Exit Branch]:
  - Bypasses model invocation
  - Formulates explanation: "Change detection requires imagery from at least two distinct timestamps. Please upload a baseline image from 2021 alongside the 2024 scene."
  - Returns AgentResponse with empty outputs, confidence = 0.0, and error log in execution trace.
```
