# Implementation Tasks & Phase-Wise Roadmap

This document outlines the phased development plan for SatQuery AI.

### Status Indicators
- `[ ]` Not started
- `[~]` In progress / Partial implementation
- `[x]` Completed and verified by automated tests

> **RULE**: Do not mark any task `[x]` unless the component is implemented, integrated, and verified with functional unit or E2E tests.

---

## PHASE 0 — Documentation, Governance & Foundation
- [x] Project overview and conceptual architecture documentation (`README.md`)
- [x] AI agent instruction guidelines and source-of-truth priority (`AGENTS.md`)
- [x] Functional and non-functional requirements (`docs/REQUIREMENTS.md`)
- [x] Technical architecture and component boundary definitions (`docs/ARCHITECTURE.md`)
- [x] End-to-end execution workflow and branch specification (`docs/WORKFLOW.md`)
- [x] Phase-wise implementation task registry (`docs/TASKS.md`)
- [x] Technology stack and dependency constraints (`docs/TECH_STACK.md`)
- [x] Proposed and active API contracts (`docs/API.md`)
- [x] ML pipeline and external model adapter specification (`docs/ML_PIPELINE.md`)
- [x] Third-party legal, license, and attribution register (`docs/THIRD_PARTY.md`)
- [x] Architecture Decision Records (`ARCHITECTURE_DECISIONS.md`)
- [x] Transitional heuristic backend prototype with E2E verification (`backend/test_e2e.py`)

---

## PHASE 1 — Data Contracts & Pydantic Schemas
- [x] Core contracts: `QueryRequest`, `EvidenceItem`, `ToolResult`, `AgentResponse` (`schemas/models.py`)
- [x] LangGraph `AgentState` TypedDict definition (`ai/graph/state.py`)
- [ ] Implement `IntentSchema` (structured task, target, temporal, spatial, modality entities)
- [ ] Implement `CompatibilityResult` schema (boolean status, missing attributes, warning logs)
- [ ] Implement `ToolRequest` schema (standardized tool execution parameters)
- [ ] Implement `ExecutionTrace` schema (step sequence, node name, duration_ms, status)
- [ ] Implement `SessionState` schema for multi-turn conversational memory
- [ ] Automated unit test suite verifying schema validation and JSON serialization

---

## PHASE 2 — Frontend Skeleton (React + Leaflet)
- [ ] React 18 + TypeScript application scaffold with Vite
- [ ] Drag-and-drop GeoTIFF / multi-image upload component
- [ ] Natural-language query input box and execution triggers
- [ ] Interactive Leaflet map canvas with base satellite tile layers
- [ ] GeoJSON vector layer rendering with dynamic polygon styling
- [ ] Answer sidebar displaying synthesized narrative, confidence bars, and key metrics
- [ ] Collapsible execution trace viewer with per-node execution timings
- [ ] Mock data fixtures for independent frontend visual testing

---

## PHASE 3 — LangGraph Orchestration Skeleton
- [ ] LangGraph graph builder in `ai/graph/graph.py`
- [ ] Implement `validate_inputs_node` (extract raster metadata, detect modality tags)
- [ ] Implement `classify_intent_node` (rule-based heuristic with LLM fallback)
- [ ] Implement `compatibility_check_node` (spatial overlap, date checks, resolution checks)
- [ ] Implement `route_tools_node` (dynamic routing to specialist tools)
- [ ] Implement `mock_tool_node` providing deterministic test fixtures for all tasks
- [ ] Implement `gis_processor_node` skeleton
- [ ] Implement `llm_synthesis_node` skeleton
- [ ] Integration test: Full LangGraph cycle runs start-to-finish across all scenarios using mock tools

---

## PHASE 4 — Bi-Temporal Change Detection (ChangeFormer)
- [ ] Verify source code and model weight licenses for `Chen-Zhiang/ChangeFormer`
- [ ] Isolate minimal ChangeFormer network definition under `models/changeformer/`
- [ ] Implement `ChangeFormerAdapter` (`load`, `preprocess`, `detect` returning binary change mask)
- [ ] Wire `ChangeDetectionTool` (`tools/change_detection.py`) to `ChangeFormerAdapter`
- [ ] Integration test with Delta-SN6 sample bi-temporal image pair
- [ ] Export ChangeFormer to ONNX (opset 17) with dynamic spatial axes
- [ ] Benchmark CPU inference latency on Intel Core Ultra 5

---

## PHASE 5 — Deterministic GIS Processing Engine
- [ ] Create `gis/processor.py` geospatial processing module
- [ ] Binary raster mask vectorization to Shapely MultiPolygons via `rasterio.features.shapes`
- [ ] Polygon coordinate reprojection from pixel space to geographic CRS (EPSG:4326)
- [ ] Deterministic area calculation in square meters and hectares
- [ ] Change fraction computation relative to valid raster extent
- [ ] Connected-component spatial clustering and feature count extraction
- [ ] Implement seasonal false-positive filter (`seasonal_filter`) using acquisition date and vegetation index
- [ ] Comprehensive unit test suite for GIS calculations with known ground-truth geometries

---

## PHASE 6 — End-to-End Change Detection Verification
- [ ] Connect Frontend → FastAPI API → LangGraph Orchestrator → ChangeFormer Tool → GIS Engine → GeoJSON → Leaflet UI
- [ ] Verify complete user flow on *"Identify new construction between 2020 and 2024"*
- [ ] Validate four-step execution trace generation with accurate duration logging
- [ ] Verify interactive GeoJSON polygon hover tooltips on the Leaflet map

---

## PHASE 7 — Single-Image Specialist Capabilities (GeoChat)
- [ ] Verify license terms for `mbzuai-oryx/GeoChat` (code and weights)
- [ ] Isolate minimal GeoChat vision-language architecture under `models/geochat/`
- [ ] Implement `GeoChatAdapter` supporting `vqa()`, `caption()`, and `ground()` methods
- [ ] Wire `VQATool` (`tools/vqa.py`) to `GeoChatAdapter.vqa()`
- [ ] Wire `CaptionTool` (`tools/captioning.py`) to `GeoChatAdapter.caption()`
- [ ] Wire `GroundingTool` (`tools/grounding.py`) to `GeoChatAdapter.ground()`
- [ ] Export vision encoder and projection layers to ONNX
- [ ] Integration tests verifying VQA, Captioning, and Grounding outputs

---

## PHASE 8 — Optical + SAR Cross-Modal Fusion (EarthGPT)
- [ ] Verify license terms for `wivizhang/EarthGPT` and MMRS-1M dataset
- [ ] Isolate dual-encoder architecture under `models/earthgpt/`
- [ ] Implement `EarthGPTAdapter.fuse(optical_bytes, sar_bytes)`
- [ ] Wire `OpticalSARTool` (`tools/optical_sar.py`) to `EarthGPTAdapter.fuse()`
- [ ] Integration test verifying fused classification on paired optical and SAR scenes
- [ ] Export fusion network to ONNX

---

## PHASE 9 — Zero-Shot Vision Fallback (RemoteCLIP)
- [ ] Verify license terms for `RemoteCLIP/RemoteCLIP`
- [ ] Isolate ViT-B-32 vision-language backbone under `models/remoteclip/`
- [ ] Implement `RemoteCLIPAdapter` for image-text similarity scoring
- [ ] Wire `FallbackTool` (`tools/fallback.py`) to activate on low intent confidence
- [ ] Export RemoteCLIP visual encoder to ONNX

---

## PHASE 10 — Result Standardization & Validation
- [ ] Enforce strict compliance: All specialist tools return valid `ToolResult` dictionaries
- [ ] Add runtime validation asserting no tool-specific raw logits escape to API layer
- [ ] Regression test: All 5 specialist tools + fallback produce standard results

---

## PHASE 11 — Evidence Engine & Confidence Calibration
- [ ] Ensure full population of `EvidenceItem` records across all tools
- [ ] Calibrate multi-factor confidence scoring (model probability * GIS spatial coherence)
- [ ] Ensure every claim in the synthesized answer references a valid `EvidenceItem`
- [ ] Audit log persistence to `/evidence` directory

---

## PHASE 12 — Conversational Memory & Refinement
- [ ] Implement session state manager with configurable TTL
- [ ] Support refinement queries (e.g., *"Filter to changes larger than 500 m²"*) without re-running deep learning inference
- [ ] Track query history and active geospatial bounds per session

---

## PHASE 13 — Impact & Trend Analysis
- [ ] Implement `impact_analysis.py` categorizing change severity (Minor, Moderate, Major)
- [ ] Compute spatial distribution metrics (clustered vs. dispersed)
- [ ] Empirical trend trajectory computation (requires $\ge 3$ temporal scenes)

---

## PHASE 14 — Scenario Analysis Module
- [ ] Implement `scenario_analysis.py` exploring baseline, sustainable, and high-impact trajectories
- [ ] Explicit labeling of mathematical assumptions and confidence bounds
- [ ] Structured scenario output schema feeding into synthesis LLM

---

## PHASE 15 — Hardware Optimization (CPU/NPU)
- [ ] Finalize ONNX opset 17 exports for all specialist models
- [ ] Apply OpenVINO INT8 post-training quantization using representative calibration datasets
- [ ] Benchmark per-model latency and memory footprint on Intel Core Ultra 5 hardware
- [ ] Verify memory usage remains $< 4\text{GB}$ under concurrent load

---

## PHASE 16 — Final Integration, Packaging & Submission
- [ ] Production Dockerfile and multi-container Docker Compose configuration
- [ ] Evaluation on ISRO Cartosat-2S and RISAT sensor benchmarks (if accessible)
- [ ] Complete test suite execution (Unit, Integration, E2E, Load)
- [ ] Clean repository state (zero secrets, zero untracked binaries, updated documentation)
- [ ] Submission deliverable package preparation


