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

## Model Separation & Architectural Responsibilities

SatQuery AI strictly enforces functional separation across specialized models and deterministic engines:

| Subsystem / Model | Implementation Location | Concrete Engine / Model | Architectural Responsibility |
|---|---|---|---|
| **Text NLP / LLM** | `ai/llm/`, `ai/intent/`, `ai/synthesis/` | **NVIDIA Nemotron 3 Ultra 550B** / **Qwen3-14B** (OpenRouter) | Intent classification, query understanding, reasoning assistance, grounded synthesis, and response refinement. |
| **Vision Language Models** | `ai/vision/` | **Gemma 4 26B**, **Gemma 4 31B**, **Nemotron 3 Nano Omni** (OpenRouter) / **GeoChat** | High-level image understanding: single-image VQA (`T1`), Captioning (`T2`), Grounding / Box localization (`T3`) with multi-model fallback. |
| **Bi-temporal Specialist** | `models/changeformer/` | **ChangeFormer** (Siamese ViT) | Bi-temporal pixel-level change detection & binary change masking (`T4`). |
| **Cross-Modal Specialist** | `models/earthgpt/` | **EarthGPT** | Optical + SAR radar feature fusion (`T5`). |
| **Zero-Shot Fallback** | `models/remoteclip/` | **RemoteCLIP** | Visual embedding similarity fallback when confidence < threshold. |
| **Deterministic GIS Engine** | `gis/` | **Python / Rasterio / Shapely / GeoPandas** | Authoritative surface area ($m^2$, ha), polygon geometries, CRS transforms. |
| **Evidence & Validation Engine** | `ai/synthesis/validator.py` | **SynthesisValidator** | Strict anti-hallucination post-validation; ensures zero altered GIS facts. |

## Provider-Agnostic LLM Foundation (`ai/llm/`)

The Master Agent and its supporting nodes interact with language models through a strictly decoupled, vendor-agnostic Protocol interface (`LLMProvider` in `ai/llm/base.py`).

### Key Capabilities:
1. **Vendor Agnostic**: Compatible with any OpenAI-style Chat Completions endpoint (OpenAI, OpenRouter, Groq, Together, Ollama, vLLM).
2. **Structured Output**: Native support for `response_format={"type": "json_object"}` with `LLMResponse.json()` helper parsing.
3. **Transient Fault Tolerance**: Automatic exponential backoff with jitter on HTTP 429, 500, 502, 503, 504 and network timeouts. Immediate fail-fast on HTTP 401/403/400.
4. **Offline Determinism**: `MockLLMProvider` enables 100% offline, deterministic CI/CD and unit testing without external API tokens.
5. **Observability**: Exposes per-request `latency_ms`, token `usage`, and sanitized diagnostics without leaking secrets or full prompts.

## Intent Classification & Routing Boundary

SatQuery AI enforces strict architectural separation between natural language interpretation and specialist execution:

```
User Query + Metadata
        ↓
[LLMIntentClassifier] (Pydantic validation, structured JSON, task enum)
        ↓
  IntentResult (task, target, temporal_scope, requires_pair, ambiguous)
        ↓
[Compatibility Gate] (Deterministic validation of actual image count, modality, CRS)
        ↓
[Master Router] (Authoritative tool assignment from ToolRegistry: T1..T5)
```

- **LLM Responsibility**: Semantic interpretation, task classification (`vqa`, `caption`, `ground`, `change`, `fusion`), target concepts, temporal boundaries, and ambiguity flags. The LLM **NEVER** dictates authoritative tool IDs or execution workflows.
- **Compatibility Gate**: Deterministic Python rules verifying whether uploaded rasters satisfy task requirements.
- **Master Router**: Authoritative tool dispatcher that binds validated requirements to specialist models.

## Evidence-Grounded LLM Synthesis (`ai/synthesis/`)

Final response generation uses `LLMSynthesizer` (`ai/synthesis/llm.py`) backed by `ai.llm.LLMProvider`.

### Core Grounding & Anti-Hallucination Rules:
1. **Factual Grounding**: The LLM is supplied only structured `ToolResult`s, `EvidenceItem`s, and authoritative GIS metrics (e.g. `area_ha`, `polygon_count`, `change_fraction`). Raw images and full graph states are never passed.
2. **Structured Output & Claims**: Emits `SynthesisPayload` with explicit `claims` mapping text statements to valid `evidence_ids` (e.g. `["E1", "E2"]`).
3. **Deterministic Post-Validation**: `SynthesisValidator` performs strict post-generation verification:
   - Validates that every referenced evidence ID exists in the supplied context (rejection of fake/invented IDs).
   - Validates numeric quantities (hectares, polygon counts, percentages, dates) against authoritative GIS measurements.
   - Forbids fabricated calibrated confidence claims (e.g. "98% confident") when model confidence is uncalibrated.
4. **Deterministic Fallback**: If LLM generation fails or the post-validator detects a hallucination, the system immediately switches to `DeterministicFallbackFormatter` (`synthesis_source="deterministic_fallback"`), ensuring high availability and 100% factual accuracy.

## Resilient Multi-Model Multimodal Vision Provider Subsystem (`ai/vision/`)

SatQuery AI decouples high-level vision tools (`T1_VQA`, `T2_Caption`, `T3_Ground`) from concrete model implementations via the `VisionProvider` Protocol interface with multi-model resilience and fallback routing.

### Configured Vision Models:
1. **Primary Vision Model**: `google/gemma-4-26b-a4b-it:free` (`VISION_PRIMARY_MODEL`)
2. **Secondary Vision Model**: `google/gemma-4-31b-it:free` (`VISION_SECONDARY_MODEL`)
3. **Tertiary Vision Model**: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` (`VISION_TERTIARY_MODEL`)

### Task-Level Routing Policies:
- **`T1_VQA`**: Primary: Gemma 4 26B -> Fallbacks: Gemma 4 31B -> Nemotron 3 Nano Omni.
- **`T2_Caption`**: Primary: Gemma 4 26B -> Fallbacks: Gemma 4 31B -> Nemotron 3 Nano Omni.
- **`T3_Ground`**: Primary: Gemma 4 31B -> Fallbacks: Gemma 4 26B -> Nemotron 3 Nano Omni.

### Fallback Logic & Rate-Limit Handling:
- **Transient Failures (Fallback Triggered)**: Upstream 429 concurrency pool limits, 5xx server errors, socket timeouts, and provider connection drops automatically trigger fallback to the next candidate model in the chain.
- **Account-Level Rate Limits (Immediate Fail-Fast)**: If OpenRouter returns `free-models-per-day` or account quota exhaustion, the system classifies it as `ACCOUNT_RATE_LIMIT` and immediately halts to avoid wasting API calls across fallback models.
- **Non-Transient Errors (No Fallback)**: Malformed application inputs (400) or missing credentials (401/403) fail immediately without fallback loops.
- **Structured Grounding Validation**: Requires machine-readable bounding boxes with normalized coordinates $0 \le x_0 \le x_1 \le 1, 0 \le y_0 \le y_1 \le 1$. If a candidate returns unstructured natural language, it is marked `grounding_unsupported` and fails over to the next candidate model without fabricating fake boxes.
- **Observability**: `VisionResponse` and tool metadata explicitly record `selected_model`, `attempted_models`, `fallback_used`, `fallback_reason`, and `latency_ms`.


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
