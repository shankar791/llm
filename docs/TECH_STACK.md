# Technology Stack

This document defines the technology stack, library versions, and architectural runtime constraints for SatQuery AI.

## Current Working Environment

| Layer | Technology | Version | Purpose & Location |
|---|---|---|---|
| **OS / Platform** | Microsoft Windows 11 (x64) | 10.0.26200 | Host execution environment |
| **Backend Framework** | FastAPI | 0.129.0 | High-performance async API server (`backend/server.py`) |
| **Python Runtime** | CPython | 3.14.3 | Main backend runtime (`C:/Python314/python.exe`) |
| **Package Manager** | pip / uv | pip 26.1.2 / uv 0.12.5 | Fast dependency resolution and environment management |
| **Raster I/O & Geospatial** | rasterio | >= 1.4.0 | GeoTIFF metadata extraction, CRS reprojection, affine transforms |
| **Node.js Environment** | Node.js / npm | v24.13.1 / 11.17.0 | Frontend build tooling and CLI execution |

---

## Planned Target Stack

### 1. Frontend Layer
- **Framework**: React 18+ (with TypeScript)
- **Bundler**: Vite (fast HMR and optimized builds)
- **Geospatial Mapping**: Leaflet & `react-leaflet` (GeoJSON vector rendering, base satellite tiles, interactive tooltips)
- **Styling**: Tailwind CSS (utility-first responsive UI)
- **Icons & Visuals**: Lucide React

### 2. AI Orchestration, Reasoning & LLM Foundation
- **Orchestration Framework**: LangGraph (state machine managing graph nodes, conditional branches, and tool routing in `ai/graph/`)
- **Data Contracts**: Pydantic v2 (strict request/response validation in `schemas/models.py`)
- **Provider-Agnostic LLM Foundation (`ai/llm/`)**: Vendor-agnostic Protocol supporting OpenAI-compatible chat endpoints (OpenAI, OpenRouter, Groq, Together, Ollama, vLLM) with automatic transient retry, exponential backoff, and structured JSON output
- **Resilient Multimodal Vision Foundation (`ai/vision/`)**: `VisionProvider` Protocol interface supporting multi-model hierarchy and fallbacks across OpenRouter free candidates (`google/gemma-4-26b-a4b-it:free`, `google/gemma-4-31b-it:free`, `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`) with task-level routing and optional `GeoChatAdapter` in `models/geochat/`
- **Intent & Synthesis Engine**: Configurable via `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY` with `MockLLMProvider` for deterministic testing
- **Agent Architecture**: Single Master Agent / Orchestrator invoking specialized tools

### 3. Specialist Machine Learning Models (Isolated Adapters)
- **Deep Learning Framework**: PyTorch (>= 2.2.0)
- **Single-Image VQA / Caption / Ground**: Gemma 4 26B / Gemma 4 31B / Nemotron 3 Nano Omni (`ai/vision/`) + GeoChat (`models/geochat/`)
- **Bi-Temporal Change Detection**: ChangeFormer (`Chen-Zhiang/ChangeFormer`, Siamese ViT)
- **Optical + SAR Fusion**: EarthGPT (`wivizhang/EarthGPT`, multimodal instruction-tuned)
- **Zero-Shot Fallback**: RemoteCLIP (`RemoteCLIP/RemoteCLIP`, ViT-B-32 remote-sensing visual backbone)

### 4. Deterministic Geospatial (GIS) Engine
- **Vector Operations**: Shapely (polygon creation, simplification, intersection, buffering)
- **Geographic DataFrames**: GeoPandas (attribute-rich GeoJSON construction)
- **Coordinate Systems & Projections**: `pyproj` & `rasterio.warp` (dynamic re-projection to EPSG:4326)
- **Raster Feature Extraction**: `rasterio.features.shapes` (mask-to-polygon conversion)

### 5. Hardware Inference & Quantization (CPU/NPU)
- **Intermediate Format**: ONNX (Open Neural Network Exchange, opset 17)
- **CPU Inference Engine**: ONNX Runtime (`onnxruntime`)
- **NPU/CPU Quantization**: OpenVINO Runtime & NNCF (Neural Network Compression Framework for INT8 post-training quantization)
- **Target Edge Hardware**: Intel Core Ultra 5 (Meteor Lake / Arrow Lake NPU)

### 6. Infrastructure & Tooling
- **Containerization**: Docker & Docker Compose
- **Version Control**: Git
- **Codebase Auditing & Token Packing**: Repomix
- **Structural Pattern Search**: `ast-grep` (AST-based code intelligence)

---

## Dependency Ingestion Guidelines

> **IMPORTANT PRINCIPLE**: Do not install all listed dependencies immediately.
> Inspect existing project requirements, add packages only when reaching their specific implementation phase, and test on the Windows / Python 3.14 environment before committing.
