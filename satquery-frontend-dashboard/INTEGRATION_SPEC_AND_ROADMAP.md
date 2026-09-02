# SatQuery AI — System Architecture & Integration Roadmap

## 1. System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     SATQUERY AI ARCHITECTURE                                    │
│                                                                                                  │
│  [ UI Layer: REST API + Interactive Dashboard + 3D Mission Control ]                            │
│                                  │                                                               │
│                                  ▼                                                               │
│  [ FastAPI Backend Engine: Raster Ingestion & S2/SAR Validator ]                                 │
│                                  │                                                               │
│                                  ▼                                                               │
│  [ 9-Node LangGraph Master Agent ]                                                               │
│    1. validate_inputs       ──► 2. classify_intent    ──► 3. compatibility_check                 │
│    4. master_router         ──► 5. execute_specialist ──► 6. standardize_results                 │
│    7. gis_processor (NDVI)  ──► 8. evidence_confidence ──► 9. llm_synthesis (Grounded Output)    │
│                                  │                                                               │
│         ┌────────────────────────┴────────────────────────┬─────────────────────────┐            │
│         ▼                                                 ▼                         ▼            │
│  [ Specialist Tools T1–T5 ]                     [ Vision Providers ]      [ Authoritative GIS ]  │
│  • T1_VQA       • T4_Change                     • GeoChat (RTX 3050)      • RasterIO Multi-band  │
│  • T2_Caption   • T5_OpticalSAR                 • OpenRouter (Qwen-VL)    • GeoJSON WGS84 Engine │
│  • T3_Ground                                    • Mock / Fallback                                │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 2. LangGraph 9-Node Pipeline
1. **`validate_inputs`**: Verifies raster payload, bounding box validity, and query text length.
2. **`classify_intent`**: Determines task category (VQA, Caption, Grounding, Change Detection, Optical/SAR Fusion).
3. **`compatibility_check`**: Verifies sensor compatibility (e.g., Change requires 2 images; SAR requires C-band/L-band data).
4. **`master_router`**: Dispatches request to selected Specialist Tool (T1–T5).
5. **`execute_specialist_tool`**: Calls selected vision provider or model microservice.
6. **`standardize_results`**: Converts model outputs into a unified Pydantic `ToolResult` contract.
7. **`gis_processor`**: Computes deterministic index layers (NDVI, NDWI, NBR, and change masks).
8. **`evidence_confidence`**: Calibrates confidence scores using geospatial evidence matrices.
9. **`llm_synthesis`**: Synthesizes natural-language response grounded in GIS calculations and detected bounding boxes.

## 3. Specialist Tool Tier
- **`T1_VQA`**: Visual Question Answering on optical and infrared bands.
- **`T2_Caption`**: Comprehensive natural language description of satellite scenes.
- **`T3_Ground`**: Spatial grounding returning WGS84 bounding boxes and labels.
- **`T4_Change`**: Bi-temporal difference analysis with change categorization.
- **`T5_OpticalSAR`**: Dual-sensor optical/SAR fusion for all-weather penetration.

## 4. REST API Endpoint Contract
- **Endpoint**: `POST /api/query`
- **Request**:
```json
{
  "query": "What changes occurred in this area between May 2023 and May 2025?",
  "session_id": "session-104",
  "aoi": {
    "type": "Polygon",
    "coordinates": [[[78.47, 17.37], [78.50, 17.37], [78.50, 17.40], [78.47, 17.40], [78.47, 17.37]]]
  },
  "images": [
    { "type": "optical", "timestamp": "2023-05-15", "url": "data:image/jpeg;base64,..." },
    { "type": "optical", "timestamp": "2025-05-20", "url": "data:image/jpeg;base64,..." }
  ]
}
```
