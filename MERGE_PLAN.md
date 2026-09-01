# SatQuery AI — Frontend Consolidation & Merge Plan

This document details the complete read-only architectural audit and integration plan to merge the 3D Animation experience (`shankar791/shankar-fronted`) and the Earth Globe Mission Dashboard (`PrakashMB-1213/-rakash-frontend`) into ONE production frontend for **SatQuery AI**, fully preserving the live backend integration.

---

## 1. FIRST PHASE — READ-ONLY AUDIT

### Summary Comparison Matrix

| Property | Source 1 (`shankar-fronted`) | Source 2 (`prakash-frontend`) | Destination (`satquery-ai`) |
| :--- | :--- | :--- | :--- |
| **A. Framework** | Vanilla JS / WebGL2 (PlayCanvas) + GSAP | PlayCanvas + Leaflet.js v1.9.4 + DOM HUD | FastAPI Static Dashboard / PlayCanvas Engine |
| **B. Build Tool** | None (Static native runtime) | None (Static native runtime) | None (Static native runtime) / Uvicorn |
| **C. React/Next/Vite Version** | None (Native ES6 / WebGL2) | None (Native ES6 / DOM / Leaflet) | None (Native ES6 / Leaflet / Vanilla JS) |
| **D. Dependencies (`package.json`)** | None (No `package.json`) | None (No `package.json`) | None (Python-driven backend & static frontend) |
| **E. Routing System** | Single Route (`/`) | Multi-Route (`/` 3D Intro $\to$ `/app/` Earth Mission) | API Routes (`/api/query`, `/api/samples`) + Static UI |
| **F. Animation Implementation** | PlayCanvas 3D Camera Loop & Odometer Loader | Sine-eased 3D Dive (`diveToChip`, `diveToClouds`) + Fog Transition | CSS Stage Animations & Live Stepper |
| **G. Earth Globe Implementation** | None | Fullscreen Leaflet + Google Satellite Tiles + HUD | 2D Leaflet Polygon / GIS Evidence Visualizer |
| **H. Required Assets** | `files/assets/` (3D `.glb`, `.ogg`, `.basis`, `.mp4`) | `files/assets/` + Leaflet CSS/JS | Real Satellite Images (`backend/real_data`) + 3D Assets |
| **I. Required CSS/Styles** | `styles.css` (reset & canvas) | `styles.css` + `app/styles.css` (Glassmorphic Sci-Fi HUD) | `backend/static/index.html` (embedded CSS) |
| **J. Environment Variables** | None | None | `OPENROUTER_API_KEY` (Backend only) |
| **K. External Libraries** | `playcanvas-stable.min.js`, `gsap.min.js` | `playcanvas-stable.min.js`, `gsap.min.js`, `Leaflet.js` | `Leaflet.js` 1.9.4 |
| **L. Entry Points** | `index.html` | `index.html` (3D) & `app/index.html` (Globe) | `backend/static/index.html` / `frontend/serve.py` |
| **M. Components Required** | Splash Loader, 3D Engine Bootstrapper | Splash Loader, 3D Gate Controller, Globe Map, Sci-Fi HUD | Live Stepper, File Ingestion, GIS/Trace Inspector |

---

### SOURCE 1: ANIMATION (`shankar791/shankar-fronted`)
- **Exact Files**:
  - `index.html`: Base WebGL canvas entrypoint.
  - `styles.css`: Viewport reset and fullscreen canvas styling.
  - `playcanvas-stable.min.js`: PlayCanvas WebGL2 engine runtime (v1.65+).
  - `gsap.min.js`: GreenSock Animation Platform runtime.
  - `__settings__.js`: PlayCanvas resolution, antialiasing, and physics settings.
  - `__modules__.js`: WebAssembly Basis Universal transcoder loader.
  - `__start__.js`: Application scene loader and event lifecycle bootstrapper.
  - `__loading__.js`: Stepped odometer loader and splash screen sequence.
  - `__game-scripts.js`: Scene script registry and 3D camera orbital controller.
  - `config.json` & `2509662.json`: PlayCanvas asset registry and scene entity hierarchy.
  - `serve.py`: Multithreaded Python server with HTTP byte-range streaming.
- **Components**:
  - Stepped Odometer Loader
  - WebGL2 3D Orbit Scene
  - Multi-track Web Audio stem player
- **Assets**:
  - `files/assets/`: 3D models (`StarlinkV2-1.glb`, `Tesla-ready2.glb`, `ComputeTrayJOINED.glb`, `DiamondV4.glb`), textures, audio stems (`.ogg`), background video (`Space-compressNOAUDIO-YTfullHD.mp4`).
- **Dependencies**: `playcanvas-stable.min.js`, `gsap.min.js`, Google Basis WASM transcoder.
- **Routes**: `/` (Single Page).

---

### SOURCE 2: EARTH GLOBE (`PrakashMB-1213/-rakash-frontend`)
- **Exact Files**:
  - `index.html`: PlayCanvas entrypoint with text interceptor, "SATQUERY AI" branding, glassmorphic UI overlay, and toggle switch.
  - `__loading__.js`: Custom "SATQUERY AI" typography, "ORBITAL TELEMETRY DECODED" title, and "STEP INTO TIME" CTA.
  - `__game-scripts.js`: Automated 2-phase sine dive animation:
    - Phase 1: `diveToChip()` dives to satellite node $\to$ reveals glassmorphic switch.
    - Phase 2: `diveToClouds()` dives into atmospheric clouds on toggle $\to$ triggers `#fog-transition` $\to$ redirects to `/app/?mission=started`.
  - `app/index.html`: Earth Globe Mission Dashboard containing:
    - Fullscreen Leaflet Map container (`#map`).
    - Mission badge (`SATQUERY AI // ISRO SIH26167`, Vignan University node `17.3425° N, 78.7169° E`).
    - Center-framed glassmorphic Sci-Fi HUD bracket frame (`#satquery-hud-root`) with 3 panels:
      1. *Image Intake*: Mode selector (`OPTICAL`, `SAR FUSION`, `PAIR DIFF`), Dropzone, metadata grid, query textarea, Execute button.
      2. *Processing Telemetry*: 8-stage pipeline status indicator & live terminal window.
      3. *Analysis Output*: Metric grid (Changed Area, NDVI Delta, NDWI Water, SAVI Index), observation card, export buttons.
    - Fog transition exit overlay (`#fog-transition-out`).
  - `app/styles.css`: Complete Sci-Fi HUD stylesheet (bracket frames, glowing neon indicators, glassmorphic blur filters, custom scrollbars, responsive grid).
  - `app/app.js`: Leaflet satellite map initialization, Google Maps tile layer, tactical reticle marker, Vignan University polygon, cinematic FlyTo zoom, and UI event handlers.
- **Components**:
  - 3D Phase-Gated Transition Controller
  - Leaflet Satellite Map Engine with Google Satellite Tiles
  - Tactical Reticle Target & Campus Polygon Overlay
  - Center-Framed Sci-Fi HUD (3-Panel Intake / Telemetry / Output)
- **Assets**:
  - Leaflet 1.9.4 CSS/JS (CDN).
  - 3D assets in `files/assets/`.
- **Dependencies**: Leaflet 1.9.4, PlayCanvas, GSAP.
- **Routes**:
  - `/`: 3D Cinematic Animation & Node Dive.
  - `/app/` (`/app/?mission=started`): Earth Globe Mission Dashboard.

---

### DESTINATION: (`satquery-ai`)
- **Existing Architecture**:
  - FastAPI backend (`backend/server.py`) serving live SatQuery AI endpoints:
    - `POST /api/query`: Ingestion, validation, LangGraph agent routing, ChangeFormer bi-temporal specialist, cross-modal Optical+SAR fusion, GIS polygon/hectares computation, LLM/VLM synthesis.
    - `GET /api/samples/{filename}`: Static sample satellite imagery server.
    - `GET /`: Serves static dashboard.
  - Production static dashboard (`backend/static/index.html`): 11-stage live telemetry stepper, real raster evidence visualizer, trace debugger, sample selector.
  - 3D asset directory (`frontend/`): Standalone PlayCanvas runtime.
- **Conflicts Identified & Resolution**:
  1. **Routing Conflict**:
     - *Issue*: `satquery-ai` currently serves `backend/static/index.html` directly at `/`.
     - *Resolution*: 
       - Serve the **3D Cinematic Intro** at `/` (`frontend/index.html`).
       - Serve the **Earth Globe & SatQuery Intelligence HUD** at `/app/` (`frontend/app/index.html`).
       - Provide seamless direct switching or bypass if accessed directly.
  2. **Telemetry & API Integration Conflict**:
     - *Issue*: `prakash-frontend/app/app.js` has simulated/mock `setTimeout` animations instead of real backend execution.
     - *Resolution*: 
       - Upgrade `frontend/app/app.js` to connect directly to the **real SatQuery API** (`POST /api/query` and `GET /api/samples/{filename}`).
       - Real drag-and-drop / file upload for Single Optical, Bi-Temporal Pair, and Optical+SAR.
       - Real live 11-stage execution stepper and real-time terminal output.
       - Real GIS analysis rendering (Changed Area in hectares, NDVI/NDWI metrics, ChangeFormer segmented overlays, LLM synthesis text, confidence scores, and GeoJSON polygon generation).
  3. **Asset Deduplication**:
     - Consolidate all 3D assets (`files/assets/`) inside `frontend/files/assets/` to ensure zero redundant file copies.

---

## 2. SECOND PHASE — PROPOSED INTEGRATION

### File Mapping (`SOURCE → DESTINATION`)

```
SOURCE 1 (shankar-fronted) / SOURCE 2 (prakash-frontend)          DESTINATION (satquery-ai)
─────────────────────────────────────────────────────────────     ───────────────────────────────────────────
prakash-frontend/index.html                                   →   frontend/index.html
prakash-frontend/__loading__.js                               →   frontend/__loading__.js
prakash-frontend/__game-scripts.js                            →   frontend/__game-scripts.js
prakash-frontend/styles.css                                   →   frontend/styles.css
prakash-frontend/app/index.html                               →   frontend/app/index.html
prakash-frontend/app/styles.css                               →   frontend/app/styles.css
prakash-frontend/app/app.js                                   →   frontend/app/app.js (Upgraded with live API)
shankar-fronted/files/assets/*                                →   frontend/files/assets/*
prakash-frontend/serve.py                                     →   frontend/serve.py
```

### Dependencies Added:
- `Leaflet.js` v1.9.4 (CDN link via `https://unpkg.com/leaflet@1.9.4/dist/leaflet.js` and `.css`).
- Zero new npm dependencies required (both projects are built on ultra-performant zero-bundle native WebGL2, Leaflet, and ES6).

---

## 3. THIRD PHASE — VERIFICATION CRITERIA

1. **Animation loads**: 3D space scene boots cleanly with SatQuery AI odometer loader.
2. **Animation completes**: Sine dive executes $\to$ locks on satellite node $\to$ reveals glassmorphic switch $\to$ dives to clouds upon toggle.
3. **Earth globe loads**: Leaflet satellite engine initializes with Google Satellite imagery.
4. **Globe interaction**: Smooth FlyTo animation zooms into target coordinates; panning/zooming works.
5. **No console errors**: Zero 404s, zero script exceptions.
6. **No missing assets**: All 3D `.glb` models, `.basis` textures, audio stems, and map tiles resolve.
7. **No broken routes**: Seamless transition from `/` $\to$ `/app/?mission=started`.
8. **SatQuery analysis works**: Uploading satellite imagery triggers live `POST /api/query`.
9. **Image upload works**: Handles Single, Before/After Pair, and Optical+SAR.
10. **Backend API Contract Preserved**: Accurately consumes `answer`, `confidence`, `outputs`, `evidence_images_b64`, `synthesis_source`, `fallback_used`, `trace`, `scenario`.
11. **Final AI Answer Appears**: Rendered in Analysis Output card with confidence and GIS metrics.
12. **Zero backend regressions**: Pytest integration test suite passes 100%.
