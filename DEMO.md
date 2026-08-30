# SatQuery AI — Live Demo & Operational Guide

This document details the exact, verified steps to run the complete SatQuery AI End-to-End MVP system.

---

## 1. Quick Start / Launch Commands

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Dependencies installed: `pip install -r requirements.txt`
- Optional: Set `OPENROUTER_API_KEY` in environment for live OpenRouter multimodal models (`Gemma 4 26B`, `Gemma 4 31B`, `Nemotron 3 Ultra 550B`).

### Launch the Backend & Live Monitor Frontend
Run from repository root:
```powershell
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:8000`**

---

## 2. Guaranteed Working Demonstration Scenarios

### Demo Scenario 1: Single-Image VQA & Land-Cover Composition
1. Set upload mode to **Single Image**.
2. Upload image: `backend/real_data/opt_0611.png` (or click on example dropzone).
3. Select or enter query:
   ```
   What objects and major land-cover types are visible in this image?
   ```
4. Click **Start Analysis Pipeline**.
5. Observe the 11-stage vertical execution timeline updating from **Ingestion** $\to$ **Validation** $\to$ **LLM Intent** $\to$ **Tool Routing** $\to$ **Specialist Execution** $\to$ **LLM Synthesis** $\to$ **Final Response**.
6. View the final grounded answer and execution latency.

---

### Demo Scenario 2: Bi-Temporal Change Detection (`ChangeFormer` + GIS Engine)
1. Set upload mode to **Before / After Pair**.
2. Upload:
   - **T0 (Before)**: `backend/real_data/opt_0611.png`
   - **T1 (After)**: `backend/real_data/opt_0810.png`
3. Enter query:
   ```
   What changed between these two dates, and where did the change occur?
   ```
4. Click **Start Analysis Pipeline**.
5. Observe execution:
   - **Scenario**: `bi_temporal_pair`
   - **Specialist**: `ChangeFormer` Siamese Vision Transformer
   - **GIS Engine**: Computes changed surface area (hectares, $m^2$), cluster polygons, and severity.
   - **Visual Evidence**: Rendered change overlay image displayed in the right panel.

---

### Demo Scenario 3: Cross-Modal Optical + SAR Feature Fusion (`T5_OpticalSAR`)
1. Set upload mode to **Optical + SAR**.
2. Upload:
   - **Optical**: `backend/real_data/opt_0810.png` (or `backend/test_images/optical_t1.png`)
   - **SAR**: `backend/real_data/sar_0810.png` (or `backend/test_images/sar_t1.png`)
3. Enter query:
   ```
   Use the optical and SAR images together to identify built-up and water-covered regions.
   ```
4. Click **Start Analysis Pipeline**.
5. Observe cross-modal fusion combining optical spectral properties with SAR speckle/texture analysis for cloud/shadow-resilient land-cover classification.

---

## 3. Automated End-to-End Verification

Run the full end-to-end integration test suite:
```powershell
python -m pytest tests/integration/test_end_to_end_satquery.py -v
```

Run the complete regression test suite:
```powershell
python -m pytest tests/ --ignore=tests/evaluation/live_smoke_test_step12.py -v
```
