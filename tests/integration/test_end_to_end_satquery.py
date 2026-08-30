"""
End-to-End Integration Test Suite for SatQuery AI MVP.
Verifies the complete pipeline from Upload -> Query -> Intent -> Routing -> Specialist Tool -> GIS -> Evidence -> Synthesis -> Response:
1. Case A: Single-image VQA
2. Case B: Single-image Scene Captioning
3. Case C: Spatial Object Grounding
4. Case D: NDVI / Spectral Vegetation Index
5. Case E: Bi-temporal Change Detection (ChangeFormer)
6. Case F: Cross-modal Optical + SAR Feature Fusion (T5)
"""
from __future__ import annotations
import os
import pytest
from fastapi.testclient import TestClient

from backend.server import app

client = TestClient(app)

REAL_OPT_0611 = "backend/real_data/opt_0611.png"
REAL_OPT_0810 = "backend/real_data/opt_0810.png"
REAL_SAR_0810 = "backend/real_data/sar_0810.png"
TEST_OPT_T0 = "backend/test_images/optical_t0.png"
TEST_OPT_T1 = "backend/test_images/optical_t1.png"
TEST_SAR_T1 = "backend/test_images/sar_t1.png"


def test_case_a_single_image_vqa():
    """Case A — Single-image VQA on real satellite imagery."""
    img_path = REAL_OPT_0611 if os.path.exists(REAL_OPT_0611) else TEST_OPT_T0
    with open(img_path, "rb") as f:
        files = [("files", (os.path.basename(img_path), f, "image/png"))]
        data = {"query": "What objects and major land-cover types are visible in this image?"}
        response = client.post("/api/query", data=data, files=files)

    assert response.status_code == 200
    res = response.json()
    assert "error" not in res
    assert "answer" in res
    assert len(res["answer"]) > 10
    assert res["confidence"] > 0.0
    assert "trace" in res
    assert any(s.get("action") == "classify_task" for s in res["trace"]["steps"])


def test_case_b_single_image_caption():
    """Case B — Single-image Captioning on real satellite imagery."""
    img_path = REAL_OPT_0611 if os.path.exists(REAL_OPT_0611) else TEST_OPT_T0
    with open(img_path, "rb") as f:
        files = [("files", (os.path.basename(img_path), f, "image/png"))]
        data = {"query": "Describe this satellite scene."}
        response = client.post("/api/query", data=data, files=files)

    assert response.status_code == 200
    res = response.json()
    assert "error" not in res
    assert "answer" in res
    assert len(res["answer"]) > 10
    assert "trace" in res


def test_case_c_grounding():
    """Case C — Spatial Object Grounding on real satellite imagery."""
    img_path = REAL_OPT_0810 if os.path.exists(REAL_OPT_0810) else TEST_OPT_T0
    with open(img_path, "rb") as f:
        files = [("files", (os.path.basename(img_path), f, "image/png"))]
        data = {"query": "Locate the major buildings visible in this satellite image."}
        response = client.post("/api/query", data=data, files=files)

    assert response.status_code == 200
    res = response.json()
    assert "error" not in res
    assert "answer" in res
    assert "trace" in res


def test_case_d_ndvi_spectral_index():
    """Case D — NDVI / Spectral Vegetation Index analysis."""
    img_path = TEST_OPT_T0 if os.path.exists(TEST_OPT_T0) else REAL_OPT_0611
    with open(img_path, "rb") as f:
        files = [("files", (os.path.basename(img_path), f, "image/png"))]
        data = {"query": "Calculate NDVI and estimate vegetation cover in this scene."}
        response = client.post("/api/query", data=data, files=files)

    assert response.status_code == 200
    res = response.json()
    assert "error" not in res
    assert "answer" in res
    assert "trace" in res


def test_case_e_bitemporal_change():
    """Case E — Bi-temporal change detection on real before/after pair."""
    t0 = REAL_OPT_0611 if os.path.exists(REAL_OPT_0611) else TEST_OPT_T0
    t1 = REAL_OPT_0810 if os.path.exists(REAL_OPT_0810) else TEST_OPT_T1
    with open(t0, "rb") as f0, open(t1, "rb") as f1:
        files = [
            ("files", (os.path.basename(t0), f0, "image/png")),
            ("files", (os.path.basename(t1), f1, "image/png")),
        ]
        data = {"query": "What changed between these images?"}
        response = client.post("/api/query", data=data, files=files)

    assert response.status_code == 200
    res = response.json()
    assert "error" not in res
    assert res.get("scenario") == "bi_temporal_pair"
    assert "answer" in res
    assert any(o.get("tool") == "T4_Change" for o in res.get("outputs", []))


def test_case_f_optical_sar_fusion():
    """Case F — Cross-modal optical + SAR feature fusion (T5)."""
    opt = REAL_OPT_0810 if os.path.exists(REAL_OPT_0810) else TEST_OPT_T1
    sar = REAL_SAR_0810 if os.path.exists(REAL_SAR_0810) else TEST_SAR_T1
    with open(opt, "rb") as fo, open(sar, "rb") as fs:
        files = [
            ("files", (os.path.basename(opt), fo, "image/png")),
            ("files", (os.path.basename(sar), fs, "image/png")),
        ]
        data = {"query": "Use optical and SAR imagery together to analyze built-up and water-covered regions."}
        response = client.post("/api/query", data=data, files=files)

    assert response.status_code == 200
    res = response.json()
    assert "error" not in res
    assert res.get("scenario") == "cross_modal_pair"
    assert any(o.get("tool") == "T5_OpticalSAR" for o in res.get("outputs", []))
    assert len(res.get("evidence_images_b64", [])) >= 1
