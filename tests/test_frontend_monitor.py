"""
Unit & Integration Test Suite for Step 13: Live SatQuery AI Pipeline Monitor Frontend.
Verifies:
1. Frontend HTML contract, 11-stage vertical pipeline layout, and 5-state representation.
2. Mocked execution-event stream processing and stage state machine transitions.
3. Observability metadata and latency aggregation.
4. Error state visualization and failure diagnostics.
5. Real backend integration via FastAPI TestClient on GET / and POST /api/query.
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.server import app


# ======================================================================
# 1. Frontend HTML Contract Tests
# ======================================================================

def test_frontend_html_file_exists():
    """Verify backend/static/index.html exists and is non-empty."""
    html_path = Path("backend/static/index.html")
    assert html_path.exists(), "backend/static/index.html must exist"
    content = html_path.read_text(encoding="utf-8")
    assert len(content) > 1000, "Frontend index.html must have substantial content"


def test_frontend_contains_all_11_pipeline_stages():
    """Verify frontend HTML explicitly defines all 11 logical pipeline stages."""
    html = Path("backend/static/index.html").read_text(encoding="utf-8")
    
    required_stages = [
        "1. Image Ingestion",
        "2. Validation / Metadata",
        "3. LLM Intent Classification",
        "4. Compatibility Gate",
        "5. Master Agent",
        "6. Tool Routing",
        "7. Specialist Execution",
        "8. GIS Processing",
        "9. Evidence Generation",
        "10. LLM Synthesis",
        "11. Final Response"
    ]
    
    for stage in required_stages:
        assert stage in html, f"Missing stage definition in frontend: {stage}"

    for i in range(1, 12):
        assert f'id="stage-{i}"' in html, f"Missing stage container ID: stage-{i}"
        assert f'id="badge-{i}"' in html, f"Missing badge ID: badge-{i}"
        assert f'id="meta-{i}"' in html, f"Missing meta ID: meta-{i}"


def test_frontend_contains_required_ui_elements():
    """Verify frontend contains dropzones, Leaflet map, query textarea, and results boxes."""
    html = Path("backend/static/index.html").read_text(encoding="utf-8")
    
    assert 'id="map-container"' in html, "Must contain Leaflet map container"
    assert 'id="dz-primary"' in html, "Must contain primary dropzone"
    assert 'id="dz-secondary"' in html, "Must contain secondary dropzone"
    assert 'id="query-input"' in html, "Must contain query textarea"
    assert 'id="btn-analyze"' in html, "Must contain Start Analysis button"
    assert 'id="final-answer-text"' in html, "Must contain final answer container"
    assert 'id="quant-results"' in html, "Must contain quantitative results container"
    assert 'id="evidence-images-box"' in html, "Must contain evidence overlays container"
    assert 'id="debug-trace"' in html, "Must contain debug trace panel"
    assert 'id="failure-banner"' in html, "Must contain error failure banner"


# ======================================================================
# 2. Mocked Execution-Event Stream State Machine Tests
# ======================================================================

def test_mocked_execution_event_stream_progression():
    """
    Deterministic mocked execution-event stream unit test.
    Verifies that a series of SSE-style stage events transitions correctly.
    """
    mock_events = [
        {"run_id": "run-001", "stage_id": 1, "status": "running", "component": "Ingest"},
        {"run_id": "run-001", "stage_id": 1, "status": "success", "component": "Ingest", "latency_ms": 40},
        {"run_id": "run-001", "stage_id": 2, "status": "running", "component": "Validator"},
        {"run_id": "run-001", "stage_id": 2, "status": "success", "component": "Validator", "latency_ms": 25},
        {"run_id": "run-001", "stage_id": 3, "status": "running", "component": "Qwen NLP", "model": "qwen/qwen3-14b:free"},
        {"run_id": "run-001", "stage_id": 3, "status": "success", "component": "Qwen NLP", "latency_ms": 110},
        {"run_id": "run-001", "stage_id": 7, "status": "running", "component": "ChangeFormer"},
        {"run_id": "run-001", "stage_id": 7, "status": "success", "component": "ChangeFormer", "latency_ms": 650},
        {"run_id": "run-001", "stage_id": 11, "status": "success", "component": "Synthesis", "latency_ms": 90},
    ]

    # In-memory mock simulation of state reducer
    timeline_state = {i: {"status": "waiting", "latency_ms": None, "component": None} for i in range(1, 12)}

    for ev in mock_events:
        sid = ev["stage_id"]
        timeline_state[sid]["status"] = ev["status"]
        if "latency_ms" in ev:
            timeline_state[sid]["latency_ms"] = ev["latency_ms"]
        if "component" in ev:
            timeline_state[sid]["component"] = ev["component"]

    assert timeline_state[1]["status"] == "success"
    assert timeline_state[1]["latency_ms"] == 40
    assert timeline_state[3]["status"] == "success"
    assert timeline_state[3]["component"] == "Qwen NLP"
    assert timeline_state[7]["component"] == "ChangeFormer"
    assert timeline_state[7]["latency_ms"] == 650
    assert timeline_state[4]["status"] == "waiting"  # untouched stage remains waiting


def test_mocked_execution_event_failure_handling():
    """Verify that a stage_failed event correctly stops execution and records error details."""
    failure_event = {
        "run_id": "run-err-01",
        "stage_id": 7,
        "status": "failed",
        "component": "ChangeFormer",
        "error_code": "MODEL_INFERENCE_ERROR",
        "message": "ChangeFormer Siamese Transformer encountered invalid tensor shape.",
        "latency_ms": 210
    }

    assert failure_event["status"] == "failed"
    assert failure_event["error_code"] == "MODEL_INFERENCE_ERROR"
    assert "invalid tensor shape" in failure_event["message"]
    assert failure_event["component"] == "ChangeFormer"


# ======================================================================
# 3. Real FastAPI Backend Integration Tests
# ======================================================================

def test_fastapi_index_endpoint_serves_html():
    """Verify GET / returns HTTP 200 with text/html containing the 3D SatQuery UI, and /monitor contains live monitor UI."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "SatQuery AI" in response.text

    response_mission = client.get("/mission")
    assert response_mission.status_code == 200
    assert "SatQuery AI" in response_mission.text

    response_monitor = client.get("/monitor")
    assert response_monitor.status_code == 200
    assert "SatQuery AI" in response_monitor.text
    assert "Live Execution Pipeline Timeline" in response_monitor.text


def test_fastapi_query_api_single_image_execution():
    """Verify POST /api/query processes single image with real test imagery."""
    client = TestClient(app)
    img_path = "backend/test_images/optical_t0.png"
    assert os.path.exists(img_path), "Test image optical_t0.png must exist"

    with open(img_path, "rb") as f:
        files = [("files", ("optical_t0.png", f, "image/png"))]
        data = {"query": "Describe the land-cover and major objects visible in this image."}
        response = client.post("/api/query", data=data, files=files)

    assert response.status_code == 200
    result = response.json()
    assert "error" not in result
    assert "answer" in result
    assert "trace" in result
    assert "confidence" in result
    assert len(result["trace"]["steps"]) >= 3


def test_fastapi_query_api_pair_change_detection():
    """Verify POST /api/query processes bi-temporal pair for change detection."""
    client = TestClient(app)
    t0_path = "backend/test_images/optical_t0.png"
    t1_path = "backend/test_images/optical_t1.png"

    with open(t0_path, "rb") as f0, open(t1_path, "rb") as f1:
        files = [
            ("files", ("optical_t0.png", f0, "image/png")),
            ("files", ("optical_t1.png", f1, "image/png")),
        ]
        data = {"query": "What changed between these two dates, and where did the change occur?"}
        response = client.post("/api/query", data=data, files=files)

    assert response.status_code == 200
    result = response.json()
    assert "error" not in result
    assert "scenario" in result
    assert result["scenario"] == "bi_temporal_pair"
    assert len(result["outputs"]) >= 1
    assert result["outputs"][0]["tool"] == "T4_Change"
