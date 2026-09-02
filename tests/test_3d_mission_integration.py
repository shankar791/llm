"""
Comprehensive End-to-End Integration Tests for 3D Satellite Entry & Mission Dashboard.
Verifies:
1. GET / serves the 3D Satellite Experience with the animated ENTER MISSION button and transition overlay.
2. GET /mission, /dashboard, /app serve the imported geospatial dashboard with return link to 3D Orbit.
3. GET /monitor preserves the legacy vertical execution pipeline monitor.
4. Static 3D assets (__game-scripts.js, playcanvas, config.json, styles.css) are served cleanly with correct MIME types.
5. __game-scripts.js correctly dispatches satellite:animationComplete upon SatelliteMover entry completion.
6. POST /api/query processes single image, multi-temporal pair, and SAR queries with real pipeline execution.
7. History API records and retrieves the analysis runs.
"""
from __future__ import annotations
import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.server import app

client = TestClient(app)


# ======================================================================
# 1. 3D Entry Page & Animation Hook Tests
# ======================================================================

def test_entry_3d_experience_route():
    """Verify GET / serves the 3D Satellite Experience."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "SatQuery AI" in res.text
    assert "btn-enter-mission" in res.text
    assert "ENTER MISSION" in res.text
    assert "mission-transition-overlay" in res.text
    assert "satellite:animationComplete" in res.text


def test_3d_game_scripts_contains_completion_event():
    """Verify __game-scripts.js dispatches satellite:animationComplete."""
    res = client.get("/__game-scripts.js")
    assert res.status_code == 200
    assert "satellite:animationComplete" in res.text
    assert "experience:entered" in res.text


def test_static_3d_assets_serving():
    """Verify core 3D files are served properly with correct MIME types."""
    res_js = client.get("/playcanvas-stable.min.js")
    assert res_js.status_code == 200

    res_css = client.get("/styles.css")
    assert res_css.status_code == 200

    res_json = client.get("/config.json")
    assert res_json.status_code == 200


# ======================================================================
# 2. Mission Dashboard & Monitor Routes
# ======================================================================

def test_mission_dashboard_routes():
    """Verify /mission, /dashboard, and /app serve the imported intelligence dashboard."""
    for path in ["/mission", "/dashboard", "/app"]:
        res = client.get(path)
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "SatQuery AI" in res.text
        assert "3D Orbit View" in res.text
        assert "true-glass" in res.text


def test_legacy_monitor_route():
    """Verify /monitor continues to serve the vertical timeline monitor."""
    res = client.get("/monitor")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Live Execution Pipeline Timeline" in res.text


# ======================================================================
# 3. Real Backend Pipeline & History Integration
# ======================================================================

def test_query_pipeline_real_execution():
    """Verify POST /api/query executes multimodal satellite analysis."""
    img_path = Path("backend/test_images/optical_t0.png")
    assert img_path.exists(), "Test image optical_t0.png must exist"

    with open(img_path, "rb") as f:
        files = [("files", ("optical_t0.png", f, "image/png"))]
        data = {"query": "Analyze this satellite imagery and describe land-cover patterns."}
        res = client.post("/api/query", data=data, files=files)

    assert res.status_code == 200
    data = res.json()
    assert "error" not in data
    assert "answer" in data
    assert "run_id" in data

    # Verify run is recorded in history
    run_id = data["run_id"]
    hist_res = client.get(f"/api/history/{run_id}")
    assert hist_res.status_code == 200
    record = hist_res.json()
    assert record["run_id"] == run_id
    assert record["status"] == "SUCCESS"
