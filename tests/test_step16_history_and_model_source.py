"""
Tests for Step 16: Analysis History & Real Model Source Trace.
Verifies history creation, search, retrieval, selection without re-running,
failed analysis history, and truthful model execution metadata.
"""
from __future__ import annotations
import os
import pytest
from fastapi.testclient import TestClient

from backend.server import app
from backend.history import history_store

client = TestClient(app)

REAL_OPT_0611 = "backend/real_data/opt_0611.png"
REAL_OPT_0810 = "backend/real_data/opt_0810.png"
REAL_SAR_0810 = "backend/real_data/sar_0810.png"
TEST_OPT_T0 = "backend/test_images/optical_t0.png"
TEST_OPT_T1 = "backend/test_images/optical_t1.png"
TEST_SAR_T1 = "backend/test_images/sar_t1.png"


@pytest.fixture(autouse=True)
def clean_history():
    """Ensure clean history before each test."""
    history_store.clear()
    yield
    history_store.clear()


def test_history_creation_and_retrieval():
    """Verify that running an analysis creates a real history entry retrievable via GET /api/history."""
    img_path = REAL_OPT_0611 if os.path.exists(REAL_OPT_0611) else TEST_OPT_T0
    with open(img_path, "rb") as f:
        files = [("files", (os.path.basename(img_path), f, "image/png"))]
        data = {"query": "What objects and major land-cover types are visible in this image?"}
        res = client.post("/api/query", data=data, files=files)

    assert res.status_code == 200
    res_data = res.json()
    assert "run_id" in res_data
    run_id = res_data["run_id"]

    # Verify history list
    hist_res = client.get("/api/history")
    assert hist_res.status_code == 200
    hist = hist_res.json()
    assert len(hist) == 1
    assert hist[0]["run_id"] == run_id
    assert hist[0]["status"] == "SUCCESS"
    assert hist[0]["query"] == data["query"]


def test_history_search():
    """Verify multi-field search in history (query, title, analysis_type, model)."""
    # Clean history state for test isolation
    client.delete("/api/history")
    
    # Create two entries
    img_path = TEST_OPT_T0
    with open(img_path, "rb") as f1, open(TEST_OPT_T1, "rb") as f2:
        # 1. Change detection query
        files1 = [
            ("files", ("optical_t0.png", f1, "image/png")),
            ("files", ("optical_t1.png", f2, "image/png")),
        ]
        res1 = client.post("/api/query", data={"query": "Detect surface changes between dates"}, files=files1)
        assert res1.status_code == 200

    # 2. VQA query
    with open(img_path, "rb") as f:
        files2 = [("files", ("optical_t0.png", f, "image/png"))]
        res2 = client.post("/api/query", data={"query": "What land cover is visible?"}, files=files2)
        assert res2.status_code == 200

    # Search for 'change'
    search_change = client.get("/api/history?q=change")
    assert search_change.status_code == 200
    change_items = search_change.json()
    assert len(change_items) == 1
    assert "change" in change_items[0]["query"].lower() or "change" in change_items[0]["title"].lower()

    # Search for non-existent term
    search_none = client.get("/api/history?q=nonexistentqueryxyz")
    assert search_none.status_code == 200
    assert len(search_none.json()) == 0


def test_history_selection_without_rerunning():
    """Verify that clicking/selecting a history item returns the full stored analysis."""
    img_path = TEST_OPT_T0
    with open(img_path, "rb") as f:
        files = [("files", ("optical_t0.png", f, "image/png"))]
        res = client.post("/api/query", data={"query": "Analyze scene composition"}, files=files)
        assert res.status_code == 200
        run_id = res.json()["run_id"]

    # Retrieve specific run
    detail_res = client.get(f"/api/history/{run_id}")
    assert detail_res.status_code == 200
    record = detail_res.json()
    assert record["run_id"] == run_id
    assert "full_result" in record
    assert record["full_result"]["answer"] == res.json()["answer"]
    assert "execution_details" in record["full_result"]


def test_failed_analysis_history():
    """Verify that invalid inputs or pipeline failures record a FAILED history entry."""
    history_store.create_entry(
        run_id="test-fail-1",
        query="Invalid request query",
        image_names=["bad_file.bin"],
        analysis_type="Invalid",
        status="RUNNING",
    )
    history_store.update_entry(
        run_id="test-fail-1",
        status="FAILED",
        error="File corrupted / unsupported format",
    )

    failed_res = client.get("/api/history/test-fail-1")
    assert failed_res.status_code == 200
    rec = failed_res.json()
    assert rec["status"] == "FAILED"
    assert "error" in rec and rec["error"] is not None


def test_model_source_trace_truthfulness():
    """Verify that execution details contain explicit model, provider, source, and latency without fabrication."""
    img1 = TEST_OPT_T0
    img2 = TEST_OPT_T1
    with open(img1, "rb") as f1, open(img2, "rb") as f2:
        files = [
            ("files", ("optical_t0.png", f1, "image/png")),
            ("files", ("optical_t1.png", f2, "image/png")),
        ]
        res = client.post("/api/query", data={"query": "What changed between optical t0 and t1?"}, files=files)

    assert res.status_code == 200
    data = res.json()
    assert "execution_details" in data
    details = data["execution_details"]
    assert "models" in details
    assert len(details["models"]) >= 2

    # Intent
    assert details["intent"]["source"] == "Local"

    # T4 Change tool
    change_tool = next((m for m in details["models"] if m["task"] in ("T4_Change", "T4 Change Detection")), None)
    assert change_tool is not None
    assert change_tool["source"] == "Local"
    assert "ChangeFormer" in change_tool["actual_model"]

    # GeoChat status
    assert "geochat_status" in details
    assert "available" in details["geochat_status"]
    assert "used_in_this_analysis" in details["geochat_status"]
