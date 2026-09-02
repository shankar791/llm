"""
Unit & Integration Test Suite for Step 2: Analysis Sessions & Intelligent Follow-Up Routing.
Verifies session creation, raster caching, follow-up context routing, specialist re-routing,
stable evidence IDs, conversation history preservation, and error handling.
"""
from __future__ import annotations

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.server import app
from backend.session import session_store
from backend.history import history_store
from backend.rasterio_utils import RasterInput
from backend.agent import execute, execute_followup

client = TestClient(app)

TEST_OPT_T0 = Path("backend/test_images/optical_t0.png")
REAL_OPT_0611 = Path("backend/real_data/opt_0611.png")


@pytest.fixture(autouse=True)
def clean_state():
    """Ensure clean session and history storage before and after each test."""
    session_store.clear()
    history_store.clear()
    yield
    session_store.clear()
    history_store.clear()


@pytest.fixture(autouse=True)
def fast_execution(monkeypatch):
    """Ensure unit tests run deterministically and fast without remote API latency."""
    from ai.synthesis.fallback import DeterministicFallbackFormatter
    from ai.synthesis.llm import LLMSynthesizer
    from tools.vqa import VQATool
    from tools.captioning import CaptioningTool
    from tools.grounding import GroundingTool

    formatter = DeterministicFallbackFormatter()

    def mock_synth(self, query, tool_results=None, confidence=None, confidence_status="uncalibrated",
                   geojson=None, intent=None, error=None, existing_evidence=None,
                   start_counter=1, conversation_history=None, **kwargs):
        return formatter.format(
            query=query,
            tool_results=tool_results or [],
            confidence=confidence,
            confidence_status=confidence_status,
            geojson=geojson,
            intent=intent,
            error=error,
            existing_evidence=existing_evidence,
        )

    monkeypatch.setattr(LLMSynthesizer, "synthesize", mock_synth)

    orig_vqa = VQATool.run
    def mock_vqa(self, query, image_bytes=None, **kwargs):
        return orig_vqa(self, query, image_bytes=image_bytes, mode="mock")
    monkeypatch.setattr(VQATool, "run", mock_vqa)

    orig_cap = CaptioningTool.run
    def mock_cap(self, image_bytes=None, **kwargs):
        return orig_cap(self, image_bytes=image_bytes, mode="mock")
    monkeypatch.setattr(CaptioningTool, "run", mock_cap)

    orig_gnd = GroundingTool.run
    def mock_gnd(self, query, image_bytes=None, **kwargs):
        return orig_gnd(self, query, image_bytes=image_bytes, mode="mock")
    monkeypatch.setattr(GroundingTool, "run", mock_gnd)


def _get_test_image_bytes() -> tuple[str, bytes]:
    path = REAL_OPT_0611 if REAL_OPT_0611.exists() else TEST_OPT_T0
    return path.name, path.read_bytes()


# ==============================================================================
# TEST 1: Initial Image Analysis Creates Session ID & Preserves Context
# ==============================================================================
def test_initial_analysis_creates_session():
    fname, img_bytes = _get_test_image_bytes()
    files = [("files", (fname, img_bytes, "image/png"))]
    data = {"query": "Analyze this satellite imagery and describe major land-cover features."}

    res = client.post("/api/query", data=data, files=files)
    assert res.status_code == 200, res.text
    body = res.json()

    # Session ID must be created and returned
    assert "session_id" in body
    assert body["session_id"] is not None
    assert len(body["session_id"]) > 0

    session_id = body["session_id"]
    assert body["run_id"] == session_id

    # Structured fields must be present
    assert "answer" in body and len(body["answer"]) > 0
    assert "sections" in body and isinstance(body["sections"], list)
    assert "claims" in body and isinstance(body["claims"], list)
    assert "uncertainties" in body and isinstance(body["uncertainties"], list)
    assert "evidence" in body and isinstance(body["evidence"], list)
    assert len(body["evidence"]) > 0

    # Evidence IDs must start with E1, E2...
    ev_ids = [e["evidence_id"] for e in body["evidence"]]
    assert "E1" in ev_ids

    # Session must be persisted in session_store
    sess = session_store.get_session(session_id)
    assert sess is not None
    assert sess["session_id"] == session_id
    assert len(sess["conversation"]) == 2  # user initial query + assistant initial answer
    assert sess["conversation"][0]["role"] == "user"
    assert sess["conversation"][1]["role"] == "assistant"

    # Cached rasters must exist on disk
    cached_rasters = session_store.get_session_rasters(session_id)
    assert len(cached_rasters) == 1
    assert cached_rasters[0].filename == fname


# ==============================================================================
# TEST 2: Follow-Up Question Uses Existing Context (NO Specialist Rerun)
# ==============================================================================
def test_followup_uses_existing_context_without_specialist_rerun():
    fname, img_bytes = _get_test_image_bytes()
    # 1. Initial analysis
    res1 = client.post("/api/query", data={"query": "Analyze scene features."}, files=[("files", (fname, img_bytes, "image/png"))])
    assert res1.status_code == 200
    initial_data = res1.json()
    session_id = initial_data["session_id"]
    initial_evidence = initial_data["evidence"]

    # 2. Follow-up query without image
    res2 = client.post("/api/query", data={
        "session_id": session_id,
        "query": "What did you find about vegetation?"
    })
    assert res2.status_code == 200, res2.text
    followup_data = res2.json()

    assert followup_data["session_id"] == session_id
    assert followup_data["analysis_type"] == "Follow-Up Analysis"
    assert followup_data["execution_details"]["intent"]["specialist_executed"] is False
    assert len(followup_data["execution_details"]["intent"]["tools_run"]) == 0

    # Answer must be grounded and structured
    assert len(followup_data["answer"]) > 0
    assert "claims" in followup_data
    assert "evidence" in followup_data

    # Evidence IDs must remain stable and unchanged
    initial_ids = [e["evidence_id"] for e in initial_evidence]
    followup_ids = [e["evidence_id"] for e in followup_data["evidence"]]
    assert initial_ids == followup_ids


# ==============================================================================
# TEST 3: Consecutive Follow-Up Questions Preserve Multi-Turn Conversation
# ==============================================================================
def test_consecutive_followups_preserve_conversation():
    fname, img_bytes = _get_test_image_bytes()
    # Turn 1: Initial
    res1 = client.post("/api/query", data={"query": "Initial scan"}, files=[("files", (fname, img_bytes, "image/png"))])
    session_id = res1.json()["session_id"]

    # Turn 2: First Follow-Up
    res2 = client.post("/api/query", data={"session_id": session_id, "query": "What about the vegetation?"})
    assert res2.status_code == 200

    # Turn 3: Second Follow-Up
    res3 = client.post("/api/query", data={"session_id": session_id, "query": "Why do you think that?"})
    assert res3.status_code == 200
    turn3_data = res3.json()

    # Session conversation must have 6 turns (3 user + 3 assistant)
    sess = session_store.get_session(session_id)
    assert len(sess["conversation"]) == 6
    roles = [m["role"] for m in sess["conversation"]]
    assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]
    assert sess["conversation"][2]["content"] == "What about the vegetation?"
    assert sess["conversation"][4]["content"] == "Why do you think that?"


# ==============================================================================
# TEST 4: Follow-Up Requiring Specialist (T3_Ground) Runs and Adds New Evidence IDs
# ==============================================================================
def test_followup_invoking_specialist_assigns_new_evidence_ids():
    fname, img_bytes = _get_test_image_bytes()
    # Initial analysis (T1_VQA / caption)
    res1 = client.post("/api/query", data={"query": "Analyze scene"}, files=[("files", (fname, img_bytes, "image/png"))])
    session_id = res1.json()["session_id"]
    initial_evidence_count = len(res1.json()["evidence"])
    initial_ids = [e["evidence_id"] for e in res1.json()["evidence"]]

    # Follow-up explicitly requiring building detection
    res2 = client.post("/api/query", data={
        "session_id": session_id,
        "query": "Detect the buildings precisely with bounding boxes."
    })
    assert res2.status_code == 200
    data2 = res2.json()

    # Verify T3_Ground was invoked
    assert data2["execution_details"]["intent"]["specialist_executed"] is True
    assert "T3_Ground" in data2["execution_details"]["intent"]["tools_run"]

    # Verify old evidence IDs were preserved and new ones appended
    new_evidence = data2["evidence"]
    new_ids = [e["evidence_id"] for e in new_evidence]
    for old_id in initial_ids:
        assert old_id in new_ids

    # New evidence IDs must start strictly after max old ID
    assert len(new_evidence) > initial_evidence_count
    expected_new_id = f"E{initial_evidence_count + 1}"
    assert expected_new_id in new_ids


# ==============================================================================
# TEST 5: Invalid Session ID Returns HTTP 404
# ==============================================================================
def test_invalid_session_id_returns_404():
    res = client.post("/api/query", data={
        "session_id": "non-existent-session-id-12345",
        "query": "What about vegetation?"
    })
    assert res.status_code == 404
    body = res.json()
    assert "not found or expired" in body.get("error", "").lower() or "not found or expired" in body.get("detail", "").lower()
    assert body.get("code") == "SESSION_NOT_FOUND"


# ==============================================================================
# TEST 6: Second Independent Analysis Does NOT Contaminate First Session
# ==============================================================================
def test_session_isolation():
    fname, img_bytes = _get_test_image_bytes()
    # Session A
    res_a = client.post("/api/query", data={"query": "First scene analysis"}, files=[("files", (fname, img_bytes, "image/png"))])
    sess_id_a = res_a.json()["session_id"]

    # Session B
    res_b = client.post("/api/query", data={"query": "Second scene analysis"}, files=[("files", (fname, img_bytes, "image/png"))])
    sess_id_b = res_b.json()["session_id"]

    assert sess_id_a != sess_id_b

    # Add follow-up to Session A
    client.post("/api/query", data={"session_id": sess_id_a, "query": "Follow-up for session A"})

    # Verify Session B is unaffected
    sess_b = session_store.get_session(sess_id_b)
    assert len(sess_b["conversation"]) == 2  # Only initial turn

    sess_a = session_store.get_session(sess_id_a)
    assert len(sess_a["conversation"]) == 4  # 2 turns


# ==============================================================================
# TEST 7: JSON Payload Support for Follow-Ups
# ==============================================================================
def test_json_payload_followup():
    fname, img_bytes = _get_test_image_bytes()
    res1 = client.post("/api/query", data={"query": "Initial check"}, files=[("files", (fname, img_bytes, "image/png"))])
    session_id = res1.json()["session_id"]

    # Send follow-up using application/json
    res2 = client.post("/api/query", json={
        "session_id": session_id,
        "query": "What did you find about land cover?"
    })
    assert res2.status_code == 200
    data = res2.json()
    assert data["session_id"] == session_id
    assert "answer" in data
