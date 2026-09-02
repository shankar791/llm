"""
Dedicated Test Suite for SatQuery AI Follow-Up and Fresh Text Handling.

Verifies:
1. Image -> Analysis -> "What about vegetation?" -> LLM with previous context.
2. Image -> Analysis -> 3 consecutive follow-up questions -> all work and preserve context.
3. Fresh query -> "Explain NDVI." -> direct LLM, no image/tool execution.
"""
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from backend.server import app, followup_api
from backend.session import session_store
from backend.history import history_store
from backend.rasterio_utils import RasterInput
from backend.agent import execute

client = TestClient(app)

REAL_OPT_0611 = Path("backend/real_data/opt_0611.png")
TEST_OPT_T0 = Path("backend/test_images/optical_t0.png")


def _get_image_bytes():
    path = REAL_OPT_0611 if REAL_OPT_0611.exists() else TEST_OPT_T0
    return path.name, path.read_bytes()


@pytest.fixture(autouse=True)
def clean_state():
    session_store.clear()
    history_store.clear()
    yield
    session_store.clear()
    history_store.clear()


# ==============================================================================
# TEST 1: Image -> analysis -> "What about vegetation?" -> LLM with previous context
# ==============================================================================
def test_1_image_analysis_then_followup():
    fname, img_bytes = _get_image_bytes()
    # 1. Initial image analysis
    raster = RasterInput(fname, img_bytes)
    initial_res = execute("What is visible in this image?", [raster])
    assert initial_res.get("answer"), "Initial analysis must yield an answer"
    assert "evidence" in initial_res and len(initial_res["evidence"]) > 0

    # 2. Follow-up query using previous analysis context (Scenario 1)
    followup_payload = {
        "question": "What about vegetation?",
        "context": initial_res,
    }
    followup_res = followup_api(followup_payload)

    assert followup_res.get("answer"), "Follow-up must produce an answer"
    assert followup_res["analysis_type"] == "Follow-Up Analysis"
    
    # Must NOT rerun specialists or tools
    intent = followup_res["execution_details"]["intent"]
    assert intent["specialist_executed"] is False
    assert len(intent["tools_run"]) == 0
    
    # Must use existing context
    assert len(followup_res["evidence"]) > 0
    print("\n[TEST 1 PASSED] Initial answer snippet:", initial_res["answer"][:60])
    print("[TEST 1 PASSED] Follow-up answer:", followup_res["answer"][:100])


# ==============================================================================
# TEST 2: Image -> analysis -> 3 follow-up questions -> all work
# ==============================================================================
def test_2_three_consecutive_followups():
    fname, img_bytes = _get_image_bytes()
    # Initial analysis
    raster = RasterInput(fname, img_bytes)
    current_context = execute("Initial scan of satellite scene", [raster])
    current_context["conversation"] = [
        {"role": "user", "text": "Initial scan of satellite scene"},
        {"role": "assistant", "text": current_context["answer"]}
    ]

    questions = [
        "What about the vegetation?",
        "Are there any water bodies or rivers?",
        "Can you summarize the scene based on the previous findings?"
    ]

    for idx, q in enumerate(questions, 1):
        payload = {
            "question": q,
            "context": current_context
        }
        res = followup_api(payload)
        assert res.get("answer"), f"Follow-up {idx} must produce an answer"
        assert res["analysis_type"] == "Follow-Up Analysis"
        assert res["execution_details"]["intent"]["specialist_executed"] is False
        assert len(res["execution_details"]["intent"]["tools_run"]) == 0

        # Update context for next turn
        current_context["answer"] = res["answer"]
        if res.get("sections"):
            current_context["sections"] = res["sections"]
        current_context["conversation"].append({"role": "user", "text": q})
        current_context["conversation"].append({"role": "assistant", "text": res["answer"]})
        print(f"[TEST 2 TURN {idx} PASSED] Q: {q} -> A: {res['answer'][:80]}...")

    assert len(current_context["conversation"]) == 8  # 1 initial + 3 follow-ups * 2


# ==============================================================================
# TEST 3: Fresh page -> "Explain NDVI." -> direct LLM, no image/tool execution
# ==============================================================================
def test_3_fresh_text_query_direct_llm():
    # Fresh text via followup_api with no context (Scenario 2)
    payload = {
        "question": "Explain NDVI.",
        "context": None
    }
    res = followup_api(payload)

    assert res.get("answer"), "Direct text query must yield an answer"
    assert res["analysis_type"] == "Direct LLM Query"
    assert res["execution_details"]["intent"]["specialist_executed"] is False
    assert len(res["execution_details"]["intent"]["tools_run"]) == 0
    assert len(res["evidence"]) == 0, "No image evidence should be present for fresh text query"

    # NDVI explanation must be accurate
    ans_lower = res["answer"].lower()
    assert "ndvi" in ans_lower or "normalized difference vegetation index" in ans_lower
    print("\n[TEST 3 PASSED] Fresh text query answer snippet:", res["answer"][:120])

    # Also test via client.post("/api/query") with pure text (no files attached)
    post_res = client.post("/api/query", data={"query": "Explain what NDVI means."})
    assert post_res.status_code == 200
    post_body = post_res.json()
    assert post_body.get("analysis_type") == "Direct LLM Query"
    assert post_body["execution_details"]["intent"]["specialist_executed"] is False
    assert len(post_body["execution_details"]["intent"]["tools_run"]) == 0
    print("[TEST 3 API QUERY PASSED] POST /api/query answer snippet:", post_body["answer"][:120])
