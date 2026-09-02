"""
End-to-End Test Suite for Step 4 — Verification of Chat and Multi-Turn Workflows.
Verifies:
1. New session -> first question -> LLM response via /api/chat.
2. Follow-up question -> previous conversation is remembered.
3. Image analysis -> follow-up about that image.
4. Image A -> Image B -> 'compare this with the previous image'.
5. New session -> old session context is NOT available.
6. Real/mocked provider visibility and transparency in execution details.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.server import app
from backend.session import session_store
from backend.chat import execute_chat_turn
from ai.llm.provider import MockLLMProvider


client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_sessions():
    """Ensure clean in-memory state for every test."""
    session_store.clear_all_sessions()
    yield
    session_store.clear_all_sessions()


# ==============================================================================
# TEST 1: New session -> first question -> LLM response via /api/chat
# ==============================================================================
def test_new_session_first_question():
    session_id = "e2e_test_new_session_01"

    # Mock provider capturing query
    mock_provider = MockLLMProvider(default_response="NDVI (Normalized Difference Vegetation Index) quantifies vegetation greenness.")

    res = execute_chat_turn(session_id=session_id, query="What is NDVI?", llm_provider=mock_provider)

    assert res["session_id"] == session_id
    assert res["query"] == "What is NDVI?"
    assert "NDVI" in res["answer"]
    assert res["model"] == "mock-llm-v1"

    # Confirm storage in session store
    sess = session_store.get_session_memory(session_id)
    assert len(sess["conversation"]) == 2
    assert sess["conversation"][0]["role"] == "user"
    assert sess["conversation"][1]["role"] == "assistant"


# ==============================================================================
# TEST 2: Follow-up question -> previous conversation is remembered
# ==============================================================================
def test_followup_question_remembers_previous_conversation():
    session_id = "e2e_test_followup_02"

    captured_prompts = []

    def tracking_handler(messages, **kwargs):
        captured_prompts.append(messages[-1]["content"])
        return "The typical formula is (NIR - Red) / (NIR + Red)."

    mock_provider = MockLLMProvider(custom_handler=tracking_handler)

    # Turn 1
    execute_chat_turn(session_id=session_id, query="What is NDVI?", llm_provider=mock_provider)

    # Turn 2: Follow-up asking for formula
    res2 = execute_chat_turn(session_id=session_id, query="How is it calculated?", llm_provider=mock_provider)

    assert len(captured_prompts) == 2
    turn2_prompt = captured_prompts[1]

    # Turn 2 must contain Turn 1 history
    assert "What is NDVI?" in turn2_prompt
    assert "How is it calculated?" in turn2_prompt
    assert "NIR - Red" in res2["answer"]

    # Verify session store contains all 4 messages
    sess = session_store.get_session_memory(session_id)
    assert len(sess["conversation"]) == 4


# ==============================================================================
# TEST 3: Image analysis -> follow-up about that image
# ==============================================================================
def test_image_analysis_followed_by_followup():
    session_id = "e2e_test_img_followup_03"

    # 1. Register image analysis (simulating initial specialist run)
    session_store.register_image(
        session_id=session_id,
        image_id="img_harbor_99",
        filename="harbor_optical.png",
        analysis={"finding": "Maritime port with 8 crane gantries and 3 berths."},
        task="T1_VQA",
        evidence=[{"evidence_id": "E1", "label": "Crane", "coverage_pct": 11.5}],
        gis_results={"crane_count": 8, "area_ha": 35.2},
    )

    captured_prompt = None

    def handler(messages, **kwargs):
        nonlocal captured_prompt
        captured_prompt = messages[-1]["content"]
        return "Based on the port analysis, there are 8 crane gantries detected."

    mock_provider = MockLLMProvider(custom_handler=handler)

    res = execute_chat_turn(session_id=session_id, query="How many cranes were identified in that harbor?", llm_provider=mock_provider)

    assert captured_prompt is not None
    assert "img_harbor_99" in captured_prompt
    assert "Maritime port with 8 crane gantries and 3 berths." in captured_prompt
    assert "crane_count=8" in captured_prompt
    assert "8 crane gantries" in res["answer"]


# ==============================================================================
# TEST 4: Image A -> Image B -> 'compare this with the previous image'
# ==============================================================================
def test_compare_image_a_with_image_b():
    session_id = "e2e_test_compare_04"

    # Image A: Pre-event baseline
    session_store.register_image(
        session_id=session_id,
        image_id="img_t0_optical",
        filename="flood_t0.tif",
        analysis={"finding": "Normal dry riverbed with riparian vegetation."},
        task="T2_Caption",
        gis_results={"water_area_ha": 4.2},
    )

    # Image B: Post-event inundation
    session_store.register_image(
        session_id=session_id,
        image_id="img_t1_optical",
        filename="flood_t1.tif",
        analysis={"finding": "Extensive floodplain inundation across agricultural perimeter."},
        task="T4_Change",
        evidence=[{"evidence_id": "E2", "label": "Floodwater", "coverage_pct": 42.8}],
        gis_results={"water_area_ha": 86.4, "changed_area_ha": 82.2},
    )

    captured_prompt = None

    def handler(messages, **kwargs):
        nonlocal captured_prompt
        captured_prompt = messages[-1]["content"]
        return "Comparing img_t0_optical to img_t1_optical, water surface expanded by 82.2 hectares."

    mock_provider = MockLLMProvider(custom_handler=handler)

    res = execute_chat_turn(
        session_id=session_id,
        query="Compare this with the previous image. What changed between them?",
        llm_provider=mock_provider,
    )

    assert captured_prompt is not None
    # Both image IDs are present in prompt
    assert "Image [img_t0_optical]" in captured_prompt
    assert "Normal dry riverbed" in captured_prompt
    assert "Image [img_t1_optical]" in captured_prompt
    assert "Extensive floodplain inundation" in captured_prompt
    assert "changed_area_ha=82.2" in captured_prompt
    assert "expanded by 82.2 hectares" in res["answer"]


# ==============================================================================
# TEST 5: New session -> old session context is NOT available
# ==============================================================================
def test_new_session_does_not_leak_old_session_context():
    session_old = "e2e_session_old"
    session_new = "e2e_session_new"

    # Populate old session
    session_store.add_user_message(session_old, "Secret classified coordinates 45.12N, 9.34E")
    session_store.register_image(session_old, image_id="img_secret", filename="secret.tif", analysis="Restricted base")

    captured_new_prompt = None

    def handler(messages, **kwargs):
        nonlocal captured_new_prompt
        captured_new_prompt = messages[-1]["content"]
        return "Fresh session initialized."

    mock_provider = MockLLMProvider(custom_handler=handler)

    execute_chat_turn(session_id=session_new, query="Hello, what can you do?", llm_provider=mock_provider)

    assert captured_new_prompt is not None
    assert "Secret" not in captured_new_prompt
    assert "45.12N" not in captured_new_prompt
    assert "img_secret" not in captured_new_prompt
    assert "Restricted base" not in captured_new_prompt


# ==============================================================================
# TEST 6: /api/chat HTTP endpoint verification
# ==============================================================================
def test_api_chat_http_endpoint():
    session_id = "e2e_http_chat_session"

    payload = {
        "session_id": session_id,
        "query": "Can you summarize your capabilities in Earth observation?",
    }
    res = client.post("/api/chat", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["session_id"] == session_id
    assert data["query"] == "Can you summarize your capabilities in Earth observation?"
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "context" in data
    assert "model" in data
