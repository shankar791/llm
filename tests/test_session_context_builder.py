"""
Unit Test Suite for Step 2 — Session Context Builder.
Verifies:
1. Empty session produces valid empty context.
2. Conversation history is included.
3. Image analysis is included.
4. Multiple images remain distinguishable.
5. Evidence/GIS results are preserved exactly.
6. Recent-message limit works.
7. Different sessions never mix context.
8. Current query is included.
9. No raw image bytes are inserted into text context.
10. Missing analysis/evidence does not create fake values.
"""
from __future__ import annotations

import json
import pytest
from backend.session import session_store
from backend.context_builder import build_session_context, SessionContextBuilder


@pytest.fixture(autouse=True)
def clean_memory():
    """Ensure a clean in-memory state before and after each test."""
    session_store.clear_all_sessions()
    yield
    session_store.clear_all_sessions()


# ==============================================================================
# TEST 1: Empty session produces valid empty context
# ==============================================================================
def test_empty_session_produces_valid_empty_context():
    session_id = "non_existent_session_001"
    query = "What is visible in this area?"

    ctx = build_session_context(session_id=session_id, query=query)

    assert ctx["session_id"] == session_id
    assert ctx["query"] == query
    assert isinstance(ctx["messages"], list)
    assert len(ctx["messages"]) == 0
    assert isinstance(ctx["images"], list)
    assert len(ctx["images"]) == 0
    assert isinstance(ctx["session_info"], dict)
    assert ctx["session_info"]["active_image_ids"] == []
    assert ctx["session_info"]["relevant_analysis_results"] == []
    assert "### CURRENT USER QUERY:" in ctx["text_context"]
    assert "No prior conversation in this session." in ctx["text_context"]
    assert "No images analyzed in this session yet." in ctx["text_context"]


# ==============================================================================
# TEST 2: Conversation history is included
# ==============================================================================
def test_conversation_history_is_included():
    session_id = "session_with_history"
    session_store.add_user_message(session_id, "Can you identify the land cover?")
    session_store.add_assistant_message(session_id, "The scene contains 40% arable land and dense forest.")

    ctx = build_session_context(session_id=session_id, query="What about the buildings?")

    assert len(ctx["messages"]) == 2
    assert ctx["messages"][0]["role"] == "user"
    assert ctx["messages"][0]["content"] == "Can you identify the land cover?"
    assert ctx["messages"][1]["role"] == "assistant"
    assert ctx["messages"][1]["content"] == "The scene contains 40% arable land and dense forest."
    assert "Can you identify the land cover?" in ctx["text_context"]
    assert "The scene contains 40% arable land and dense forest." in ctx["text_context"]


# ==============================================================================
# TEST 3: Image analysis is included
# ==============================================================================
def test_image_analysis_is_included():
    session_id = "session_with_image"
    analysis_data = {"answer": "Arable land (40.4%) identified across the central valley.", "class": "agriculture"}

    session_store.register_image(
        session_id=session_id,
        image_id="img_valley_opt",
        image_path="/data/valley_opt.png",
        filename="valley_opt.png",
        analysis=analysis_data,
        task="T1_VQA",
    )

    ctx = build_session_context(session_id=session_id, query="Describe the vegetation.")

    assert len(ctx["images"]) == 1
    img = ctx["images"][0]
    assert img["image_id"] == "img_valley_opt"
    assert img["filename"] == "valley_opt.png"
    assert img["task"] == "T1_VQA"
    assert img["analysis"] == analysis_data
    assert "Image [img_valley_opt] (valley_opt.png):" in ctx["text_context"]
    assert "Arable land (40.4%)" in ctx["text_context"]


# ==============================================================================
# TEST 4: Multiple images remain distinguishable
# ==============================================================================
def test_multiple_images_remain_distinguishable():
    session_id = "session_multi_img_distinct"

    # Image A: Optical Before
    session_store.register_image(
        session_id=session_id,
        image_id="img_before_2020",
        filename="before_2020.tif",
        analysis={"finding": "Dense forest canopy prior to deforestation."},
        task="T2_Caption",
    )

    # Image B: Optical After
    session_store.register_image(
        session_id=session_id,
        image_id="img_after_2024",
        filename="after_2024.tif",
        analysis={"finding": "Surface clearing and road network visible."},
        task="T4_Change",
    )

    ctx = build_session_context(session_id=session_id, query="Compare this with the previous image.")

    assert len(ctx["images"]) == 2
    img_ids = [im["image_id"] for im in ctx["images"]]
    assert img_ids == ["img_before_2020", "img_after_2024"]

    # Both images must have their own unique entries in the context
    assert ctx["images"][0]["image_id"] == "img_before_2020"
    assert ctx["images"][0]["task"] == "T2_Caption"
    assert ctx["images"][1]["image_id"] == "img_after_2024"
    assert ctx["images"][1]["task"] == "T4_Change"

    # Both image IDs appear in the formatted text context for LLM reference
    assert "Image [img_before_2020]" in ctx["text_context"]
    assert "Image [img_after_2024]" in ctx["text_context"]
    assert "img_before_2020, img_after_2024" in ctx["text_context"]


# ==============================================================================
# TEST 5: Evidence/GIS results are preserved exactly
# ==============================================================================
def test_evidence_and_gis_results_preserved_exactly():
    session_id = "session_exact_metrics"
    evidence = [
        {"evidence_id": "E1", "label": "Arable land", "coverage_pct": 40.42},
        {"evidence_id": "E2", "label": "Industrial buildings", "coverage_pct": 12.85},
    ]
    gis = {
        "area_ha": 154.67,
        "changed_area_ha": 23.45,
        "polygon_count": 14,
        "crs": "EPSG:32632",
    }

    session_store.register_image(
        session_id=session_id,
        image_id="img_precision_01",
        filename="precision.tif",
        analysis={"answer": "Precision land audit completed."},
        task="T4_Change",
        evidence=evidence,
        gis_results=gis,
    )

    ctx = build_session_context(session_id=session_id, query="How much area changed?")

    img_ctx = ctx["images"][0]
    # Exact equality without rounding or modification
    assert img_ctx["evidence"] == evidence
    assert img_ctx["gis_results"] == gis
    assert img_ctx["gis_results"]["area_ha"] == 154.67
    assert img_ctx["gis_results"]["changed_area_ha"] == 23.45
    assert img_ctx["evidence"][0]["coverage_pct"] == 40.42

    # Formatted context reflects exact metrics
    assert "area_ha=154.67" in ctx["text_context"]
    assert "changed_area_ha=23.45" in ctx["text_context"]
    assert "E1: Arable land (~40.42%)" in ctx["text_context"]


# ==============================================================================
# TEST 6: Recent-message limit works
# ==============================================================================
def test_recent_message_limit_works():
    session_id = "session_many_turns"

    # Add 10 turns (5 user, 5 assistant)
    for i in range(1, 6):
        session_store.add_user_message(session_id, f"User question {i}")
        session_store.add_assistant_message(session_id, f"Assistant answer {i}")

    # Build context with limit of 4 recent messages
    ctx = build_session_context(session_id=session_id, query="Follow-up question", max_recent_messages=4)

    assert len(ctx["messages"]) == 4
    # Chronological order of the most recent 4 messages:
    # Turn 4: User question 4, Assistant answer 4
    # Turn 5: User question 5, Assistant answer 5
    assert ctx["messages"][0]["content"] == "User question 4"
    assert ctx["messages"][1]["content"] == "Assistant answer 4"
    assert ctx["messages"][2]["content"] == "User question 5"
    assert ctx["messages"][3]["content"] == "Assistant answer 5"

    # Older messages should not appear in the recent window
    assert "User question 1" not in ctx["text_context"]
    assert "User question 2" not in ctx["text_context"]
    assert "User question 3" not in ctx["text_context"]
    assert "User question 4" in ctx["text_context"]
    assert "User question 5" in ctx["text_context"]


# ==============================================================================
# TEST 7: Different sessions never mix context
# ==============================================================================
def test_different_sessions_never_mix_context():
    session_a = "session_alpha"
    session_b = "session_beta"

    # Populate Alpha
    session_store.add_user_message(session_a, "Alpha confidential observation.")
    session_store.register_image(session_a, image_id="img_alpha", filename="alpha.png", analysis="Alpha analysis")

    # Populate Beta
    session_store.add_user_message(session_b, "Beta private terrain scan.")
    session_store.register_image(session_b, image_id="img_beta", filename="beta.png", analysis="Beta analysis")

    # Build context for Alpha
    ctx_a = build_session_context(session_id=session_a, query="Query for Alpha")
    # Build context for Beta
    ctx_b = build_session_context(session_id=session_b, query="Query for Beta")

    # Alpha assertions
    assert ctx_a["query"] == "Query for Alpha"
    assert len(ctx_a["messages"]) == 1
    assert ctx_a["messages"][0]["content"] == "Alpha confidential observation."
    assert len(ctx_a["images"]) == 1
    assert ctx_a["images"][0]["image_id"] == "img_alpha"
    assert "Beta" not in ctx_a["text_context"]
    assert "img_beta" not in ctx_a["text_context"]

    # Beta assertions
    assert ctx_b["query"] == "Query for Beta"
    assert len(ctx_b["messages"]) == 1
    assert ctx_b["messages"][0]["content"] == "Beta private terrain scan."
    assert len(ctx_b["images"]) == 1
    assert ctx_b["images"][0]["image_id"] == "img_beta"
    assert "Alpha" not in ctx_b["text_context"]
    assert "img_alpha" not in ctx_b["text_context"]


# ==============================================================================
# TEST 8: Current query is included
# ==============================================================================
def test_current_query_is_included():
    session_id = "session_query_check"
    target_query = "Are there any newly formed water bodies near the coastline?"

    ctx = build_session_context(session_id=session_id, query=target_query)

    assert ctx["query"] == target_query
    assert f"### CURRENT USER QUERY:\n{target_query}" in ctx["text_context"]


# ==============================================================================
# TEST 9: No raw image bytes are inserted into text context
# ==============================================================================
def test_no_raw_image_bytes_inserted_into_text_context():
    session_id = "session_bytes_check"
    raw_payload = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR..."

    session_store.register_image(
        session_id=session_id,
        image_id="img_with_bytes",
        image_path="/path/to/raster.tif",
        filename="raster.tif",
        analysis="Valid text findings.",
        metadata={"raw_data": raw_payload, "b64_preview": "iVBORw0KGgoAAAANSU..."},
    )

    ctx = build_session_context(session_id=session_id, query="Check for byte leakage.")

    # 1. Text context must be a valid UTF-8 string with no binary artifacts
    assert isinstance(ctx["text_context"], str)
    assert "b'\\x89PNG" not in ctx["text_context"]
    assert "iVBORw0KGgoAAA" not in ctx["text_context"]

    # 2. Entire context dictionary must be cleanly JSON-serializable
    serialized = json.dumps(ctx)
    assert isinstance(serialized, str)
    assert "raw_data" not in serialized


# ==============================================================================
# TEST 10: Missing analysis/evidence does not create fake values
# ==============================================================================
def test_missing_analysis_evidence_does_not_create_fake_values():
    session_id = "session_sparse_data"

    # Register an image where analysis, evidence, and GIS metrics are completely absent
    session_store.register_image(
        session_id=session_id,
        image_id="img_sparse_01",
        image_path="/data/sparse.tif",
        filename="sparse.tif",
        analysis=None,
        task=None,
        evidence=None,
        gis_results=None,
    )

    ctx = build_session_context(session_id=session_id, query="What is the area percentage?")

    img_ctx = ctx["images"][0]
    # No fake data created:
    assert img_ctx["analysis"] is None
    assert img_ctx["evidence"] == []
    assert img_ctx["gis_results"] == {}

    # Text context must NOT contain hallucinated measurements or percentages
    assert "coverage_pct" not in ctx["text_context"]
    assert "area_ha" not in ctx["text_context"]
    assert "GIS Metrics:" not in ctx["text_context"]
    assert "Verified Evidence:" not in ctx["text_context"]
