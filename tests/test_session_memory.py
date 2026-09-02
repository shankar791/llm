"""
Unit Test Suite for Step 1 — Current-Session Memory Foundation.
Verifies:
1. New session starts empty.
2. Messages are stored correctly (user message, assistant response, timestamp).
3. Image + analysis are stored correctly (unique image_id, image_path, filename, analysis, task, evidence, GIS results).
4. Multiple images can exist in one session.
5. Different sessions have isolated memory.
6. Clearing a session removes its memory only.
7. Memory operations are purely in-memory and keep images separate from messages.
"""
from __future__ import annotations

import pytest
from backend.session import session_store


@pytest.fixture(autouse=True)
def clean_memory_store():
    """Ensure a clean in-memory state before and after each test."""
    session_store.clear_all_sessions()
    yield
    session_store.clear_all_sessions()


# ==============================================================================
# TEST 1: New session starts empty
# ==============================================================================
def test_new_session_starts_empty():
    session_id = "session_test_empty_01"
    sess = session_store.create_or_get_session(session_id)

    assert sess["session_id"] == session_id
    assert isinstance(sess["conversation"], list)
    assert len(sess["conversation"]) == 0
    assert isinstance(sess["messages"], list)
    assert len(sess["messages"]) == 0
    assert isinstance(sess["images"], list)
    assert len(sess["images"]) == 0
    assert isinstance(sess["context"], dict)
    assert sess["context"]["active_image_ids"] == []
    assert sess["context"]["relevant_analysis_results"] == []
    assert "created_at" in sess
    assert "updated_at" in sess


# ==============================================================================
# TEST 2: Messages are stored correctly (user, assistant, timestamp)
# ==============================================================================
def test_messages_are_stored_correctly():
    session_id = "session_test_msg_01"

    # 1. Add user message
    msg_u = session_store.add_user_message(session_id, "What land-cover classes are present in this image?")
    assert msg_u["role"] == "user"
    assert msg_u["content"] == "What land-cover classes are present in this image?"
    assert "timestamp" in msg_u and len(msg_u["timestamp"]) > 0

    # 2. Add assistant response
    msg_a = session_store.add_assistant_message(session_id, "Arable land (40.4%) and forest canopy were identified.")
    assert msg_a["role"] == "assistant"
    assert msg_a["content"] == "Arable land (40.4%) and forest canopy were identified."
    assert "timestamp" in msg_a and len(msg_a["timestamp"]) > 0

    # 3. Retrieve session memory and verify order and completeness
    sess = session_store.get_session_memory(session_id)
    assert sess is not None
    assert len(sess["conversation"]) == 2
    assert sess["conversation"][0]["role"] == "user"
    assert sess["conversation"][0]["content"] == "What land-cover classes are present in this image?"
    assert sess["conversation"][1]["role"] == "assistant"
    assert sess["conversation"][1]["content"] == "Arable land (40.4%) and forest canopy were identified."
    # Both messages must have valid timestamps
    assert sess["conversation"][0]["timestamp"] is not None
    assert sess["conversation"][1]["timestamp"] is not None


# ==============================================================================
# TEST 3: Image + analysis are stored correctly
# ==============================================================================
def test_image_and_analysis_stored_correctly():
    session_id = "session_test_img_01"

    evidence_items = [
        {"evidence_id": "E1", "label": "Arable land", "coverage_pct": 40.4, "source": "spectral_classification"},
        {"evidence_id": "E2", "label": "Industrial buildings", "coverage_pct": 12.8, "source": "structural_detection"},
    ]
    gis_metrics = {
        "area_ha": 40.4,
        "polygon_count": 8,
        "crs": "EPSG:32632",
    }
    analysis_result = {
        "answer": "Dominant land cover is arable land (40.4%) with localized industrial structures.",
        "confidence": 0.88,
    }

    img_entry = session_store.register_image(
        session_id=session_id,
        image_id="img_opt_test_01",
        image_path="/data/rasters/opt_0611.png",
        filename="opt_0611.png",
        analysis=analysis_result,
        task="T1_VQA",
        evidence=evidence_items,
        gis_results=gis_metrics,
    )

    assert img_entry["image_id"] == "img_opt_test_01"
    assert img_entry["image_path"] == "/data/rasters/opt_0611.png"
    assert img_entry["image_ref"] == "/data/rasters/opt_0611.png"
    assert img_entry["filename"] == "opt_0611.png"
    assert img_entry["task"] == "T1_VQA"
    assert img_entry["analysis"] == analysis_result
    assert img_entry["evidence"] == evidence_items
    assert img_entry["gis_results"] == gis_metrics
    assert "timestamp" in img_entry

    # Check session memory state
    sess = session_store.get_session_memory(session_id)
    assert sess is not None
    assert len(sess["images"]) == 1
    assert sess["images"][0]["image_id"] == "img_opt_test_01"
    assert "img_opt_test_01" in sess["context"]["active_image_ids"]
    assert len(sess["context"]["relevant_analysis_results"]) == 1
    assert sess["context"]["relevant_analysis_results"][0] == analysis_result

    # Crucial: Image analysis data is kept separate from conversation messages
    assert len(sess["conversation"]) == 0


# ==============================================================================
# TEST 4: Multiple images can exist in one session
# ==============================================================================
def test_multiple_images_can_exist_in_one_session():
    session_id = "session_test_multi_img"

    # Register Image 1: Optical Before
    img1 = session_store.register_image(
        session_id=session_id,
        image_id="img_optical_t0",
        image_path="/data/opt_t0.tif",
        filename="opt_t0.tif",
        analysis={"answer": "Baseline optical scene before event."},
        task="T2_Caption",
        evidence=[{"evidence_id": "E1", "label": "Vegetation"}],
        gis_results={"area_ha": 120.0},
    )

    # Register Image 2: Optical After
    img2 = session_store.register_image(
        session_id=session_id,
        image_id="img_optical_t1",
        image_path="/data/opt_t1.tif",
        filename="opt_t1.tif",
        analysis={"answer": "Post-event optical capture showing change."},
        task="T4_Change",
        evidence=[{"evidence_id": "E2", "label": "Changed Surface", "changed_area_ha": 18.5}],
        gis_results={"changed_area_ha": 18.5, "total_area_ha": 120.0},
    )

    # Register Image 3: SAR Cross-Modal
    img3 = session_store.register_image(
        session_id=session_id,
        image_id="img_sar_vv",
        image_path="/data/sar_vv.tif",
        filename="sar_vv.tif",
        analysis={"answer": "Co-registered SAR VV backscatter amplitude."},
        task="T5_OpticalSAR",
        evidence=[{"evidence_id": "E3", "label": "Water Surface"}],
        gis_results={"water_fraction": 0.22},
    )

    sess = session_store.get_session_memory(session_id)
    assert sess is not None
    assert len(sess["images"]) == 3
    assert [im["image_id"] for im in sess["images"]] == ["img_optical_t0", "img_optical_t1", "img_sar_vv"]
    assert sess["context"]["active_image_ids"] == ["img_optical_t0", "img_optical_t1", "img_sar_vv"]
    assert len(sess["context"]["relevant_analysis_results"]) == 3

    # Confirm tasks and metadata are preserved individually per image
    assert sess["images"][0]["task"] == "T2_Caption"
    assert sess["images"][1]["task"] == "T4_Change"
    assert sess["images"][2]["task"] == "T5_OpticalSAR"
    assert sess["images"][1]["gis_results"]["changed_area_ha"] == 18.5


# ==============================================================================
# TEST 5: Different sessions have isolated memory
# ==============================================================================
def test_different_sessions_have_isolated_memory():
    sess_id_a = "session_user_alice"
    sess_id_b = "session_user_bob"

    # User Alice interacts with Session A
    session_store.add_user_message(sess_id_a, "Analyze flood area in Region A.")
    session_store.add_assistant_message(sess_id_a, "Region A shows 45 hectares inundated.")
    session_store.register_image(
        sess_id_a,
        image_id="img_alice_01",
        filename="region_a_optical.png",
        analysis={"finding": "Flooding detected"},
        task="T1_VQA",
    )

    # User Bob interacts with Session B
    session_store.add_user_message(sess_id_b, "Inspect urban growth in Region B.")
    session_store.add_assistant_message(sess_id_b, "Region B shows 12 new structures.")
    session_store.register_image(
        sess_id_b,
        image_id="img_bob_01",
        filename="region_b_sar.png",
        analysis={"finding": "Urban expansion detected"},
        task="T3_Ground",
    )

    # Retrieve Alice's session memory
    mem_a = session_store.get_session_memory(sess_id_a)
    assert mem_a is not None
    assert len(mem_a["conversation"]) == 2
    assert mem_a["conversation"][0]["content"] == "Analyze flood area in Region A."
    assert mem_a["conversation"][1]["content"] == "Region A shows 45 hectares inundated."
    assert len(mem_a["images"]) == 1
    assert mem_a["images"][0]["image_id"] == "img_alice_01"
    assert mem_a["context"]["active_image_ids"] == ["img_alice_01"]

    # Verify Alice's memory contains ZERO trace of Bob's data
    for msg in mem_a["conversation"]:
        assert "Region B" not in msg["content"]
        assert "Bob" not in msg["content"]
    assert "img_bob_01" not in mem_a["context"]["active_image_ids"]

    # Retrieve Bob's session memory
    mem_b = session_store.get_session_memory(sess_id_b)
    assert mem_b is not None
    assert len(mem_b["conversation"]) == 2
    assert mem_b["conversation"][0]["content"] == "Inspect urban growth in Region B."
    assert mem_b["conversation"][1]["content"] == "Region B shows 12 new structures."
    assert len(mem_b["images"]) == 1
    assert mem_b["images"][0]["image_id"] == "img_bob_01"
    assert mem_b["context"]["active_image_ids"] == ["img_bob_01"]

    # Verify Bob's memory contains ZERO trace of Alice's data
    for msg in mem_b["conversation"]:
        assert "Region A" not in msg["content"]
        assert "Alice" not in msg["content"]
    assert "img_alice_01" not in mem_b["context"]["active_image_ids"]


# ==============================================================================
# TEST 6: Clearing a session removes its memory
# ==============================================================================
def test_clearing_session_removes_memory():
    sess_id_target = "session_to_clear"
    sess_id_survivor = "session_to_keep"

    # Populate both sessions
    session_store.add_user_message(sess_id_target, "Temporary question.")
    session_store.register_image(sess_id_target, image_id="img_temp", filename="temp.png")

    session_store.add_user_message(sess_id_survivor, "Keep this question.")
    session_store.register_image(sess_id_survivor, image_id="img_keep", filename="keep.png")

    # Clear target session
    cleared = session_store.clear_session(sess_id_target)
    assert cleared is True

    # Verify target session memory is completely gone
    assert session_store.get_session_memory(sess_id_target) is None

    # Verify surviving session is completely intact (isolation on deletion)
    mem_survivor = session_store.get_session_memory(sess_id_survivor)
    assert mem_survivor is not None
    assert len(mem_survivor["conversation"]) == 1
    assert mem_survivor["conversation"][0]["content"] == "Keep this question."
    assert len(mem_survivor["images"]) == 1
    assert mem_survivor["images"][0]["image_id"] == "img_keep"


# ==============================================================================
# TEST 7: In-memory only: no files written to disk
# ==============================================================================
def test_in_memory_only_no_disk_files():
    session_id = "pure_in_memory_session_99"

    # Use in-memory session API
    session_store.create_or_get_session(session_id)
    session_store.add_user_message(session_id, "In memory only message.")
    session_store.register_image(session_id, image_id="img_mem_only", filename="mem.png")

    # Verify data is available in memory
    mem = session_store.get_session_memory(session_id)
    assert mem is not None
    assert len(mem["conversation"]) == 1
    assert len(mem["images"]) == 1

    # Verify no persistent file was written on disk
    expected_path = session_store._session_path(session_id)
    assert not expected_path.exists(), f"File {expected_path} should not exist for in-memory session"
