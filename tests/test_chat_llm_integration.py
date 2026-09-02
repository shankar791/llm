"""
Unit Test Suite for Step 3 — Connect Session Context to Existing LLM.
Verifies:
1. Chat query reaches the existing LLM.
2. Current session context is included.
3. Previous conversation is available to the LLM.
4. Previous image analysis is available.
5. Multiple images remain distinguishable.
6. Assistant response is stored in session memory.
7. LLM failure does not create a fake assistant response.
8. Different sessions remain isolated.
9. No raw image bytes are sent as text context.
10. Existing LLM/provider configuration is reused.
"""
from __future__ import annotations

import json
import pytest

from ai.llm.base import LLMResponse
from ai.llm.provider import MockLLMProvider, get_llm_provider
from ai.llm.config import LLMConfig
from backend.session import session_store
from backend.chat import execute_chat_turn, DEFAULT_CHAT_SYSTEM_PROMPT


@pytest.fixture(autouse=True)
def clean_memory():
    """Ensure clean in-memory state before and after each test."""
    session_store.clear_all_sessions()
    yield
    session_store.clear_all_sessions()


# ==============================================================================
# TEST 1: Chat query reaches the existing LLM
# ==============================================================================
def test_chat_query_reaches_existing_llm():
    session_id = "test_chat_reach_01"
    query = "What features are visible in the central sector?"

    mock_provider = MockLLMProvider(default_response="Central sector contains industrial facilities.")
    result = execute_chat_turn(session_id=session_id, query=query, llm_provider=mock_provider)

    # 1. Verify LLM received the call
    assert len(mock_provider.call_history) == 1
    call = mock_provider.call_history[0]
    messages = call["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert query in messages[1]["content"]

    # 2. Verify returned result
    assert result["session_id"] == session_id
    assert result["query"] == query
    assert result["answer"] == "Central sector contains industrial facilities."
    assert result["model"] == "mock-llm-v1"


# ==============================================================================
# TEST 2: Current session context is included
# ==============================================================================
def test_current_session_context_is_included():
    session_id = "test_chat_context_02"
    session_store.create_or_get_session(session_id, metadata={"mission": "Sentinel-2 Land Cover"})

    mock_provider = MockLLMProvider(default_response="Context processed.")
    execute_chat_turn(session_id=session_id, query="Describe the mission context.", llm_provider=mock_provider)

    sent_user_prompt = mock_provider.call_history[0]["messages"][1]["content"]
    assert "### CURRENT USER QUERY:" in sent_user_prompt
    assert "Describe the mission context." in sent_user_prompt
    assert "### SESSION OVERVIEW:" in sent_user_prompt


# ==============================================================================
# TEST 3: Previous conversation is available to the LLM
# ==============================================================================
def test_previous_conversation_is_available_to_llm():
    session_id = "test_chat_prev_convo_03"

    # Turn 1
    session_store.add_user_message(session_id, "Identify all agricultural parcels.")
    session_store.add_assistant_message(session_id, "Detected 3 parcel groups covering 45 hectares of arable crops.")

    # Turn 2: Follow-up question
    followup_query = "What was the total area of those crop parcels?"
    mock_provider = MockLLMProvider(default_response="As noted earlier, the total area is 45 hectares.")

    result = execute_chat_turn(session_id=session_id, query=followup_query, llm_provider=mock_provider)

    sent_user_prompt = mock_provider.call_history[0]["messages"][1]["content"]

    # Turn 1 must be present in the prompt context
    assert "Identify all agricultural parcels." in sent_user_prompt
    assert "Detected 3 parcel groups covering 45 hectares of arable crops." in sent_user_prompt
    assert "What was the total area of those crop parcels?" in sent_user_prompt
    assert result["answer"] == "As noted earlier, the total area is 45 hectares."


# ==============================================================================
# TEST 4: Previous image analysis is available
# ==============================================================================
def test_previous_image_analysis_is_available():
    session_id = "test_chat_image_analysis_04"

    session_store.register_image(
        session_id=session_id,
        image_id="img_harbor_01",
        filename="harbor.png",
        analysis={"finding": "Deepwater maritime terminal with 4 cargo container vessels."},
        task="T1_VQA",
        evidence=[{"evidence_id": "E1", "label": "Vessel", "coverage_pct": 14.2}],
        gis_results={"vessel_count": 4},
    )

    mock_provider = MockLLMProvider(default_response="There were 4 cargo container vessels identified in the harbor.")
    execute_chat_turn(session_id=session_id, query="How many vessels were docked?", llm_provider=mock_provider)

    sent_user_prompt = mock_provider.call_history[0]["messages"][1]["content"]

    assert "img_harbor_01" in sent_user_prompt
    assert "Deepwater maritime terminal with 4 cargo container vessels." in sent_user_prompt
    assert "E1: Vessel (~14.2%)" in sent_user_prompt
    assert "vessel_count=4" in sent_user_prompt


# ==============================================================================
# TEST 5: Multiple images remain distinguishable
# ==============================================================================
def test_multiple_images_remain_distinguishable():
    session_id = "test_chat_multi_image_05"

    session_store.register_image(
        session_id=session_id,
        image_id="img_optical_t0",
        filename="pre_event.tif",
        analysis={"finding": "Dense vegetation canopy before wildfire."},
        task="T2_Caption",
    )

    session_store.register_image(
        session_id=session_id,
        image_id="img_optical_t1",
        filename="post_event.tif",
        analysis={"finding": "Burn scar covering 320 hectares."},
        task="T4_Change",
        gis_results={"burned_area_ha": 320.0},
    )

    mock_provider = MockLLMProvider(default_response="Image img_optical_t0 shows dense vegetation, while img_optical_t1 reveals a 320 ha burn scar.")
    execute_chat_turn(session_id=session_id, query="Compare this with the previous image.", llm_provider=mock_provider)

    sent_user_prompt = mock_provider.call_history[0]["messages"][1]["content"]

    assert "Image [img_optical_t0]" in sent_user_prompt
    assert "Dense vegetation canopy before wildfire." in sent_user_prompt
    assert "Image [img_optical_t1]" in sent_user_prompt
    assert "Burn scar covering 320 hectares." in sent_user_prompt
    assert "burned_area_ha=320.0" in sent_user_prompt


# ==============================================================================
# TEST 6: Assistant response is stored in session memory
# ==============================================================================
def test_assistant_response_is_stored_in_session_memory():
    session_id = "test_chat_storage_06"
    mock_response = "Forest canopy coverage has stabilized at 62%."

    mock_provider = MockLLMProvider(default_response=mock_response)
    execute_chat_turn(session_id=session_id, query="What is the forest status?", llm_provider=mock_provider)

    sess = session_store.get_session_memory(session_id)
    assert sess is not None
    assert len(sess["conversation"]) == 2

    # User message
    assert sess["conversation"][0]["role"] == "user"
    assert sess["conversation"][0]["content"] == "What is the forest status?"
    assert "timestamp" in sess["conversation"][0]

    # Assistant response
    assert sess["conversation"][1]["role"] == "assistant"
    assert sess["conversation"][1]["content"] == mock_response
    assert "timestamp" in sess["conversation"][1]


# ==============================================================================
# TEST 7: LLM failure does not create a fake assistant response
# ==============================================================================
def test_llm_failure_does_not_create_fake_assistant_response():
    session_id = "test_chat_fail_07"

    class FailingLLMProvider(MockLLMProvider):
        def generate_sync(self, *args, **kwargs):
            raise ConnectionError("TokenRouter endpoint unavailable (503 Service Unavailable)")

    failing_provider = FailingLLMProvider()

    # The chat turn must raise the error
    with pytest.raises(ConnectionError) as exc_info:
        execute_chat_turn(session_id=session_id, query="Analyze this area.", llm_provider=failing_provider)

    assert "503 Service Unavailable" in str(exc_info.value)

    # Session memory check:
    sess = session_store.get_session_memory(session_id)
    assert sess is not None

    # User message IS preserved:
    assert len(sess["conversation"]) == 1
    assert sess["conversation"][0]["role"] == "user"
    assert sess["conversation"][0]["content"] == "Analyze this area."

    # Assistant response is NOT fabricated:
    assert not any(msg["role"] == "assistant" for msg in sess["conversation"])


# ==============================================================================
# TEST 8: Different sessions remain isolated
# ==============================================================================
def test_different_sessions_remain_isolated():
    session_alpha = "session_alpha_secure"
    session_beta = "session_beta_secure"

    provider_alpha = MockLLMProvider(default_response="Alpha response verified.")
    provider_beta = MockLLMProvider(default_response="Beta response verified.")

    execute_chat_turn(session_id=session_alpha, query="Query for Alpha.", llm_provider=provider_alpha)
    execute_chat_turn(session_id=session_beta, query="Query for Beta.", llm_provider=provider_beta)

    # Alpha prompt must not contain Beta data
    alpha_prompt = provider_alpha.call_history[0]["messages"][1]["content"]
    assert "Query for Alpha." in alpha_prompt
    assert "Beta" not in alpha_prompt

    # Beta prompt must not contain Alpha data
    beta_prompt = provider_beta.call_history[0]["messages"][1]["content"]
    assert "Query for Beta." in beta_prompt
    assert "Alpha" not in beta_prompt

    # Storage isolation
    sess_a = session_store.get_session_memory(session_alpha)
    sess_b = session_store.get_session_memory(session_beta)
    assert sess_a["conversation"][0]["content"] == "Query for Alpha."
    assert sess_b["conversation"][0]["content"] == "Query for Beta."


# ==============================================================================
# TEST 9: No raw image bytes are sent as text context
# ==============================================================================
def test_no_raw_image_bytes_sent_as_text_context():
    session_id = "test_chat_no_bytes_09"

    session_store.register_image(
        session_id=session_id,
        image_id="img_with_bytes",
        image_path="/data/binary.tif",
        filename="binary.tif",
        analysis="Clean terrain observation.",
        metadata={"raw_payload": b"\x89PNG\r\n\x1a\n\x00\x00\x00", "b64": "data:image/png;base64,..."},
    )

    mock_provider = MockLLMProvider(default_response="No bytes received.")
    execute_chat_turn(session_id=session_id, query="Check bytes", llm_provider=mock_provider)

    sent_prompt = mock_provider.call_history[0]["messages"][1]["content"]

    # Verify no binary representations leaked into the text prompt
    assert "b'\\x89PNG" not in sent_prompt
    assert "data:image/png;base64" not in sent_prompt

    # Verify message payload is JSON-serializable
    assert json.dumps(mock_provider.call_history[0]["messages"])


# ==============================================================================
# TEST 10: Existing LLM/provider configuration is reused
# ==============================================================================
def test_existing_llm_provider_configuration_is_reused():
    session_id = "test_chat_cfg_reuse_10"

    # Use LLMConfig with mock provider to test that get_llm_provider() factory is reused seamlessly
    cfg = LLMConfig(provider="mock")
    provider = get_llm_provider(config=cfg)

    result = execute_chat_turn(session_id=session_id, query="Factory reuse query", llm_provider=provider)

    assert result["provider"] == "mock"
    assert result["query"] == "Factory reuse query"
    assert "answer" in result
    assert result["session_id"] == session_id
