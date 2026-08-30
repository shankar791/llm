"""
Unit, edge-case, and benchmark test suite for LLMIntentClassifier (STEP 8B).
Uses MockLLMProvider for 100% offline, deterministic execution.
"""
from __future__ import annotations
import json
import pytest

from ai.intent.classifier import LLMIntentClassifier, RuleBasedIntentClassifier
from ai.intent.schema import LLMIntentPayload, IntentResult
from ai.llm.base import LLMResponse
from ai.llm.errors import LLMAuthenticationError, LLMTimeoutError, LLMResponseError
from ai.llm.provider import MockLLMProvider
from ai.compatibility.router import ToolCompatibilityRouter


# ============================================================
# 1. Standard Task Classification Tests
# ============================================================

def test_vqa_classification_success():
    mock_payload = {
        "task": "vqa",
        "target": "solar_panel",
        "modality": "optical",
        "spatial_scope": "entire_scene",
        "requires_temporal_pair": False,
        "requires_cross_modal_pair": False,
        "ambiguous": False,
        "clarification_needed": False,
        "reasoning": "User is asking about the presence and count of solar panels in the image.",
    }
    mock_provider = MockLLMProvider(default_response=json.dumps(mock_payload))
    classifier = LLMIntentClassifier(provider=mock_provider)

    result = classifier.classify("Are there any solar panels installed on these roofs?")
    assert result.primary_task == "vqa"
    assert result.target == "solar_panel"
    assert result.workflow == ["T1_VQA"]
    assert result.classifier_source == "llm"
    assert result.fallback_used is False
    assert result.confidence is None
    assert result.confidence_status == "uncalibrated"


def test_caption_classification_success():
    mock_payload = {
        "task": "caption",
        "target": "harbor",
        "modality": "optical",
        "requires_temporal_pair": False,
        "requires_cross_modal_pair": False,
        "ambiguous": False,
        "clarification_needed": False,
        "reasoning": "User requested an overview and summary description of the port.",
    }
    mock_provider = MockLLMProvider(default_response=json.dumps(mock_payload))
    classifier = LLMIntentClassifier(provider=mock_provider)

    result = classifier.classify("Describe this coastal harbor scene in detail.")
    assert result.primary_task == "caption"
    assert result.target == "harbor"
    assert result.workflow == ["T2_Caption"]
    assert result.classifier_source == "llm"
    assert result.fallback_used is False


def test_grounding_classification_success():
    mock_payload = {
        "task": "ground",
        "target": "storage_tank",
        "modality": "optical",
        "requires_temporal_pair": False,
        "requires_cross_modal_pair": False,
        "ambiguous": False,
        "clarification_needed": False,
        "reasoning": "User wants to locate and pinpoint circular oil storage tanks.",
    }
    mock_provider = MockLLMProvider(default_response=json.dumps(mock_payload))
    classifier = LLMIntentClassifier(provider=mock_provider)

    result = classifier.classify("Locate and mark all oil storage tanks in the refinery.")
    assert result.primary_task == "ground"
    assert result.target == "storage_tank"
    assert result.workflow == ["T3_Ground"]
    assert result.classifier_source == "llm"
    assert result.fallback_used is False


def test_change_detection_classification_success():
    mock_payload = {
        "task": "change",
        "target": "building",
        "modality": "optical",
        "temporal_scope": {"start_date": "2020", "end_date": "2024"},
        "requires_temporal_pair": True,
        "requires_cross_modal_pair": False,
        "ambiguous": False,
        "clarification_needed": False,
        "reasoning": "User is asking for new construction comparison between 2020 and 2024.",
    }
    mock_provider = MockLLMProvider(default_response=json.dumps(mock_payload))
    classifier = LLMIntentClassifier(provider=mock_provider)

    result = classifier.classify("Identify new construction between 2020 and 2024.", n_images=2)
    assert result.primary_task == "change"
    assert result.target == "building"
    assert result.requires_temporal_pair is True
    assert result.temporal_scope == {"start_date": "2020", "end_date": "2024"}
    assert result.workflow == ["T4_Change"]
    assert result.classifier_source == "llm"


def test_fusion_classification_success():
    mock_payload = {
        "task": "fusion",
        "target": "coastal_flood",
        "modality": "multimodal",
        "requires_temporal_pair": False,
        "requires_cross_modal_pair": True,
        "ambiguous": False,
        "clarification_needed": False,
        "reasoning": "User asked to combine optical and SAR radar data to evaluate flooding.",
    }
    mock_provider = MockLLMProvider(default_response=json.dumps(mock_payload))
    classifier = LLMIntentClassifier(provider=mock_provider)

    result = classifier.classify("Use optical and SAR imagery to map flooded areas.", n_images=2, modalities=["optical", "sar"])
    assert result.primary_task == "fusion"
    assert result.requires_cross_modal_pair is True
    assert result.workflow == ["T5_OpticalSAR"]
    assert result.classifier_source == "llm"


# ============================================================
# 2. Ambiguity & Clarification Tests
# ============================================================

def test_ambiguous_query_representation():
    mock_payload = {
        "task": "vqa",
        "target": None,
        "modality": "optical",
        "requires_temporal_pair": False,
        "requires_cross_modal_pair": False,
        "ambiguous": True,
        "clarification_needed": True,
        "reasoning": "The query 'tell me stuff' is underspecified and lacks actionable remote-sensing intent.",
    }
    mock_provider = MockLLMProvider(default_response=json.dumps(mock_payload))
    classifier = LLMIntentClassifier(provider=mock_provider)

    result = classifier.classify("tell me stuff")
    assert result.ambiguous is True
    assert result.clarification_needed is True
    assert result.classifier_source == "llm"


# ============================================================
# 3. Explicit Fallback Tests (Zero Silent Failures)
# ============================================================

def test_fallback_on_malformed_json():
    mock_provider = MockLLMProvider(default_response="This is raw prose, not JSON.")
    classifier = LLMIntentClassifier(provider=mock_provider)

    result = classifier.classify("What is the dominant land cover in this scene?")
    assert result.classifier_source == "rule_fallback"
    assert result.fallback_used is True
    assert "JSONDecodeError" in result.fallback_reason or "ValueError" in result.fallback_reason
    assert result.primary_task == "vqa"


def test_fallback_on_invalid_task_enum():
    mock_payload = {
        "task": "unsupported_arbitrary_task_xyz",
        "target": "roads",
    }
    mock_provider = MockLLMProvider(default_response=json.dumps(mock_payload))
    classifier = LLMIntentClassifier(provider=mock_provider)

    result = classifier.classify("Highlight the roads in the image.")
    assert result.classifier_source == "rule_fallback"
    assert result.fallback_used is True
    assert "ValidationError" in result.fallback_reason
    assert result.primary_task == "ground"


def test_fallback_on_llm_timeout():
    def timeout_handler(*args, **kwargs):
        raise LLMTimeoutError("LLM call timed out after 30s", provider="mock")

    mock_provider = MockLLMProvider(custom_handler=timeout_handler)
    classifier = LLMIntentClassifier(provider=mock_provider)

    result = classifier.classify("Describe this satellite scene.")
    assert result.classifier_source == "rule_fallback"
    assert result.fallback_used is True
    assert "LLMTimeoutError" in result.fallback_reason
    assert result.primary_task == "caption"


def test_fallback_on_llm_auth_failure():
    def auth_handler(*args, **kwargs):
        raise LLMAuthenticationError("HTTP 401: Invalid API key", provider="mock", status_code=401)

    mock_provider = MockLLMProvider(custom_handler=auth_handler)
    classifier = LLMIntentClassifier(provider=mock_provider)

    result = classifier.classify("Identify new construction between 2020 and 2024.", n_images=2)
    assert result.classifier_source == "rule_fallback"
    assert result.fallback_used is True
    assert "LLMAuthenticationError" in result.fallback_reason
    assert result.primary_task == "change"


# ============================================================
# 4. Proving LLM Output Does NOT Control Tool Selection
# ============================================================

def test_llm_output_does_not_control_tool_selection():
    """
    Architectural Proof: Even if LLM returns change intent, if the user only provides 1 image,
    the deterministic Compatibility Gate rejects execution and prevents invalid model execution.
    """
    mock_payload = {
        "task": "change",
        "target": "urban",
        "requires_temporal_pair": True,
        "reasoning": "User requested change detection.",
    }
    mock_provider = MockLLMProvider(default_response=json.dumps(mock_payload))
    classifier = LLMIntentClassifier(provider=mock_provider)

    # 1 image provided for a change detection task
    intent_result = classifier.classify("What changed in this area?", n_images=1)
    assert intent_result.primary_task == "change"

    # Pass through deterministic Compatibility Gate
    router = ToolCompatibilityRouter()
    compat_result = router.check_compatibility(intent_result, n_images=1, modalities=["optical"])

    # Proves compatibility gate strictly overrides and blocks execution:
    assert compat_result.compatible is False
    assert len(compat_result.validated_tool_ids) == 0
    assert "requires 2 images" in compat_result.explanation


# ============================================================
# 5. 20-Query Offline Benchmark Suite
# ============================================================

OFFLINE_TEST_BENCHMARK = [
    # 1. Obvious VQA
    ("What is the primary land use type visible here?", "vqa", "land_use", False, False),
    ("How many aircraft are parked on the runway apron?", "vqa", "aircraft", False, False),
    ("Is there standing water visible in these agricultural fields?", "vqa", "water", False, False),
    ("What type of crop is planted in the southern sector?", "vqa", "crop", False, False),

    # 2. Indirect / Contextual VQA
    ("Estimate the density of residential buildings in this area.", "vqa", "residential_building", False, False),
    ("Assess whether the vegetation appears stressed or healthy.", "vqa", "vegetation", False, False),

    # 3. Captioning
    ("Describe this satellite scene in a comprehensive overview.", "caption", None, False, False),
    ("Provide a summary caption for this urban satellite image.", "caption", "urban", False, False),
    ("Summarize the landscape features shown in this image.", "caption", "landscape", False, False),

    # 4. Grounding & Localization
    ("Highlight all container ships docked in the port.", "ground", "ship", False, False),
    ("Locate the solar farm near the western highway.", "ground", "solar_farm", False, False),
    ("Find and mark the building footprints in this sector.", "ground", "building", False, False),
    ("Where is the boundary between the forest and cropland?", "ground", "forest_cropland_boundary", False, False),

    # 5. Temporal Change Detection
    ("Identify new construction between 2020 and 2024.", "change", "construction", True, False),
    ("What changed in the forest canopy between the before and after dates?", "change", "forest", True, False),
    ("Detect urban expansion across these two temporal acquisitions.", "change", "urban_expansion", True, False),
    ("Compare the shoreline between 2018 and 2023.", "change", "shoreline", True, False),

    # 6. Optical + SAR Cross-Modal Fusion
    ("Use optical and SAR imagery to evaluate soil moisture.", "fusion", "soil_moisture", False, True),
    ("Perform optical and radar fusion to detect vessels through cloud cover.", "fusion", "vessel", False, True),

    # 7. Ambiguous / Underspecified
    ("Analyze this data.", "vqa", None, False, False),
]


@pytest.mark.parametrize("query,expected_task,expected_target,req_temp,req_fusion", OFFLINE_TEST_BENCHMARK)
def test_offline_20_query_benchmark(query, expected_task, expected_target, req_temp, req_fusion):
    """Test 20 diverse remote sensing queries through LLMIntentClassifier."""
    mock_payload = {
        "task": expected_task,
        "target": expected_target,
        "modality": "multimodal" if req_fusion else "optical",
        "requires_temporal_pair": req_temp,
        "requires_cross_modal_pair": req_fusion,
        "ambiguous": (query == "Analyze this data."),
        "clarification_needed": (query == "Analyze this data."),
        "reasoning": f"Benchmark classification for '{query}'.",
    }
    mock_provider = MockLLMProvider(default_response=json.dumps(mock_payload))
    classifier = LLMIntentClassifier(provider=mock_provider)

    result = classifier.classify(query, n_images=2 if (req_temp or req_fusion) else 1)
    assert result.primary_task == expected_task
    assert result.target == expected_target
    assert result.requires_temporal_pair == req_temp
    assert result.requires_cross_modal_pair == req_fusion
    assert result.classifier_source == "llm"
    assert result.fallback_used is False

    # Check Pydantic serialization roundtrip
    schema = result.to_schema()
    assert schema.task == expected_task
    assert schema.requires_temporal_pair == req_temp
