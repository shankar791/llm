"""
Unit and regression test suite for evidence-grounded LLM synthesis (ai.synthesis).
Uses MockLLMProvider and deterministic validators for 100% offline testing.
"""
from __future__ import annotations
import asyncio
import json
import pytest

from ai.llm.errors import LLMAuthenticationError, LLMTimeoutError, LLMResponseError
from ai.llm.provider import MockLLMProvider
from ai.synthesis.fallback import DeterministicFallbackFormatter
from ai.synthesis.llm import LLMSynthesizer
from ai.synthesis.schema import SynthesisPayload, SynthesisClaim, SynthesisResult
from ai.synthesis.validator import SynthesisValidator


# Sample standard fixtures
SAMPLE_CHANGE_TOOL_RESULT = {
    "tool_id": "T4_Change",
    "answer": "Bi-temporal change detection identified significant surface modifications.",
    "confidence": None,
    "confidence_status": "uncalibrated",
    "evidence": [
        {"label": "new_construction", "coverage_pct": 7.0, "bbox_pixels": [10, 10, 80, 80]},
        {"label": "ground_clearing", "coverage_pct": 5.2, "bbox_pixels": [90, 100, 150, 160]},
        {"label": "infrastructure", "coverage_pct": 2.1, "bbox_pixels": [170, 180, 220, 240]},
    ],
    "metadata": {
        "area_ha": 12.4,
        "area_m2": 124000.0,
        "polygon_count": 3,
        "change_fraction_pct": 7.0,
    },
}

SAMPLE_INTENT = {
    "task": "change",
    "target": "building",
    "temporal_scope": {"start_date": "2020", "end_date": "2024"},
    "requires_temporal_pair": True,
}


# ============================================================
# 1. Normal Grounded Synthesis Tests
# ============================================================

def test_normal_grounded_synthesis_success():
    """Test successful grounded synthesis when LLM adheres to evidence."""
    mock_payload = {
        "answer": "Between 2020 and 2024, approximately 12.4 ha of new construction was detected across 3 identified regions. Confidence is uncalibrated.",
        "claims": [
            {"text": "12.4 ha of new construction detected across 3 regions", "evidence_ids": ["E1", "E2", "E3"]},
        ],
        "uncertainties": ["Model confidence is uncalibrated."],
        "justification": "Grounded in T4_Change GIS metrics and 3 evidence polygon clusters.",
    }
    mock_p = MockLLMProvider(default_response=json.dumps(mock_payload))
    synthesizer = LLMSynthesizer(provider=mock_p)

    res = synthesizer.synthesize(
        query="Identify new construction between 2020 and 2024.",
        tool_results=[SAMPLE_CHANGE_TOOL_RESULT],
        confidence=None,
        confidence_status="uncalibrated",
        intent=SAMPLE_INTENT,
    )

    assert res.synthesis_source == "llm"
    assert res.fallback_used is False
    assert "12.4 ha" in res.answer
    assert len(res.claims) == 1
    assert res.claims[0].evidence_ids == ["E1", "E2", "E3"]


def test_async_synthesis_execution():
    """Test asynchronous synthesis pathway."""
    mock_payload = {
        "answer": "Analysis indicates 12.4 ha of construction activity across 3 regions.",
        "claims": [{"text": "12.4 ha detected", "evidence_ids": ["E1"]}],
        "uncertainties": ["Uncalibrated."],
        "justification": "Verified GIS metrics.",
    }
    mock_p = MockLLMProvider(default_response=json.dumps(mock_payload))
    synthesizer = LLMSynthesizer(provider=mock_p)

    res = asyncio.run(
        synthesizer.synthesize_async(
            query="Identify new construction.",
            tool_results=[SAMPLE_CHANGE_TOOL_RESULT],
            intent=SAMPLE_INTENT,
        )
    )
    assert res.synthesis_source == "llm"
    assert res.fallback_used is False


# ============================================================
# 2. Anti-Hallucination & Post-Validator Tests
# ============================================================

def test_hallucination_area_metric_rejection():
    """
    CRITICAL TEST: Evidence specifies area_ha = 12.4.
    Mock LLM hallucinates '21.7 hectares'.
    Post-validator MUST catch the violation and trigger deterministic fallback.
    """
    hallucinated_payload = {
        "answer": "Between 2020 and 2024, approximately 21.7 hectares of massive urban expansion was detected.",
        "claims": [
            {"text": "21.7 hectares of expansion", "evidence_ids": ["E1"]},
        ],
        "uncertainties": [],
        "justification": "Fake inflated measurement.",
    }
    mock_p = MockLLMProvider(default_response=json.dumps(hallucinated_payload))
    synthesizer = LLMSynthesizer(provider=mock_p)

    res = synthesizer.synthesize(
        query="Identify new construction between 2020 and 2024.",
        tool_results=[SAMPLE_CHANGE_TOOL_RESULT],
        confidence=None,
        confidence_status="uncalibrated",
        intent=SAMPLE_INTENT,
    )

    # Post-validator catches 21.7 ha vs 12.4 ha mismatch:
    assert res.synthesis_source == "deterministic_fallback"
    assert res.fallback_used is True
    assert "Anti-hallucination validation failed" in res.fallback_reason
    assert "21.7" in res.fallback_reason
    # Fallback answer uses the TRUE GIS metric:
    assert "12.40 hectares" in res.answer or "12.4" in res.answer


def test_hallucination_fake_evidence_id_rejection():
    """
    Test rejection when LLM references non-existent evidence IDs (e.g. 'E99').
    """
    fake_id_payload = {
        "answer": "Detected new construction in the area.",
        "claims": [
            {"text": "Construction detected", "evidence_ids": ["E99", "E100"]},
        ],
        "uncertainties": [],
        "justification": "Referencing invented IDs.",
    }
    mock_p = MockLLMProvider(default_response=json.dumps(fake_id_payload))
    synthesizer = LLMSynthesizer(provider=mock_p)

    res = synthesizer.synthesize(
        query="What changed?",
        tool_results=[SAMPLE_CHANGE_TOOL_RESULT],
        intent=SAMPLE_INTENT,
    )

    assert res.synthesis_source == "deterministic_fallback"
    assert res.fallback_used is True
    assert "E99" in res.fallback_reason


def test_hallucination_calibrated_confidence_claim_rejection():
    """
    Test rejection when confidence is uncalibrated, but LLM claims '98% confident'.
    """
    overconfident_payload = {
        "answer": "We are 98% confident that 12.4 ha changed.",
        "claims": [{"text": "12.4 ha changed", "evidence_ids": ["E1"]}],
        "uncertainties": [],
        "justification": "Fake probability.",
    }
    mock_p = MockLLMProvider(default_response=json.dumps(overconfident_payload))
    synthesizer = LLMSynthesizer(provider=mock_p)

    res = synthesizer.synthesize(
        query="What changed?",
        tool_results=[SAMPLE_CHANGE_TOOL_RESULT],
        confidence=None,
        confidence_status="uncalibrated",
        intent=SAMPLE_INTENT,
    )

    assert res.synthesis_source == "deterministic_fallback"
    assert res.fallback_used is True
    assert "Claimed calibrated confidence" in res.fallback_reason


# ============================================================
# 3. Provider Failure & Deterministic Fallback Tests
# ============================================================

def test_fallback_on_provider_timeout():
    """Test deterministic fallback when LLM provider times out."""
    def timeout_handler(*args, **kwargs):
        raise LLMTimeoutError("Request timed out after 30s", provider="mock")

    mock_p = MockLLMProvider(custom_handler=timeout_handler)
    synthesizer = LLMSynthesizer(provider=mock_p)

    res = synthesizer.synthesize(
        query="Identify new construction between 2020 and 2024.",
        tool_results=[SAMPLE_CHANGE_TOOL_RESULT],
        confidence=None,
        confidence_status="uncalibrated",
        intent=SAMPLE_INTENT,
    )

    assert res.synthesis_source == "deterministic_fallback"
    assert res.fallback_used is True
    assert "LLMTimeoutError" in res.fallback_reason
    assert "12.40 hectares" in res.answer or "12.4" in res.answer
    assert "Confidence is currently uncalibrated." in res.answer


def test_fallback_on_provider_auth_error():
    """Test deterministic fallback when LLM auth fails."""
    def auth_handler(*args, **kwargs):
        raise LLMAuthenticationError("HTTP 401: Invalid API Key", provider="mock", status_code=401)

    mock_p = MockLLMProvider(custom_handler=auth_handler)
    synthesizer = LLMSynthesizer(provider=mock_p)

    res = synthesizer.synthesize(
        query="Identify new construction between 2020 and 2024.",
        tool_results=[SAMPLE_CHANGE_TOOL_RESULT],
        intent=SAMPLE_INTENT,
    )

    assert res.synthesis_source == "deterministic_fallback"
    assert res.fallback_used is True
    assert "LLMAuthenticationError" in res.fallback_reason


def test_fallback_on_malformed_json():
    """Test deterministic fallback when LLM emits invalid JSON."""
    mock_p = MockLLMProvider(default_response="This is not valid JSON at all.")
    synthesizer = LLMSynthesizer(provider=mock_p)

    res = synthesizer.synthesize(
        query="Identify new construction.",
        tool_results=[SAMPLE_CHANGE_TOOL_RESULT],
        intent=SAMPLE_INTENT,
    )

    assert res.synthesis_source == "deterministic_fallback"
    assert res.fallback_used is True


# ============================================================
# 4. Edge Cases & Multi-Tool Results
# ============================================================

def test_empty_tool_results_handling():
    synthesizer = LLMSynthesizer(provider=MockLLMProvider())
    res = synthesizer.synthesize(
        query="What is in the image?",
        tool_results=[],
        error=None,
    )
    assert res.synthesis_source == "deterministic_fallback"
    assert "No analysis output was generated" in res.answer


def test_pipeline_error_propagation():
    synthesizer = LLMSynthesizer(provider=MockLLMProvider())
    res = synthesizer.synthesize(
        query="What changed?",
        tool_results=[],
        error="Missing temporal image for 2024",
    )
    assert res.synthesis_source == "deterministic_fallback"
    assert "Analysis could not be completed: Missing temporal image for 2024" in res.answer


def test_vqa_tool_result_synthesis():
    """Test synthesis with a standard VQA ToolResult."""
    vqa_result = {
        "tool_id": "T1_VQA",
        "answer": "The image shows an industrial airport with three commercial aircraft parked on the tarmac.",
        "confidence": 0.85,
        "confidence_status": "calibrated",
        "evidence": [
            {"label": "aircraft", "coverage_pct": 4.5, "bbox_pixels": [50, 60, 120, 140]},
        ],
        "metadata": {"aircraft_count": 3},
    }
    mock_payload = {
        "answer": "The scene contains an industrial airport with three commercial aircraft visible on the tarmac.",
        "claims": [{"text": "Three commercial aircraft visible", "evidence_ids": ["E1"]}],
        "uncertainties": [],
        "justification": "Derived from T1_VQA detection.",
    }
    mock_p = MockLLMProvider(default_response=json.dumps(mock_payload))
    synthesizer = LLMSynthesizer(provider=mock_p)

    res = synthesizer.synthesize(
        query="How many aircraft are visible at the airport?",
        tool_results=[vqa_result],
        confidence=0.85,
        confidence_status="calibrated",
        intent={"task": "vqa", "target": "aircraft"},
    )

    assert res.synthesis_source == "llm"
    assert res.fallback_used is False
    assert "aircraft" in res.answer
