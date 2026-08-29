"""
Unit tests for SatQuery AI Master Agent LangGraph nodes.
"""
from __future__ import annotations
import pytest
from ai.graph.nodes import (
    validate_inputs_node,
    classify_intent_node,
    compatibility_check_node,
    master_router_node,
    execute_specialist_tool_node,
    standardize_results_node,
    gis_processor_node,
    evidence_confidence_node,
    llm_synthesis_node,
)

BASE_STATE = {
    "session_id": "test-session",
    "query": "What land-cover types are visible?",
    "image_bytes": [b"fakejpegbytes"],
    "image_modalities": ["optical"],
    "image_filenames": ["test.jpg"],
    "tool_results": [],
    "trace": [],
    "error": None,
}


class TestValidateInputsNode:
    def test_single_image_scenario(self):
        result = validate_inputs_node(BASE_STATE)
        assert result["scenario"] == "single_image"
        assert result["n_images"] == 1
        assert len(result["trace"]) == 1

    def test_no_images_sets_error(self):
        state = {**BASE_STATE, "image_bytes": []}
        result = validate_inputs_node(state)
        assert "error" in result
        assert result["error"]
        assert result["is_compatible"] is False

    def test_bi_temporal_pair(self):
        state = {
            **BASE_STATE,
            "image_bytes": [b"t0", b"t1"],
            "image_modalities": ["optical", "optical"],
        }
        result = validate_inputs_node(state)
        assert result["scenario"] == "bi_temporal_pair"

    def test_cross_modal_pair(self):
        state = {
            **BASE_STATE,
            "image_bytes": [b"opt", b"sar"],
            "image_modalities": ["optical", "sar"],
        }
        result = validate_inputs_node(state)
        assert result["scenario"] == "cross_modal_pair"


class TestClassifyIntentNode:
    def test_change_keyword_scores_change(self):
        state = {
            **BASE_STATE,
            "query": "Identify new construction between 2020 and 2024.",
            "n_images": 2,
            "scenario": "bi_temporal_pair",
        }
        result = classify_intent_node(state)
        assert result["intent"] == "change"
        assert "T4_Change" in result["workflow"]

    def test_fusion_cross_modal_pair(self):
        state = {
            **BASE_STATE,
            "query": "Analyze this optical and SAR pair.",
            "n_images": 2,
            "scenario": "cross_modal_pair",
            "image_modalities": ["optical", "sar"],
        }
        result = classify_intent_node(state)
        assert result["intent"] == "fusion"
        assert "T5_OpticalSAR" in result["workflow"]


class TestCompatibilityAndMasterRouterNodes:
    def test_compatibility_passes_for_valid_change(self):
        state = {
            **BASE_STATE,
            "query": "Identify new construction between 2020 and 2024.",
            "n_images": 2,
            "image_modalities": ["optical", "optical"],
        }
        result = compatibility_check_node(state)
        assert result["is_compatible"] is True

    def test_master_router_produces_decision_log(self):
        state = {
            **BASE_STATE,
            "query": "Identify new construction between 2020 and 2024.",
            "workflow": ["T4_Change"],
            "is_compatible": True,
            "intent": "change",
        }
        result = master_router_node(state)
        assert result["selected_tool"] == "T4_Change"
        assert result["decision_log"]["decision"] == "route"
        assert result["decision_log"]["inputs_verified"] is True
        assert result["tool_request"]["tool_id"] == "T4_Change"


class TestSpecialistExecutionAndStandardize:
    def test_execute_specialist_tool_runs_mock(self):
        state = {
            **BASE_STATE,
            "selected_tool": "T1_VQA",
            "query": "What is visible?",
            "image_bytes": [b"img"],
            "image_modalities": ["optical"],
            "tool_results": []
        }
        result = execute_specialist_tool_node(state)
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["tool_id"] == "T1_VQA"
        assert result["tool_results"][0]["metadata"]["mock"] is True

    def test_standardize_results_validates_schema(self):
        state = {
            **BASE_STATE,
            "tool_results": [{
                "tool_id": "T1_VQA",
                "answer": "Land cover summary",
                "confidence": 0.85,
                "evidence": [],
                "metadata": {"mock": True}
            }]
        }
        result = standardize_results_node(state)
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["confidence"] == 0.85


class TestGISAndEvidenceNodes:
    def test_gis_processor_returns_feature_collection(self):
        state = {**BASE_STATE, "tool_results": []}
        result = gis_processor_node(state)
        assert result["geojson"] is None or result["geojson"]["type"] == "FeatureCollection"

    def test_confidence_node_returns_float(self):
        state = {**BASE_STATE, "tool_results": [{"tool_id": "T1_VQA", "confidence": 0.8}]}
        result = evidence_confidence_node(state)
        assert isinstance(result["confidence"], float)
        assert result["confidence"] > 0.0


class TestLLMSynthesisNode:
    def test_synthesizes_grounded_answer(self):
        state = {
            **BASE_STATE,
            "tool_results": [{"tool_id": "T1_VQA", "answer": "Urban area detected", "confidence": 0.85}],
        }
        result = llm_synthesis_node(state)
        assert "Urban area detected" in result["final_answer"]
