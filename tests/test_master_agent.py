"""
Comprehensive Test Suite for SatQuery AI Master Agent Core.
Tests requirements A through M.
"""
from __future__ import annotations
import pytest
from ai.graph.builder import build_graph
from tools.registry import ToolRegistry
from tools.base import BaseTool, ToolExecutionError


# ==============================================================================
# ROUTING TESTS (A - E)
# ==============================================================================

def test_a_vqa_routing():
    """Verify 'What is visible in this image?' routes to T1_VQA."""
    graph = build_graph()
    state = {
        "session_id": "test-vqa",
        "query": "What is visible in this image?",
        "image_bytes": [b"optical_raw"],
        "image_modalities": ["optical"],
        "image_filenames": ["scene1.tif"]
    }
    out = graph.invoke(state)
    assert out.get("intent") == "vqa"
    assert out.get("selected_tool") == "T1_VQA"
    assert out.get("is_compatible") is True
    assert len(out.get("tool_results", [])) == 1
    assert out["tool_results"][0]["tool_id"] == "T1_VQA"
    assert out["tool_results"][0]["metadata"]["mock"] is True


def test_b_caption_routing():
    """Verify 'Describe this image.' routes to T2_Caption."""
    graph = build_graph()
    state = {
        "session_id": "test-caption",
        "query": "Describe this image.",
        "image_bytes": [b"optical_raw"],
        "image_modalities": ["optical"],
        "image_filenames": ["scene1.tif"]
    }
    out = graph.invoke(state)
    assert out.get("intent") == "caption"
    assert out.get("selected_tool") == "T2_Caption"
    assert out.get("is_compatible") is True
    assert len(out.get("tool_results", [])) == 1
    assert out["tool_results"][0]["tool_id"] == "T2_Caption"
    assert out["tool_results"][0]["metadata"]["mock"] is True


def test_c_grounding_routing():
    """Verify 'Where are the buildings?' routes to T3_Ground."""
    graph = build_graph()
    state = {
        "session_id": "test-ground",
        "query": "Where are the buildings?",
        "image_bytes": [b"optical_raw"],
        "image_modalities": ["optical"],
        "image_filenames": ["scene1.tif"]
    }
    out = graph.invoke(state)
    assert out.get("intent") == "ground"
    assert out.get("selected_tool") == "T3_Ground"
    assert out.get("is_compatible") is True
    assert len(out.get("tool_results", [])) == 1
    assert out["tool_results"][0]["tool_id"] == "T3_Ground"
    assert out["tool_results"][0]["metadata"]["mock"] is True


def test_d_change_detection_routing():
    """Verify 'Identify new construction between 2020 and 2024.' routes to T4_Change."""
    graph = build_graph()
    state = {
        "session_id": "test-change",
        "query": "Identify new construction between 2020 and 2024.",
        "image_bytes": [b"raw_2020", b"raw_2024"],
        "image_modalities": ["optical", "optical"],
        "image_filenames": ["t0_2020.tif", "t1_2024.tif"]
    }
    out = graph.invoke(state)
    assert out.get("intent") == "change"
    assert out.get("selected_tool") == "T4_Change"
    assert out.get("is_compatible") is True
    assert len(out.get("tool_results", [])) == 1
    assert out["tool_results"][0]["tool_id"] == "T4_Change"
    assert out["tool_results"][0]["metadata"]["mock"] is True
    assert out.get("geojson") is not None


def test_e_optical_sar_routing():
    """Verify 'Analyze this optical and SAR pair.' routes to T5_OpticalSAR."""
    graph = build_graph()
    state = {
        "session_id": "test-fusion",
        "query": "Analyze this optical and SAR pair.",
        "image_bytes": [b"opt_raw", b"sar_raw"],
        "image_modalities": ["optical", "sar"],
        "image_filenames": ["opt.tif", "sar.tif"]
    }
    out = graph.invoke(state)
    assert out.get("intent") == "fusion"
    assert out.get("selected_tool") == "T5_OpticalSAR"
    assert out.get("is_compatible") is True
    assert len(out.get("tool_results", [])) == 1
    assert out["tool_results"][0]["tool_id"] == "T5_OpticalSAR"
    assert out["tool_results"][0]["metadata"]["mock"] is True


# ==============================================================================
# COMPATIBILITY & EARLY STOP TESTS (F - G)
# ==============================================================================

def test_f_missing_temporal_image():
    """Verify change detection with 1 image triggers compatibility failure and STOPS before routing."""
    graph = build_graph()
    state = {
        "session_id": "test-missing-temporal",
        "query": "Identify new construction between 2020 and 2024.",
        "image_bytes": [b"single_image"],
        "image_modalities": ["optical"],
        "image_filenames": ["t0_2020.tif"]
    }
    out = graph.invoke(state)
    # Compatibility must fail
    assert out.get("is_compatible") is False
    # Specialist tool must NOT have been executed
    assert out.get("selected_tool") is None
    assert len(out.get("tool_results", [])) == 0
    # Trace must not contain execute_specialist_tool node
    nodes_executed = [step.get("node") for step in out.get("trace", [])]
    assert "execute_specialist_tool" not in nodes_executed
    assert "compatibility_check" in nodes_executed
    assert "Incompatible request" in out.get("final_answer", "")


def test_g_missing_sar_modality():
    """Verify optical+SAR query with only 2 optical images fails compatibility and STOPS."""
    graph = build_graph()
    state = {
        "session_id": "test-missing-sar",
        "query": "Analyze this optical and SAR pair.",
        "image_bytes": [b"opt1", b"opt2"],
        "image_modalities": ["optical", "optical"],
        "image_filenames": ["opt1.tif", "opt2.tif"]
    }
    out = graph.invoke(state)
    assert out.get("is_compatible") is False
    assert out.get("selected_tool") is None
    assert len(out.get("tool_results", [])) == 0
    nodes_executed = [step.get("node") for step in out.get("trace", [])]
    assert "execute_specialist_tool" not in nodes_executed
    assert "Incompatible" in out.get("final_answer", "")


# ==============================================================================
# INPUT VALIDATION & ERROR TESTS (H - K)
# ==============================================================================

def test_h_empty_query():
    """Verify empty query triggers validation failure."""
    graph = build_graph()
    state = {
        "session_id": "test-empty-q",
        "query": "   ",
        "image_bytes": [b"opt1"],
        "image_modalities": ["optical"],
        "image_filenames": ["opt1.tif"]
    }
    out = graph.invoke(state)
    assert out.get("is_compatible") is False
    assert "Query string cannot be empty" in out.get("error", "")


def test_i_invalid_data_zero_images():
    """Verify 0 images triggers validation failure."""
    graph = build_graph()
    state = {
        "session_id": "test-zero-img",
        "query": "What is in this image?",
        "image_bytes": [],
        "image_modalities": [],
        "image_filenames": []
    }
    out = graph.invoke(state)
    assert out.get("is_compatible") is False
    assert "No images provided" in out.get("error", "")
    assert out.get("selected_tool") is None


def test_j_unknown_tool_rejection():
    """Verify ToolRegistry rejects unapproved tool IDs."""
    assert ToolRegistry.is_allowed("T1_VQA") is True
    assert ToolRegistry.is_allowed("T4_Change") is True
    assert ToolRegistry.is_allowed("UNAPPROVED_ARBITRARY_TOOL") is False
    with pytest.raises(ValueError, match="Unknown or unapproved tool ID"):
        ToolRegistry.instantiate("UNAPPROVED_ARBITRARY_TOOL")


def test_k_tool_failure_resilience(monkeypatch):
    """Verify unhandled tool exception is trapped and recorded in trace without server crash."""
    graph = build_graph()

    # Temporarily monkeypatch ChangeDetectionTool to simulate a runtime failure
    class FailingTool(BaseTool):
        tool_id = "T4_Change"
        description = "Failing mock"
        def run(self, **kwargs):
            raise ToolExecutionError("Simulated memory allocation failure")

    monkeypatch.setattr("tools.registry.ToolRegistry._REGISTRY", {
        **ToolRegistry._REGISTRY,
        "T4_Change": ToolRegistry.get("T4_Change").__class__(
            tool_id="T4_Change",
            name="Failing Change Tool",
            description="Simulates failure",
            supported_task="change",
            required_modalities=set(),
            min_images=2,
            max_images=2,
            tool_class=FailingTool
        )
    })

    state = {
        "session_id": "test-failing-tool",
        "query": "Identify new construction between 2020 and 2024.",
        "image_bytes": [b"t0", b"t1"],
        "image_modalities": ["optical", "optical"],
        "image_filenames": ["t0.tif", "t1.tif"]
    }
    out = graph.invoke(state)
    # Must capture error without raising unhandled exception
    assert out.get("error") is not None
    assert "Simulated memory allocation failure" in out["error"]
    assert len(out.get("tool_results", [])) == 1
    assert out["tool_results"][0]["metadata"]["status"] == "error"


# ==============================================================================
# FULL GRAPH EXECUTION TESTS (L - M)
# ==============================================================================

def test_l_full_successful_execution_with_decision_log():
    """Verify complete successful pipeline produces verified decision log and trace."""
    graph = build_graph()
    state = {
        "session_id": "test-full-success",
        "query": "Identify new construction between 2020 and 2024.",
        "image_bytes": [b"img_2020", b"img_2024"],
        "image_modalities": ["optical", "optical"],
        "image_filenames": ["img_2020.tif", "img_2024.tif"]
    }
    out = graph.invoke(state)

    # 1. State integrity
    assert out.get("session_id") == "test-full-success"
    assert out.get("selected_tool") == "T4_Change"
    assert out.get("is_compatible") is True
    assert out.get("confidence") > 0.8
    assert "14.25 hectares" in out.get("final_answer", "")

    # 2. Decision Log
    decision_log = out.get("decision_log")
    assert decision_log is not None
    assert decision_log["decision"] == "route"
    assert decision_log["selected_tool"] == "T4_Change"
    assert decision_log["inputs_verified"] is True

    # 3. ToolRequest
    tool_req = out.get("tool_request")
    assert tool_req is not None
    assert tool_req["tool_id"] == "T4_Change"

    # 4. Trace integrity
    trace = out.get("trace", [])
    assert len(trace) >= 8
    node_names = [step["node"] for step in trace]
    expected_order = [
        "validate_inputs",
        "classify_intent",
        "compatibility_check",
        "master_router",
        "execute_specialist_tool",
        "standardize_results",
        "gis_processor",
        "evidence_confidence",
        "llm_synthesis"
    ]
    assert node_names == expected_order


def test_m_full_compatibility_failure_execution():
    """Verify full compatibility failure returns structured explanation without tool execution."""
    graph = build_graph()
    state = {
        "session_id": "test-full-compat-fail",
        "query": "Identify new construction between 2020 and 2024.",
        "image_bytes": [b"only_one_image"],
        "image_modalities": ["optical"],
        "image_filenames": ["t0.tif"]
    }
    out = graph.invoke(state)

    assert out.get("is_compatible") is False
    assert out.get("selected_tool") is None
    assert len(out.get("tool_results", [])) == 0
    assert "Incompatible request" in out.get("final_answer", "")

    trace = out.get("trace", [])
    node_names = [step["node"] for step in trace]
    assert "validate_inputs" in node_names
    assert "classify_intent" in node_names
    assert "compatibility_check" in node_names
    assert "master_router" not in node_names
    assert "execute_specialist_tool" not in node_names
    assert "llm_synthesis" in node_names
