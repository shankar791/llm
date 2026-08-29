"""
Phase 1 Foundation Test Suite — Validates Schemas, AI Foundation, and Critical Execution Flows.
"""
from __future__ import annotations
import pytest
from schemas.models import (
    RasterReference,
    QueryRequest,
    IntentSchema,
    CompatibilityResult,
    ToolRequest,
    EvidenceItem,
    ToolResult,
    AgentResponse,
)
from ai.intent.classifier import RuleBasedIntentClassifier
from ai.compatibility.router import ToolCompatibilityRouter
from ai.graph.builder import build_graph
from tools import TOOL_REGISTRY


def test_schema_instantiation():
    """Verify all canonical Pydantic schemas instantiate and validate correctly."""
    ref = RasterReference(filename="img1.tif", modality="optical", crs="EPSG:4326")
    assert ref.filename == "img1.tif"

    query_req = QueryRequest(query="Find new construction", rasters=[ref])
    assert query_req.query == "Find new construction"
    assert len(query_req.rasters) == 1

    intent = IntentSchema(task="change", workflow=["T4_Change"], confidence=0.9)
    assert intent.task == "change"

    compat = CompatibilityResult(compatible=True, validated_tool_ids=["T4_Change"])
    assert compat.compatible is True

    tool_req = ToolRequest(tool_id="T4_Change", query=query_req.query, rasters=[ref])
    assert tool_req.tool_id == "T4_Change"

    evidence = EvidenceItem(tool_id="T4_Change", label="new_building", coverage_pct=5.5)
    assert evidence.coverage_pct == 5.5

    tool_res = ToolResult(tool_id="T4_Change", answer="5.5% change detected", confidence=0.88, evidence=[evidence])
    assert tool_res.confidence == 0.88

    response = AgentResponse(
        session_id=query_req.session_id,
        query=query_req.query,
        final_answer="Detected 5.5% change.",
        confidence=0.88,
        tool_results=[tool_res],
        trace_id="tr-test-01",
        elapsed_ms=120,
    )
    assert response.confidence == 0.88
    assert len(response.tool_results) == 1


def test_intent_and_compatibility_pipeline():
    """Verify Query -> Intent -> Compatibility check flow."""
    classifier = RuleBasedIntentClassifier()
    router = ToolCompatibilityRouter()

    # 1. Compatible Change Detection
    intent = classifier.classify("Identify new construction between 2020 and 2024", n_images=2, modalities=["optical", "optical"])
    assert intent.primary_task == "change"
    assert "T4_Change" in intent.workflow

    compat = router.check_compatibility(intent, n_images=2, modalities=["optical", "optical"])
    assert compat.compatible is True
    assert compat.validated_tool_ids == ["T4_Change"]

    # 2. Incompatible Change Detection (1 image only)
    compat_fail = router.check_compatibility(intent, n_images=1, modalities=["optical"])
    assert compat_fail.compatible is False
    assert len(compat_fail.missing_requirements) > 0


def test_langgraph_master_agent_execution():
    """Verify Master Agent execution graph over all task pathways."""
    graph = build_graph()

    # Flow A: Change Detection
    state_change = {
        "session_id": "test-s1",
        "query": "Identify new construction between 2020 and 2024",
        "image_bytes": [b"raw_bytes_t0", b"raw_bytes_t1"],
        "image_modalities": ["optical", "optical"],
        "image_filenames": ["t0.tif", "t1.tif"],
    }
    out_change = graph.invoke(state_change)
    assert out_change.get("intent") == "change"
    assert out_change.get("is_compatible") is True
    assert len(out_change.get("tool_results", [])) == 1
    assert out_change["tool_results"][0]["tool_id"] == "T4_Change"
    assert out_change.get("confidence", 0) > 0.8

    # Flow B: Early exit on Incompatible Data
    state_incompat = {
        "session_id": "test-s2",
        "query": "Compare forest change between 2020 and 2024",
        "image_bytes": [b"single_img"],
        "image_modalities": ["optical"],
        "image_filenames": ["t0.tif"],
    }
    out_incompat = graph.invoke(state_incompat)
    assert out_incompat.get("is_compatible") is False
    assert "Incompatible" in out_incompat.get("final_answer", "")

    # Flow C: Single Image Grounding
    state_ground = {
        "session_id": "test-s3",
        "query": "Where are the storage tanks in this scene?",
        "image_bytes": [b"port_img"],
        "image_modalities": ["optical"],
        "image_filenames": ["port.tif"],
    }
    out_ground = graph.invoke(state_ground)
    assert out_ground.get("intent") == "ground"
    assert out_ground.get("is_compatible") is True
    assert len(out_ground.get("tool_results", [])) == 1
    assert out_ground["tool_results"][0]["tool_id"] == "T3_Ground"


if __name__ == "__main__":
    test_schema_instantiation()
    test_intent_and_compatibility_pipeline()
    test_langgraph_master_agent_execution()
    print("All Phase 1 tests passed successfully!")
