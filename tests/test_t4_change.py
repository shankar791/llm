"""
Integration test suite for T4_Change specialist tool and ChangeFormer connection.
Verifies Step 4C requirements.
"""
from __future__ import annotations
import os
import pytest
import numpy as np
from PIL import Image
from tools.change_detection import ChangeDetectionTool
from tools.base import ToolExecutionError
from ai.graph.builder import build_graph


# 1. Real T4_Change execution
def test_1_real_t4_change_execution():
    """Verify T4_Change in real mode executes pretrained ChangeFormer and returns valid ToolResult."""
    tool = ChangeDetectionTool(mode="real")
    with open("backend/real_data/opt_0611.png", "rb") as f0, open("backend/real_data/opt_0810.png", "rb") as f1:
        b0 = f0.read()
        b1 = f1.read()

    res = tool.run(image_bytes_t0=b0, image_bytes_t1=b1, mode="real")
    assert res["tool_id"] == "T4_Change"
    assert "ChangeFormer" in res["answer"]
    assert res["metadata"]["is_mock"] is False
    assert res["metadata"]["mode"] == "real"
    assert res["metadata"]["checkpoint"] == "ChangeFormer_MNCD256.safetensors"
    assert res["metadata"]["changed_pixels"] > 0
    assert 0.0 < res["metadata"]["change_fraction"] < 1.0
    assert "change_mask" in res["metadata"]
    assert isinstance(res["metadata"]["change_mask"], np.ndarray)


# 2. Mock T4_Change execution
def test_2_mock_t4_change_execution():
    """Verify T4_Change in mock mode remains independent and returns deterministic mock."""
    tool = ChangeDetectionTool(mode="mock")
    res = tool.run(image_bytes_t0=b"fake1", image_bytes_t1=b"fake2", mode="mock")
    assert res["tool_id"] == "T4_Change"
    assert res["metadata"]["is_mock"] is True
    assert res["metadata"]["mode"] == "mock"
    assert res["metadata"]["change_fraction"] == 0.072
    assert len(res["evidence"]) == 2


# 3. Metadata Propagation
def test_3_geospatial_metadata_preservation():
    """Verify input geospatial metadata (CRS, transform, dates) is preserved in ToolResult metadata."""
    tool = ChangeDetectionTool(mode="real")
    geo_meta = {
        "crs": "EPSG:4326",
        "transform": [77.5, 0.0001, 0, 12.9, 0, -0.0001],
        "bounds": [77.5, 12.8, 77.6, 12.9],
        "acquisition_dates": {"t0": "2020-01-15", "t1": "2024-03-20"}
    }
    with open("backend/real_data/opt_0611.png", "rb") as f0, open("backend/real_data/opt_0810.png", "rb") as f1:
        b0 = f0.read()
        b1 = f1.read()

    res = tool.run(image_bytes_t0=b0, image_bytes_t1=b1, mode="real", **geo_meta)
    saved_geo = res["metadata"]["geospatial"]
    assert saved_geo["crs"] == "EPSG:4326"
    assert saved_geo["bounds"] == [77.5, 12.8, 77.6, 12.9]
    assert saved_geo["acquisition_dates"]["t0"] == "2020-01-15"


# 4. Invalid Two-Image Input
def test_4_invalid_image_inputs():
    """Verify invalid or corrupted byte payloads raise ToolExecutionError."""
    tool = ChangeDetectionTool(mode="real")
    with pytest.raises(ToolExecutionError):
        tool.run(image_bytes_t0=b"not_a_valid_image", image_bytes_t1=b"also_corrupt", mode="real")


# 5. Missing Temporal Image
def test_5_missing_temporal_image_raises_error():
    """Verify calling real mode without both images raises ToolExecutionError without silent fallback."""
    tool = ChangeDetectionTool(mode="real")
    with pytest.raises(ToolExecutionError, match="requires two images"):
        tool.run(image_bytes_t0=b"single_image", image_bytes_t1=None, mode="real")


# 6. Full LangGraph Execution with Real T4_Change
def test_6_full_langgraph_real_change_execution():
    """Verify Master Agent LangGraph pipeline runs end-to-end with real ChangeFormer tool."""
    graph = build_graph()
    with open("backend/real_data/opt_0611.png", "rb") as f0, open("backend/real_data/opt_0810.png", "rb") as f1:
        b0 = f0.read()
        b1 = f1.read()

    state = {
        "session_id": "lg-real-test",
        "query": "Identify new construction between 2020 and 2024.",
        "image_bytes": [b0, b1],
        "image_modalities": ["optical", "optical"],
        "image_filenames": ["opt_0611.png", "opt_0810.png"],
        "metadata": {"mode": "real"}
    }
    out = graph.invoke(state)
    assert out.get("intent") == "change"
    assert out.get("selected_tool") == "T4_Change"
    assert out.get("is_compatible") is True
    assert len(out.get("tool_results", [])) == 1

    tool_res = out["tool_results"][0]
    assert tool_res["tool_id"] == "T4_Change"
    assert tool_res["metadata"]["is_mock"] is False
    assert tool_res["metadata"]["checkpoint"] == "ChangeFormer_MNCD256.safetensors"
    assert "ChangeFormer" in out.get("final_answer", "")
