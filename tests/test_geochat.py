"""
Standalone Unit & Integration Test Suite for GeoChat VLM Adapter (T1_VQA, T2_Caption, T3_Ground).
Verifies unified model execution, robust coordinate parsing, explicit error handling, and canonical ToolResult compliance.
"""
from __future__ import annotations
import os
import pytest
import numpy as np
from PIL import Image
import torch
from models.geochat.adapter import GeoChatAdapter, CoordinateParser
from tools.vqa import VQATool
from tools.captioning import CaptioningTool
from tools.grounding import GroundingTool
from tools.base import ToolExecutionError


# 1. Coordinate Parser Valid Extraction
def test_1_coordinate_parser_valid_extraction():
    text = "Detected storage tanks at [150, 200, 320, 380] and [160, 450, 340, 620]."
    img_size = (1000, 800)  # (width=1000, height=800)

    norm_boxes, pixel_boxes, warnings = CoordinateParser.extract_and_convert(text, img_size)
    assert len(norm_boxes) == 2
    assert len(pixel_boxes) == 2
    assert len(warnings) == 0

    assert norm_boxes[0] == [150, 200, 320, 380]
    assert pixel_boxes[0] == [120, 200, 256, 380]


# 2. Coordinate Parser Rejection of Out-of-Bounds & Inverted Coordinates
def test_2_coordinate_parser_rejection_of_malformed_boxes():
    text = (
        "Valid box [100, 100, 400, 400], "
        "Out-of-bounds [1200, 200, 1500, 400], "
        "Inverted y [500, 200, 300, 400], "
        "Inverted x [200, 600, 400, 300]."
    )
    norm_boxes, pixel_boxes, warnings = CoordinateParser.extract_and_convert(text, (500, 500))
    assert len(norm_boxes) == 1
    assert norm_boxes[0] == [100, 100, 400, 400]
    assert len(warnings) == 3


# 3. Unified GeoChatAdapter Mock Loading & Metadata
def test_3_unified_geochat_adapter_mock_loading():
    adapter = GeoChatAdapter(mode="mock")
    info = adapter.load()
    assert info["model"] == "GeoChat"
    assert info["mode"] == "mock"
    assert info["is_mock"] is True
    assert info["mock"] is True
    assert info["device"] in {"cpu", "cuda"}
    assert info["load_time_ms"] >= 0


# 4. GeoChatAdapter Explicit Error When Real Checkpoint Missing
def test_4_real_mode_raises_explicit_error_without_fallback():
    # Attempting to load an invalid checkpoint in real mode must raise RuntimeError
    adapter = GeoChatAdapter(checkpoint_path="/nonexistent/geochat/weights", mode="real")
    with pytest.raises(RuntimeError, match="Failed to load real GeoChat model"):
        adapter.load(mode="real")


# 5. VQATool Mock Execution & Schema Integrity
def test_5_vqa_tool_mock_execution():
    tool = VQATool(mode="mock")
    opt_path = "backend/real_data/opt_0611.png"
    assert os.path.exists(opt_path)

    with open(opt_path, "rb") as f:
        img_bytes = f.read()

    res_mock = tool.run(query="What land cover dominates?", image_bytes=img_bytes, mode="mock")
    assert res_mock["tool_id"] == "T1_VQA"
    assert res_mock["metadata"]["is_mock"] is True
    assert "cropland" in res_mock["answer"].lower()
    assert res_mock["confidence"] is not None


# 6. CaptioningTool Mock Execution
def test_6_captioning_tool_mock_execution():
    tool = CaptioningTool(mode="mock")
    opt_path = "backend/real_data/opt_0810.png"
    with open(opt_path, "rb") as f:
        img_bytes = f.read()

    res = tool.run(image_bytes=img_bytes, mode="mock")
    assert res["tool_id"] == "T2_Caption"
    assert res["confidence"] is not None
    assert len(res["evidence"]) == 1
    assert res["metadata"]["is_mock"] is True


# 7. GroundingTool Mock Execution & Bounding Box Evidence
def test_7_grounding_tool_mock_execution():
    tool = GroundingTool(mode="mock")
    img_arr = np.zeros((600, 800, 3), dtype=np.uint8)

    res = tool.run(query="Locate buildings in the scene", image_bytes=img_arr, mode="mock")
    assert res["tool_id"] == "T3_Ground"
    assert res["confidence"] is not None
    assert len(res["evidence"]) >= 2

    # Verify bounding boxes are within pixel space (height=600, width=800)
    for ev in res["evidence"]:
        bbox = ev["bbox_pixels"]
        assert bbox is not None
        assert 0 <= bbox[0] < bbox[2] <= 600
        assert 0 <= bbox[1] < bbox[3] <= 800
        assert ev["coverage_pct"] > 0


# 8. Error Handling on Missing Inputs
def test_8_missing_inputs_error_handling():
    vqa = VQATool()
    with pytest.raises(ToolExecutionError, match="requires a valid input image"):
        vqa.run(query="What is here?", image_bytes=None)

    with pytest.raises(ToolExecutionError, match="requires a non-empty query"):
        vqa.run(query="", image_bytes=b"dummy")
