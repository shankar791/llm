"""
Unit tests for SatQuery AI specialist tools and ToolRegistry.
"""
from __future__ import annotations
import pytest
from tools.base import BaseTool
from tools.vqa import VQATool
from tools.captioning import CaptioningTool
from tools.grounding import GroundingTool
from tools.change_detection import ChangeDetectionTool
from tools.optical_sar import OpticalSARTool
from tools.registry import ToolRegistry


class TestToolRegistry:
    def test_all_five_tools_registered(self):
        expected_ids = {"T1_VQA", "T2_Caption", "T3_Ground", "T4_Change", "T5_OpticalSAR"}
        registered = {t.tool_id for t in ToolRegistry.list_tools()}
        assert expected_ids.issubset(registered)

    def test_tool_definitions_have_required_metadata(self):
        for tool_def in ToolRegistry.list_tools():
            assert tool_def.tool_id.startswith("T")
            assert tool_def.name
            assert tool_def.description
            assert tool_def.supported_task
            assert issubclass(tool_def.tool_class, BaseTool)


class TestMockSpecialistTools:
    def test_vqa_tool_mock(self):
        tool = VQATool()
        res = tool.run(query="What is visible?", image_bytes=[b"fake"], modalities=["optical"])
        assert res["tool_id"] == "T1_VQA"
        assert res["confidence"] > 0.0
        assert res["metadata"]["mock"] is True

    def test_caption_tool_mock(self):
        tool = CaptioningTool()
        res = tool.run(image_bytes=b"fake", modality="optical")
        assert res["tool_id"] == "T2_Caption"
        assert res["metadata"]["mock"] is True

    def test_grounding_tool_mock(self):
        tool = GroundingTool()
        res = tool.run(query="Locate water bodies", image_bytes=b"fake")
        assert res["tool_id"] == "T3_Ground"
        assert len(res["evidence"]) > 0
        assert res["metadata"]["mock"] is True

    def test_change_detection_tool_mock(self):
        tool = ChangeDetectionTool()
        res = tool.run(image_bytes_t0=b"t0", image_bytes_t1=b"t1")
        assert res["tool_id"] == "T4_Change"
        assert res["metadata"]["mock"] is True
        assert "change_fraction" in res["metadata"]

    def test_optical_sar_tool_mock(self):
        tool = OpticalSARTool()
        res = tool.run(optical_bytes=b"opt", sar_bytes=b"sar")
        assert res["tool_id"] == "T5_OpticalSAR"
        assert res["metadata"]["mock"] is True
        assert "stats_pct" in res["metadata"]
