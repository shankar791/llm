"""
Pytest stubs for SatQuery AI Pydantic schemas.
"""
from __future__ import annotations
import pytest
from schemas.models import QueryRequest, ToolResult, AgentResponse, EvidenceItem


class TestQueryRequest:
    def test_auto_session_id(self):
        r = QueryRequest(query="test")
        assert r.session_id and len(r.session_id) == 36  # UUID4

    def test_explicit_session_id(self):
        r = QueryRequest(query="test", session_id="my-session")
        assert r.session_id == "my-session"


class TestToolResult:
    def test_valid_result(self):
        r = ToolResult(tool_id="T1_VQA", answer="Urban area", confidence=0.85)
        assert r.confidence == 0.85
        assert r.evidence == []

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            ToolResult(tool_id="T1_VQA", answer="x", confidence=1.5)  # > 1.0


class TestEvidenceItem:
    def test_coverage_pct_bounds(self):
        with pytest.raises(Exception):
            EvidenceItem(tool_id="T3_Ground", label="water", coverage_pct=120.0)  # > 100
