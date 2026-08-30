"""
Live integration test suite for LLMSynthesizer.
Strictly gated with LLM_INTEGRATION_TEST=true to ensure zero unwanted paid API calls during standard testing.
"""
from __future__ import annotations
import json
import os
import time
import pytest

from ai.llm.config import LLMConfig
from ai.llm.provider import get_llm_provider
from ai.synthesis.llm import LLMSynthesizer


@pytest.mark.skipif(
    os.environ.get("LLM_INTEGRATION_TEST", "false").lower() != "true",
    reason="Live integration test requires LLM_INTEGRATION_TEST=true and valid LLM_API_KEY",
)
def test_live_llm_synthesizer():
    """Execute live LLM grounded synthesis against configured remote provider."""
    cfg = LLMConfig.from_env()
    assert cfg.api_key is not None, "LLM_API_KEY must be configured for live integration test"

    provider = get_llm_provider(cfg)
    synthesizer = LLMSynthesizer(provider=provider)

    sample_tool_result = {
        "tool_id": "T4_Change",
        "answer": "Bi-temporal change detection identified significant surface modifications.",
        "confidence": None,
        "confidence_status": "uncalibrated",
        "evidence": [
            {"label": "new_construction", "coverage_pct": 7.0, "bbox_pixels": [10, 10, 80, 80]},
        ],
        "metadata": {
            "area_ha": 12.4,
            "polygon_count": 3,
            "change_fraction_pct": 7.0,
        },
    }

    query = "Identify new construction between 2020 and 2024."
    start_time = time.perf_counter()
    result = synthesizer.synthesize(
        query=query,
        tool_results=[sample_tool_result],
        confidence=None,
        confidence_status="uncalibrated",
        intent={"task": "change", "target": "construction", "temporal_scope": {"start_date": "2020", "end_date": "2024"}},
    )
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    print("\n" + "=" * 60)
    print("LIVE LLM SYNTHESIS AUDIT TRACE")
    print("=" * 60)
    print(f"Query:            {query}")
    print(f"Provider:         {cfg.provider}")
    print(f"Model:            {cfg.model}")
    print(f"Latency:          {latency_ms:.2f} ms")
    print(f"Synthesis Source: {result.synthesis_source}")
    print(f"Fallback Used:    {result.fallback_used}")
    print(f"Final Answer:     {result.answer}")
    print(f"Claims:           {result.claims}")
    print(f"Uncertainties:    {result.uncertainties}")
    print("=" * 60)

    assert result.synthesis_source == "llm"
    assert result.fallback_used is False
    assert len(result.answer) > 0
