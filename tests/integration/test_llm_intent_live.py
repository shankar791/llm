"""
Live integration test suite for LLMIntentClassifier.
Strictly gated with LLM_INTEGRATION_TEST=true to ensure zero unwanted paid API calls during standard testing.
"""
from __future__ import annotations
import json
import os
import time
import pytest

from ai.intent.classifier import LLMIntentClassifier
from ai.llm.config import LLMConfig
from ai.llm.provider import get_llm_provider


@pytest.mark.skipif(
    os.environ.get("LLM_INTEGRATION_TEST", "false").lower() != "true",
    reason="Live integration test requires LLM_INTEGRATION_TEST=true and valid LLM_API_KEY",
)
def test_live_llm_intent_classifier():
    """Execute live LLM intent classification against configured remote provider."""
    cfg = LLMConfig.from_env()
    assert cfg.api_key is not None, "LLM_API_KEY must be configured for live integration test"

    provider = get_llm_provider(cfg)
    classifier = LLMIntentClassifier(provider=provider)

    query = "Identify new construction between 2020 and 2024."
    start_time = time.perf_counter()
    result = classifier.classify(
        query=query,
        n_images=2,
        modalities=["optical", "optical"],
        timestamps=["2020-03-15", "2024-03-20"],
    )
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    print("\n" + "=" * 60)
    print("LIVE LLM INTENT CLASSIFICATION AUDIT TRACE")
    print("=" * 60)
    print(f"Query:             {query}")
    print(f"Provider:          {cfg.provider}")
    print(f"Model:             {cfg.model}")
    print(f"Latency:           {latency_ms:.2f} ms")
    print(f"Classifier Source: {result.classifier_source}")
    print(f"Fallback Used:     {result.fallback_used}")
    print(f"Task:              {result.primary_task}")
    print(f"Target:            {result.target}")
    print(f"Temporal Scope:    {result.temporal_scope}")
    print(f"Raw LLM JSON:      {json.dumps(result.raw_llm_response, indent=2)}")
    print("=" * 60)

    assert result.primary_task == "change"
    assert result.classifier_source == "llm"
    assert result.fallback_used is False
    assert result.temporal_scope is not None
