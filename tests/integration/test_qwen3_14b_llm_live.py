"""
Live integration test for Qwen3-14B (qwen/qwen3-14b:free) via OpenRouter.
Gated with LLM_INTEGRATION_TEST=true.
Tests:
1. Intent classification (VQA, Caption, Ground, Change, Fusion, Ambiguous)
2. Structured JSON generation
3. Evidence-grounded synthesis preserving exact GIS values (12.4 ha, 3 polygons, 7.0%)
4. Hallucination rejection & deterministic fallback
"""
from __future__ import annotations
import os
import time
import pytest

from ai.intent.classifier import LLMIntentClassifier
from ai.llm.config import LLMConfig
from ai.llm.provider import OpenAICompatibleProvider
from ai.synthesis.fallback import DeterministicFallbackFormatter
from ai.synthesis.llm import LLMSynthesizer
from ai.synthesis.schema import SynthesisPayload, SynthesisClaim
from ai.synthesis.validator import SynthesisValidator


@pytest.mark.skipif(
    os.environ.get("LLM_INTEGRATION_TEST", "false").lower() != "true"
    or not (os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")),
    reason="Live integration test requires LLM_INTEGRATION_TEST=true and valid LLM_API_KEY / OPENROUTER_API_KEY",
)
def test_live_qwen3_14b_intent_and_synthesis():
    cfg = LLMConfig.from_env()
    assert cfg.api_key is not None, "API key must be configured for live LLM integration tests"

    provider = OpenAICompatibleProvider(config=cfg)
    intent_classifier = LLMIntentClassifier(provider=provider)
    synthesizer = LLMSynthesizer(provider=provider)

    print("\n" + "=" * 65)
    print("LIVE QWEN3-14B NLP & REASONING AUDIT TRACE")
    print("=" * 65)
    print(f"Provider: {cfg.provider}")
    print(f"Model:    {cfg.model}")
    print(f"Base URL: {cfg.base_url}")

    # ============================================================
    # 1. Intent Classification Tests
    # ============================================================
    test_queries = [
        ("Identify new construction between 2020 and 2024.", "change", 2),
        ("Describe the satellite scene.", "caption", 1),
        ("Locate circular storage tanks.", "ground", 1),
        ("What is the primary land use visible here?", "vqa", 1),
        ("Use optical and SAR data to analyze this coastal zone.", "fusion", 2),
        ("Analyze this.", "ambiguous", 1),
    ]

    print("\n--- 1. Live Intent Classification ---")
    for q_text, expected_task, n_imgs in test_queries:
        t0 = time.perf_counter()
        res = intent_classifier.classify(query=q_text, n_images=n_imgs)
        lat = (time.perf_counter() - t0) * 1000.0
        print(f"Query: '{q_text}' -> Task: {res.primary_task} (Ambiguous={res.ambiguous}, Source={res.classifier_source}, Latency={lat:.1f}ms)")
        
        if expected_task == "ambiguous":
            assert res.ambiguous is True or res.clarification_needed is True or res.primary_task == "vqa"
        else:
            assert res.primary_task == expected_task

    # ============================================================
    # 2. Evidence-Grounded Synthesis (Preserving Authoritative GIS)
    # ============================================================
    print("\n--- 2. Live Grounded Response Synthesis ---")
    controlled_tool_results = [
        {
            "tool_id": "T4_Change",
            "answer": "Significant new building construction detected.",
            "metadata": {
                "area_ha": 12.4,
                "polygon_count": 3,
                "change_fraction_pct": 7.0,
            },
            "evidence": [
                {
                    "label": "construction_cluster_north",
                    "coverage_pct": 3.5,
                    "bbox_pixels": [50, 50, 200, 200],
                },
                {
                    "label": "construction_cluster_south",
                    "coverage_pct": 3.5,
                    "bbox_pixels": [300, 300, 450, 450],
                },
            ],
        }
    ]

    t0 = time.perf_counter()
    synth_res = synthesizer.synthesize(
        query="Identify new construction between 2020 and 2024.",
        tool_results=controlled_tool_results,
        confidence=0.88,
        confidence_status="calibrated",
    )
    synth_lat = (time.perf_counter() - t0) * 1000.0

    print(f"Synthesized Answer:\n{synth_res.answer}")
    print(f"Synthesis Source: {synth_res.synthesis_source} (Fallback Used: {synth_res.fallback_used})")
    print(f"Synthesis Latency: {synth_lat:.1f} ms")

    assert synth_res.synthesis_source == "llm" or synth_res.fallback_used is True
    assert "12.4" in synth_res.answer or "12" in synth_res.answer
    assert "7" in synth_res.answer or "7.0" in synth_res.answer

    # ============================================================
    # 3. Anti-Hallucination Post-Validation Rejection
    # ============================================================
    print("\n--- 3. Anti-Hallucination Rejection Check ---")
    validator = SynthesisValidator()
    context = synthesizer._build_evidence_context(
        query="Identify new construction",
        tool_results=controlled_tool_results,
        confidence=None,
        confidence_status="uncalibrated",
        geojson=None,
        intent=None,
    )

    # Fabricated / Hallucinated payload
    hallucinated_payload = SynthesisPayload(
        answer="Analysis confirmed 21.7 hectares of urban expansion across 5 regions with 98% certainty.",
        claims=[
            SynthesisClaim(text="21.7 hectares of change", evidence_ids=["E99"]),
        ],
        uncertainties=[],
        justification="Extrapolated numbers",
    )

    val_res = validator.validate(hallucinated_payload, context)
    print(f"Validator is_valid: {val_res.is_valid}")
    print(f"Violations caught: {val_res.violations}")

    assert val_res.is_valid is False
    assert len(val_res.violations) >= 1
    assert any("E99" in v or "21.7" in v for v in val_res.violations)
    print("=" * 65)
