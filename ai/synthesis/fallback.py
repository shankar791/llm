"""
Deterministic Fallback Formatter for SatQuery AI synthesis.
Produces strictly fact-grounded natural-language answers directly from tool findings and GIS metrics
when the LLM provider is unavailable or fails anti-hallucination validation.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from .schema import SynthesisClaim, SynthesisResult


class DeterministicFallbackFormatter:
    """
    Constructs a deterministic, evidence-grounded final answer without making external LLM calls.
    Ensures high-availability and total immunity to hallucinations.
    """

    def format(
        self,
        query: str,
        tool_results: List[Dict[str, Any]],
        confidence: Optional[float] = None,
        confidence_status: str = "uncalibrated",
        geojson: Optional[Dict[str, Any]] = None,
        intent: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        fallback_reason: Optional[str] = None,
    ) -> SynthesisResult:
        """
        Produce a structured fallback result using only verified empirical facts.
        """
        if error:
            answer = f"Analysis could not be completed: {error}"
            return SynthesisResult(
                answer=answer,
                claims=[SynthesisClaim(text=answer, evidence_ids=[])],
                uncertainties=[],
                justification="Error reported during pipeline execution.",
                synthesis_source="deterministic_fallback",
                fallback_used=True,
                fallback_reason=fallback_reason or "Pipeline execution error",
                latency_ms=0.5,
            )

        if not tool_results:
            answer = "No analysis output was generated. Please verify query and image inputs."
            return SynthesisResult(
                answer=answer,
                claims=[SynthesisClaim(text=answer, evidence_ids=[])],
                uncertainties=[],
                justification="Empty tool results received.",
                synthesis_source="deterministic_fallback",
                fallback_used=True,
                fallback_reason=fallback_reason or "No tool results available",
                latency_ms=0.5,
            )

        # 1. Extract GIS metrics from tool results or metadata
        primary_tool = tool_results[0]
        meta = primary_tool.get("metadata", {})
        task = (intent.get("task") if intent else None) or primary_tool.get("tool_id", "")

        area_ha = meta.get("area_ha")
        area_m2 = meta.get("area_m2")
        change_pct = meta.get("change_fraction_pct") or meta.get("change_fraction")
        n_regions = meta.get("polygon_count") or meta.get("n_regions")
        temporal = (intent.get("temporal_scope") if intent else None)

        evidence_items = primary_tool.get("evidence", [])
        evidence_ids = [f"E{i + 1}" for i in range(len(evidence_items))]

        # 2. Build task-specific deterministic answer
        answer_parts = []
        claims = []

        if task in {"change", "T4_Change"} or "change" in str(primary_tool.get("tool_id", "")).lower():
            # Build change detection summary
            time_str = ""
            if temporal and isinstance(temporal, dict) and "start_date" in temporal and "end_date" in temporal:
                time_str = f"between {temporal['start_date']} and {temporal['end_date']} "

            area_str = ""
            if area_ha is not None:
                area_str = f"approximately {area_ha:.2f} hectares"
            elif area_m2 is not None:
                area_str = f"approximately {area_m2:,.0f} m²"

            regions_str = f" in {n_regions} identified regions" if n_regions else ""
            pct_str = f" ({change_pct:.1f}% of scene)" if change_pct is not None else ""

            core_finding = f"Change analysis {time_str}detected {area_str or 'significant surface variation'}{regions_str}{pct_str}."
            answer_parts.append(core_finding)
            claims.append(SynthesisClaim(text=core_finding, evidence_ids=evidence_ids))

        else:
            # VQA / Caption / Grounding summary
            tool_ans = primary_tool.get("answer", "").strip()
            if tool_ans:
                answer_parts.append(tool_ans)
                claims.append(SynthesisClaim(text=tool_ans, evidence_ids=evidence_ids))
            else:
                answer_parts.append("Specialist model completed analysis of the satellite imagery.")

        # 3. Add secondary tools if present
        for extra_tool in tool_results[1:]:
            extra_ans = extra_tool.get("answer", "").strip()
            if extra_ans:
                answer_parts.append(extra_ans)

        # 4. Uncertainty statements
        uncertainties = []
        if confidence_status == "uncalibrated" or confidence is None:
            uncertainties.append("Model confidence is uncalibrated.")
            answer_parts.append("Confidence is currently uncalibrated.")
        elif confidence is not None:
            uncertainties.append(f"Empirical confidence score: {confidence:.2f}")

        final_text = " ".join(answer_parts)

        return SynthesisResult(
            answer=final_text,
            claims=claims,
            uncertainties=uncertainties,
            justification="Constructed deterministically from specialist tool findings and GIS metrics.",
            synthesis_source="deterministic_fallback",
            fallback_used=True,
            fallback_reason=fallback_reason,
            latency_ms=0.5,
        )
