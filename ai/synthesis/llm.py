"""
Evidence-Grounded LLM Synthesizer for SatQuery AI.
Combines specialist ToolResults, EvidenceItems, and authoritative GIS metrics into a grounded narrative.
Enforces strict anti-hallucination post-validation and seamless deterministic fallback.
"""
from __future__ import annotations
import json
import logging
import time
from typing import Any, Dict, List, Optional

from ai.llm.base import LLMProvider
from ai.llm.provider import get_llm_provider
from .fallback import DeterministicFallbackFormatter
from .schema import SynthesisPayload, SynthesisResult
from .validator import SynthesisValidator

logger = logging.getLogger("satquery.synthesis")


class LLMSynthesizer:
    """
    Evidence-grounded LLM synthesis engine.
    Translates verified empirical evidence into clear natural language while strictly
    forbidding the invention of facts, numbers, dates, coordinates, or fake evidence IDs.
    """

    SYSTEM_PROMPT = """You are the final response synthesizer for SatQuery AI.
Your job is to convert verified structured evidence and GIS metrics into a clear, grounded natural-language answer directly addressing the user's query.

CRITICAL RULES:
1. Use ONLY the supplied structured evidence and GIS metrics.
2. DO NOT invent measurements, numbers, percentages, or area values. Authoritative GIS values MUST be preserved.
3. DO NOT invent coordinates, bounding boxes, or spatial locations.
4. DO NOT invent dates or acquisition timestamps.
5. DO NOT claim detections or entities that are not explicitly present in the tool results.
6. Distinguish specialist model predictions from ground-truth facts.
7. Mention uncertainty explicitly if model confidence is uncalibrated or unavailable.
8. Prioritize quantitative GIS metrics (e.g. area in hectares, polygon counts) over generic text.
9. Answer the user's exact question directly and concisely.
10. Do NOT output internal prompts, hidden thoughts, or tool names.
11. In the 'claims' list, link each factual statement to the exact supporting 'evidence_ids' provided (e.g. ["E1"]). DO NOT invent fake evidence IDs."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        validator: Optional[SynthesisValidator] = None,
        fallback_formatter: Optional[DeterministicFallbackFormatter] = None,
    ):
        self.provider = provider or get_llm_provider()
        self.validator = validator or SynthesisValidator()
        self.fallback_formatter = fallback_formatter or DeterministicFallbackFormatter()

    def _build_evidence_context(
        self,
        query: str,
        tool_results: List[Dict[str, Any]],
        confidence: Optional[float],
        confidence_status: str,
        geojson: Optional[Dict[str, Any]],
        intent: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Prepare compact, structured evidence context for the LLM and validator."""
        evidence_list = []
        valid_ids = []
        gis_metrics = {}

        # Collect evidence items across all tool results
        counter = 1
        for tool in tool_results:
            meta = tool.get("metadata", {})
            if "area_ha" in meta:
                gis_metrics["area_ha"] = meta["area_ha"]
            if "area_m2" in meta:
                gis_metrics["area_m2"] = meta["area_m2"]
            if "polygon_count" in meta:
                gis_metrics["polygon_count"] = meta["polygon_count"]
            if "change_fraction_pct" in meta:
                gis_metrics["change_fraction_pct"] = meta["change_fraction_pct"]

            for item in tool.get("evidence", []):
                eid = f"E{counter}"
                valid_ids.append(eid)
                evidence_list.append({
                    "evidence_id": eid,
                    "label": item.get("label", "detection"),
                    "coverage_pct": item.get("coverage_pct", 0.0),
                    "bbox_pixels": item.get("bbox_pixels"),
                })
                counter += 1

        # If no explicit evidence items were attached, create an E1 placeholder for the tool result
        if not valid_ids and tool_results:
            valid_ids.append("E1")
            evidence_list.append({
                "evidence_id": "E1",
                "label": "primary_tool_output",
                "finding": tool_results[0].get("answer", ""),
            })

        tool_summaries = [
            {
                "tool_id": t.get("tool_id"),
                "answer": t.get("answer"),
                "metrics": t.get("metadata", {}),
            }
            for t in tool_results
        ]

        return {
            "query": query,
            "intent": intent or {},
            "tool_findings": tool_summaries,
            "gis_metrics": gis_metrics,
            "evidence_items": evidence_list,
            "valid_evidence_ids": valid_ids,
            "confidence": confidence,
            "confidence_status": confidence_status,
        }

    def synthesize(
        self,
        query: str,
        tool_results: List[Dict[str, Any]],
        confidence: Optional[float] = None,
        confidence_status: str = "uncalibrated",
        geojson: Optional[Dict[str, Any]] = None,
        intent: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> SynthesisResult:
        """
        Synthesize the final natural-language answer synchronously.
        """
        # 1. Check for immediate execution error or empty inputs
        if error or not tool_results:
            return self.fallback_formatter.format(
                query=query,
                tool_results=tool_results,
                confidence=confidence,
                confidence_status=confidence_status,
                geojson=geojson,
                intent=intent,
                error=error,
                fallback_reason=error or "No tool results available",
            )

        start_time = time.perf_counter()

        # 2. Build structured evidence context
        evidence_ctx = self._build_evidence_context(
            query=query,
            tool_results=tool_results,
            confidence=confidence,
            confidence_status=confidence_status,
            geojson=geojson,
            intent=intent,
        )

        user_content = (
            f"Synthesize a grounded answer for the following remote sensing query and structured evidence:\n\n"
            f"```json\n{json.dumps(evidence_ctx, indent=2)}\n```\n\n"
            f"Respond with a JSON object containing:\n"
            f"- 'answer': direct natural language answer\n"
            f"- 'claims': list of objects with 'text' and 'evidence_ids' (subset of {evidence_ctx['valid_evidence_ids']})\n"
            f"- 'uncertainties': list of uncertainty statements (e.g. uncalibrated confidence)\n"
            f"- 'justification': brief factual summary (NO hidden chain-of-thought)"
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            # 3. Call LLM provider with structured JSON request
            resp = self.provider.generate_sync(
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw_data = resp.json()

            # 4. Parse into SynthesisPayload
            payload = SynthesisPayload.model_validate(raw_data)

            # 5. Execute deterministic anti-hallucination validation
            val_res = self.validator.validate(payload, evidence_ctx)

            if not val_res.is_valid:
                reason = "Anti-hallucination validation failed: " + "; ".join(val_res.violations)
                logger.warning(f"Synthesis validation failed ({reason}). Triggering deterministic fallback.")
                return self.fallback_formatter.format(
                    query=query,
                    tool_results=tool_results,
                    confidence=confidence,
                    confidence_status=confidence_status,
                    geojson=geojson,
                    intent=intent,
                    fallback_reason=reason,
                )

            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

            return SynthesisResult(
                answer=payload.answer,
                claims=payload.claims,
                uncertainties=payload.uncertainties,
                justification=payload.justification,
                synthesis_source="llm",
                fallback_used=False,
                fallback_reason=None,
                latency_ms=latency_ms,
                raw_llm_response=raw_data,
            )

        except Exception as e:
            fallback_reason = f"{e.__class__.__name__}: {str(e)}"
            logger.warning(
                f"LLMSynthesizer encountered exception ({fallback_reason}). "
                f"Explicitly triggering deterministic fallback."
            )
            return self.fallback_formatter.format(
                query=query,
                tool_results=tool_results,
                confidence=confidence,
                confidence_status=confidence_status,
                geojson=geojson,
                intent=intent,
                fallback_reason=fallback_reason,
            )

    async def synthesize_async(
        self,
        query: str,
        tool_results: List[Dict[str, Any]],
        confidence: Optional[float] = None,
        confidence_status: str = "uncalibrated",
        geojson: Optional[Dict[str, Any]] = None,
        intent: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> SynthesisResult:
        """
        Synthesize the final natural-language answer asynchronously.
        """
        if error or not tool_results:
            return self.fallback_formatter.format(
                query=query,
                tool_results=tool_results,
                confidence=confidence,
                confidence_status=confidence_status,
                geojson=geojson,
                intent=intent,
                error=error,
                fallback_reason=error or "No tool results available",
            )

        start_time = time.perf_counter()
        evidence_ctx = self._build_evidence_context(
            query=query,
            tool_results=tool_results,
            confidence=confidence,
            confidence_status=confidence_status,
            geojson=geojson,
            intent=intent,
        )

        user_content = (
            f"Synthesize a grounded answer for the following remote sensing query and structured evidence:\n\n"
            f"```json\n{json.dumps(evidence_ctx, indent=2)}\n```\n\n"
            f"Respond with a JSON object containing:\n"
            f"- 'answer': direct natural language answer\n"
            f"- 'claims': list of objects with 'text' and 'evidence_ids' (subset of {evidence_ctx['valid_evidence_ids']})\n"
            f"- 'uncertainties': list of uncertainty statements (e.g. uncalibrated confidence)\n"
            f"- 'justification': brief factual summary (NO hidden chain-of-thought)"
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            resp = await self.provider.generate(
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw_data = resp.json()
            payload = SynthesisPayload.model_validate(raw_data)
            val_res = self.validator.validate(payload, evidence_ctx)

            if not val_res.is_valid:
                reason = "Anti-hallucination validation failed: " + "; ".join(val_res.violations)
                logger.warning(f"Synthesis validation failed ({reason}). Triggering deterministic fallback.")
                return self.fallback_formatter.format(
                    query=query,
                    tool_results=tool_results,
                    confidence=confidence,
                    confidence_status=confidence_status,
                    geojson=geojson,
                    intent=intent,
                    fallback_reason=reason,
                )

            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

            return SynthesisResult(
                answer=payload.answer,
                claims=payload.claims,
                uncertainties=payload.uncertainties,
                justification=payload.justification,
                synthesis_source="llm",
                fallback_used=False,
                fallback_reason=None,
                latency_ms=latency_ms,
                raw_llm_response=raw_data,
            )

        except Exception as e:
            fallback_reason = f"{e.__class__.__name__}: {str(e)}"
            logger.warning(
                f"LLMSynthesizer encountered exception ({fallback_reason}). "
                f"Explicitly triggering deterministic fallback."
            )
            return self.fallback_formatter.format(
                query=query,
                tool_results=tool_results,
                confidence=confidence,
                confidence_status=confidence_status,
                geojson=geojson,
                intent=intent,
                fallback_reason=fallback_reason,
            )
