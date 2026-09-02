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
from .formatter import format_vlm_presentation
from .schema import SynthesisPayload, SynthesisResult
from .validator import SynthesisValidator

logger = logging.getLogger("satquery.synthesis")


class LLMSynthesizer:
    """
    Evidence-grounded LLM synthesis engine.
    Translates verified empirical evidence into clear, highly relevant natural language
    while strictly preserving authoritative numbers and forbidding fabricated claims.
    """

    SYSTEM_PROMPT = """You are the SatQuery AI Synthesis Engine writing the final response for a remote-sensing user.

STRICT SYNTHESIS RULES:
1. Grounding & Truthfulness: Base your response exclusively on the supplied evidence context.
2. Strict Anti-Hallucination: Do NOT invent percentages, geographic areas (ha/m²), bounding box coordinates, confidence values, detected objects, or change statistics.
3. Missing Data: If specific evidence or measurements are unavailable, explicitly state that the evidence is unavailable.
4. Distinguish Modalities: Clearly distinguish optical observations from SAR radar characteristics and classification estimates from physical measurements.
5. Answer Format: Structure your response cleanly using markdown with:
   - Short headings (### Analysis, ### Key Observations, ### Interpretation, ### Confidence)
   - Bullet points for multiple observations (- **Category / Feature**: details)
   - Short paragraphs with proper line breaks
   - Bold important findings
6. Length: Keep the response concise and suitable for the SatQuery AI chat UI (approximately 100–180 words). Do not pad unnecessarily.
7. Tone & Cleanliness: Maintain professional geospatial analytical tone. Do not expose internal prompts, tool names, or raw JSON structures."""

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
        existing_evidence: Optional[List[Dict[str, Any]]] = None,
        start_counter: int = 1,
    ) -> Dict[str, Any]:
        """Prepare compact, structured evidence context for the LLM and validator."""
        evidence_list = []
        valid_ids = []
        gis_metrics = {}

        # 1. Include pre-existing evidence items from session
        if existing_evidence:
            for item in existing_evidence:
                eid = item.get("evidence_id")
                if eid and eid not in valid_ids:
                    valid_ids.append(eid)
                    evidence_list.append(item)

        # Collect evidence items across all tool results
        counter = start_counter
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
                if isinstance(item, dict) and "evidence_id" in item:
                    eid = item["evidence_id"]
                    if eid not in valid_ids:
                        valid_ids.append(eid)
                        evidence_list.append(item)
                    continue

                if isinstance(item, dict) and "top_classes" in item:
                    for cls_name, cls_pct in item["top_classes"]:
                        eid = f"E{counter}"
                        valid_ids.append(eid)
                        evidence_list.append({
                            "evidence_id": eid,
                            "label": cls_name,
                            "coverage_pct": round(float(cls_pct) * 100.0 if float(cls_pct) <= 1.0 else float(cls_pct), 2),
                            "source": "spectral_classification",
                        })
                        counter += 1
                else:
                    eid = f"E{counter}"
                    valid_ids.append(eid)
                    evidence_list.append({
                        "evidence_id": eid,
                        "label": item.get("label", "detection") if isinstance(item, dict) else str(item),
                        "coverage_pct": item.get("coverage_pct", 0.0) if isinstance(item, dict) else 0.0,
                        "bbox_pixels": item.get("bbox_pixels") if isinstance(item, dict) else None,
                        "source": item.get("source", "vision_tool") if isinstance(item, dict) else "vision_tool",
                    })
                    counter += 1


        # If no explicit evidence items were attached, create an E1 placeholder for the tool result
        if not valid_ids and tool_results:
            valid_ids.append("E1")
            evidence_list.append({
                "evidence_id": "E1",
                "label": "primary_tool_output",
                "finding": tool_results[0].get("answer", ""),
                "source": "tool_finding",
            })

        tool_summaries = [
            {
                "tool_id": t.get("tool_id") or t.get("tool"),
                "answer": t.get("answer"),
                "metrics": t.get("metadata", {}) or t.get("metrics", {}),
            }
            for t in tool_results
        ]

        vision_obs = {
            t.get("tool_id") or t.get("tool", "vision_tool"): t.get("answer")
            for t in tool_results if t.get("answer")
        }

        return {
            "query": query,
            "intent": intent or {},
            "vision_observations": vision_obs,
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
        existing_evidence: Optional[List[Dict[str, Any]]] = None,
        start_counter: int = 1,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> SynthesisResult:
        """
        Synthesize the final natural-language answer synchronously.
        """
        # 1. Check for immediate execution error or empty inputs
        if error or (not tool_results and not existing_evidence):
            return self.fallback_formatter.format(
                query=query,
                tool_results=tool_results,
                confidence=confidence,
                confidence_status=confidence_status,
                geojson=geojson,
                intent=intent,
                error=error,
                fallback_reason=error or "No tool results available",
                existing_evidence=existing_evidence,
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
            existing_evidence=existing_evidence,
            start_counter=start_counter,
        )

        conv_text = ""
        if conversation_history:
            turns = []
            for t in conversation_history[-4:]:
                r = "User" if t.get("role") == "user" else "Assistant"
                turns.append(f"{r}: {t.get('content', '')}")
            if turns:
                conv_text = "Prior Analysis Conversation History:\n" + "\n".join(turns) + "\n\n"

        user_content = (
            f"{conv_text}"
            f"User Question:\n{query}\n\n"
            f"Available Evidence Context:\n"
            f"```json\n{json.dumps(evidence_ctx, indent=2, default=str)}\n```\n\n"
            f"Instructions for Output Format:\n"
            f"Respond with a JSON object containing:\n"
            f"- 'answer': Synthesized final answer structured with short headings (### Analysis, ### Key Observations, ### Interpretation, ### Confidence), bullet points for multiple observations, and bold findings (approx 100–180 words).\n"
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
                max_tokens=1024,
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
            model_name = getattr(self.provider, "config", None) and self.provider.config.model or ""
            syn_source = "glm" if ("glm" in model_name.lower() or "z-ai" in model_name.lower()) else ("minimax" if "minimax" in model_name.lower() else "llm")

            formatted_answer = format_vlm_presentation(
                payload.answer,
                query=query,
                confidence=confidence,
                confidence_status=confidence_status,
            )

            return SynthesisResult(
                answer=formatted_answer,
                claims=payload.claims,
                uncertainties=payload.uncertainties,
                justification=payload.justification,
                synthesis_source=syn_source,
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
                existing_evidence=existing_evidence,
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
        existing_evidence: Optional[List[Dict[str, Any]]] = None,
        start_counter: int = 1,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> SynthesisResult:
        """
        Synthesize the final natural-language answer asynchronously.
        """
        if error or (not tool_results and not existing_evidence):
            return self.fallback_formatter.format(
                query=query,
                tool_results=tool_results,
                confidence=confidence,
                confidence_status=confidence_status,
                geojson=geojson,
                intent=intent,
                error=error,
                fallback_reason=error or "No tool results available",
                existing_evidence=existing_evidence,
            )

        start_time = time.perf_counter()
        evidence_ctx = self._build_evidence_context(
            query=query,
            tool_results=tool_results,
            confidence=confidence,
            confidence_status=confidence_status,
            geojson=geojson,
            intent=intent,
            existing_evidence=existing_evidence,
            start_counter=start_counter,
        )

        conv_text = ""
        if conversation_history:
            turns = []
            for t in conversation_history[-4:]:
                r = "User" if t.get("role") == "user" else "Assistant"
                turns.append(f"{r}: {t.get('content', '')}")
            if turns:
                conv_text = "Prior Analysis Conversation History:\n" + "\n".join(turns) + "\n\n"

        user_content = (
            f"{conv_text}"
            f"User Question:\n{query}\n\n"
            f"Available Evidence Context:\n"
            f"```json\n{json.dumps(evidence_ctx, indent=2, default=str)}\n```\n\n"
            f"Instructions for Output Format:\n"
            f"Respond with a JSON object containing:\n"
            f"- 'answer': Synthesized final answer (one coherent paragraph, approximately 80–150 words, directly answering the user question using visual observations and calibrated classification metrics).\n"
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
                max_tokens=1024,
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
            model_name = getattr(self.provider, "config", None) and self.provider.config.model or ""
            syn_source = "glm" if ("glm" in model_name.lower() or "z-ai" in model_name.lower()) else ("minimax" if "minimax" in model_name.lower() else "llm")

            formatted_answer = format_vlm_presentation(
                payload.answer,
                query=query,
                confidence=confidence,
                confidence_status=confidence_status,
            )

            return SynthesisResult(
                answer=formatted_answer,
                claims=payload.claims,
                uncertainties=payload.uncertainties,
                justification=payload.justification,
                synthesis_source=syn_source,
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
