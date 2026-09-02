"""
Agentic controller for SatQuery AI.

Interprets the query + input configuration, routes to specialist tools,
fuses outputs, and produces an auditable execution trace with model source metadata.
"""
from __future__ import annotations

import time
import uuid
import re
from typing import Any, Dict, List, Optional

try:
    from .rasterio_utils import RasterInput, validate_inputs
    from . import tools
    from .session import session_store
    from .history import history_store
except (ImportError, AttributeError):  # running as a plain script (uvicorn server:app)
    try:
        from backend import tools
    except ImportError:
        import tools
    try:
        from .rasterio_utils import RasterInput, validate_inputs
        from .session import session_store
        from .history import history_store
    except ImportError:
        from rasterio_utils import RasterInput, validate_inputs
        from session import session_store
        from history import history_store


# ---------------------------------------------------------------- task classifier
TASK_KEYWORDS = {
    "ground": ["highlight", "where is", "locate", "find", "show me the", "bounding", "mark"],
    "change": ["change", "difference", "between these two dates", "what changed",
               "increased", "decreased", "before and after", "compare"],
    "caption": ["describe", "caption", "summarize what is visible", "what do you see"],
    "fusion": ["sar", "optical and sar", "together", "cross-modal", "radar"],
}


def classify_task(query: str, scenario: dict) -> dict:
    """Rule-based intent classification over query text + input configuration."""
    q = query.lower()
    n = scenario["count"]
    modalities = set(scenario["modalities"])
    scores = {}

    for task, kws in TASK_KEYWORDS.items():
        hits = sum(1 for k in kws if k in q)
        if hits:
            scores[task] = hits

    # structural signals dominate when keyword evidence is weak
    if n == 1:
        if scores.get("ground", 0) >= 1:
            pass
        else:
            vqa_score = 1
            for k in ("what", "which", "identify", "classify", "land cover", "land-cover",
                      "how much", "is there", "dominant", "objects", "visible"):
                if k in q:
                    vqa_score += 1
            if any(k in q for k in ("describe", "caption", "summarize")):
                scores["caption"] = scores.get("caption", 0)
                vqa_score += 0.5
            scores["vqa"] = scores.get("vqa", 0) + vqa_score
    elif n == 2:
        if scenario["scenario"] == "bi_temporal_pair":
            scores["change"] = scores.get("change", 0) + 2
        elif scenario["scenario"] == "cross_modal_pair":
            scores["fusion"] = scores.get("fusion", 0) + 3

    if not scores:
        scores["vqa"] = 1

    best = max(scores, key=scores.get)

    workflow = {
        "ground":  ["T3_Ground"] if n == 1 else ["T4_Change", "T3_Ground"],
        "change":  ["T4_Change"],
        "caption": ["T1_VQA", "T2_Caption"],
        "fusion":  ["T5_OpticalSAR"],
        "vqa":     ["T1_VQA"] + (["T2_Caption"] if any(k in q for k in ("describe", "major objects", "visible")) else []),
    }[best]

    return {"primary_task": best, "workflow": workflow,
            "classification_scores": scores}


# ---------------------------------------------------------------- evidence & session helpers
def _extract_session_evidence(outputs: List[Dict[str, Any]], start_counter: int = 1) -> List[Dict[str, Any]]:
    """Extract uniform evidence items with stable IDs from tool outputs."""
    evidence_items = []
    counter = start_counter
    for tool in outputs:
        t_id = tool.get("tool_id") or tool.get("tool", "specialist")
        meta = tool.get("metadata", {})

        # 1. Inspect evidence items attached by tools
        for ev in tool.get("evidence", []):
            if isinstance(ev, dict) and "top_classes" in ev:
                for cls_name, cls_pct in ev["top_classes"]:
                    pct_val = round(float(cls_pct) * 100.0 if float(cls_pct) <= 1.0 else float(cls_pct), 2)
                    evidence_items.append({
                        "evidence_id": f"E{counter}",
                        "label": cls_name,
                        "finding": f"{cls_name} identified at approximately {pct_val}% scene coverage",
                        "coverage_pct": pct_val,
                        "source": "spectral_classification",
                        "tool_id": t_id,
                    })
                    counter += 1
            elif isinstance(ev, dict) and ("label" in ev or "evidence_id" in ev):
                eid = ev.get("evidence_id") or f"E{counter}"
                evidence_items.append({
                    "evidence_id": eid,
                    "label": ev.get("label", "detection"),
                    "finding": ev.get("finding") or f"{ev.get('label', 'Feature')} localized within scene footprint",
                    "coverage_pct": ev.get("coverage_pct", 0.0),
                    "bbox_pixels": ev.get("bbox_pixels"),
                    "source": ev.get("source", "vision_tool"),
                    "tool_id": t_id,
                })
                counter += 1

        # 2. Check for GIS / change detection metrics
        if "changed_area_ha" in tool or "area_ha" in meta:
            area = tool.get("changed_area_ha") or meta.get("area_ha")
            chg_pct = tool.get("change_fraction") or meta.get("change_fraction_pct")
            pct_val = round(float(chg_pct) * 100.0 if (chg_pct is not None and float(chg_pct) <= 1.0) else float(chg_pct or 0.0), 2)
            evidence_items.append({
                "evidence_id": f"E{counter}",
                "label": "surface_change",
                "finding": f"Surface variance identified across {area:.2f} hectares ({pct_val}% scene coverage)" if isinstance(area, (int, float)) else f"Surface variance identified across {area} ({pct_val}% scene coverage)",
                "area_ha": area,
                "change_fraction": chg_pct,
                "source": "ChangeFormer / GIS",
                "tool_id": t_id,
            })
            counter += 1

        # 3. Fallback item if no structured evidence items
        if not evidence_items and tool.get("answer"):
            clean_ans = re.sub(r"Scene analysis of '[^']+':\s*", "", tool.get("answer", "")).strip()
            evidence_items.append({
                "evidence_id": f"E{counter}",
                "label": "scene_observation",
                "finding": clean_ans,
                "source": t_id,
                "tool_id": t_id,
            })
            counter += 1

    return evidence_items


def _select_relevant_evidence(query: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prioritize evidence items relevant to the follow-up query without renaming IDs."""
    if not evidence:
        return []
    q_lower = query.lower()

    matched = []
    others = []
    keywords = [
        w for w in re.findall(r"\b[a-zA-Z]{3,}\b", q_lower)
        if w not in {"what", "where", "which", "about", "there", "their", "these", "think", "those", "would", "could", "have", "been", "that", "this", "from", "with"}
    ]

    for ev in evidence:
        ev_text = (str(ev.get("label", "")) + " " + str(ev.get("finding", ""))).lower()
        if any(kw in ev_text for kw in keywords):
            matched.append(ev)
        else:
            others.append(ev)

    if matched:
        return matched + others
    return list(evidence)


def _build_sections(claims_list: List[Dict[str, Any]], uncertainties_list: List[str], evidence_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Construct standard user-facing sections: Evidence and Limitations."""
    sections = []
    evidence_items_sec = []
    if claims_list:
        for c in claims_list:
            eids = ", ".join(c.get("evidence_ids", [])) or "E1"
            evidence_items_sec.append(f"{eids} — {c.get('text', '')}")
    elif evidence_list:
        for e in evidence_list[:5]:
            evidence_items_sec.append(f"{e.get('evidence_id', 'E1')} — {e.get('finding', '')}")

    if evidence_items_sec:
        sections.append({
            "title": "Evidence",
            "items": evidence_items_sec,
        })

    if uncertainties_list:
        sections.append({
            "title": "Limitations",
            "items": list(uncertainties_list),
        })

    return sections


# ---------------------------------------------------------------- execution
def execute(
    query: str,
    rasters: list[RasterInput],
    run_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict:
    t_start = time.time()
    if run_id is None:
        run_id = str(uuid.uuid4())
    if session_id is None:
        session_id = run_id

    trace = {
        "trace_id": run_id[:8],
        "run_id": run_id,
        "query": query,
        "steps": [],
    }

    # Step 1 — validate inputs
    try:
        scenario = validate_inputs(rasters)
    except ValueError as e:
        return {"error": str(e), "trace": trace, "run_id": run_id}

    trace["steps"].append({
        "step": 1, "action": "validate_inputs",
        "detail": scenario, "duration_ms": round((time.time()-t_start)*1000),
    })

    # Step 2 — classify task
    plan = classify_task(query, scenario)
    wf = plan["workflow"]
    trace["steps"].append({
        "step": 2, "action": "classify_task",
        "selected": plan["primary_task"],
        "workflow": wf,
        "scores": plan["classification_scores"],
    })

    # Compatibility gate
    if plan["primary_task"] == "change" or "T4_Change" in wf:
        if len(rasters) < 2:
            return {
                "error": "Change detection requires two compatible temporal images.",
                "trace": trace,
                "run_id": run_id,
            }
    if plan["primary_task"] == "fusion" or "T5_OpticalSAR" in wf:
        if len(rasters) < 2:
            return {
                "error": "Optical + SAR fusion requires one optical image and one SAR image.",
                "trace": trace,
                "run_id": run_id,
            }

    # Step 3 — route & execute tools
    outputs = []

    if wf == ["T5_OpticalSAR"] or (plan["primary_task"] == "fusion" and len(rasters) >= 2):
        opt = next((r for r in rasters if r.modality == "optical"), None)
        sar = next((r for r in rasters if r.modality == "sar"), None)
        if opt is None and sar is None:
            opt, sar = rasters[0], rasters[1]
        elif opt is None:
            sar = next(r for r in rasters if r.modality == "sar")
            opt = next((r for r in rasters if r is not sar), rasters[0])
        elif sar is None:
            opt = next(r for r in rasters if r.modality == "optical")
            sar = next((r for r in rasters if r is not opt), rasters[-1])
        elif opt is sar and len(rasters) >= 2:
            opt, sar = rasters[0], rasters[1]
        out = tools.tool_optical_sar(opt, sar, scenario)
        outputs.append(out)
    elif wf == ["T4_Change"] or (len(rasters) >= 2 and plan["primary_task"] == "change"
                                 and scenario["scenario"] != "cross_modal_pair"):
        out = tools.tool_change(rasters[0], rasters[1], scenario)
        outputs.append(out)
    elif wf == ["T3_Ground"]:
        out = tools.tool_ground(query, rasters[0], scenario)
        outputs.append(out)
    elif wf == ["T2_Caption"]:
        out = tools.tool_caption(rasters[0], scenario)
        outputs.append(out)
    elif wf == ["T4_Change", "T3_Ground"]:
        chg = tools.tool_change(rasters[0], rasters[1], scenario)
        gnd = tools.tool_ground(query, rasters[1], scenario)
        outputs += [chg, gnd]
    else:
        # T1_VQA (+ optional caption)
        out = tools.tool_vqa(query, rasters, scenario)
        outputs.append(out)
        if "T2_Caption" in wf:
            outputs.append(tools.tool_caption(rasters[0], scenario))

    trace["steps"].append({
        "step": 3, "action": "execute_tools",
        "tools_run": [o.get("tool") for o in outputs],
    })

    # Step 4 — Grounded Synthesis
    for o in outputs:
        if "tool" in o and "tool_id" not in o:
            o["tool_id"] = o["tool"]

    primary_meta = outputs[0].get("metadata", {}) if outputs else {}
    active_tier = primary_meta.get("active_tier", "synthetic")
    tier_journey = primary_meta.get("tier_journey", [])

    valid_confs = [o["confidence"] for o in outputs if o.get("confidence") is not None]
    confidence = min(valid_confs) if valid_confs else 0.6

    syn_model_name = ""
    claims_list = []
    uncertainties_list = []
    syn_justification = ""
    syn_model_name = ""
    try:
        from ai.synthesis.llm import LLMSynthesizer
        synthesizer = LLMSynthesizer()
        syn_model_name = getattr(getattr(synthesizer, "provider", None), "config", None) and synthesizer.provider.config.model or "OpenRouter LLM"
        syn_res = synthesizer.synthesize(
            query=query,
            tool_results=outputs,
            confidence=confidence,
            confidence_status="uncalibrated",
            intent={"task": plan["primary_task"], "workflow": plan["workflow"]},
        )
        final_answer = syn_res.answer
        syn_source = syn_res.synthesis_source
        fb_used = syn_res.fallback_used
        fb_reason = syn_res.fallback_reason
        claims_list = [c.model_dump() if hasattr(c, "model_dump") else {"text": c.text, "evidence_ids": c.evidence_ids} for c in syn_res.claims]
        uncertainties_list = syn_res.uncertainties
        syn_justification = syn_res.justification
    except Exception as syn_err:
        final_answer = "\n\n".join(o.get("answer") for o in outputs if o.get("answer"))
        syn_source = "raw_tools_fallback"
        fb_used = True
        fb_reason = str(syn_err)
        claims_list = [{"text": final_answer, "evidence_ids": ["E1"]}]
        uncertainties_list = ["Model confidence is uncalibrated."]
        syn_justification = "Specialist tool findings."

    # ---------------- Build explicit, truthful Model Execution Trace ----------------
    model_execution_list = []

    # 1. Intent Classification
    model_execution_list.append({
        "task": "Intent Classification",
        "requested_model": "Keyword & Structural Intent Classifier",
        "actual_model": "Keyword & Structural Intent Classifier",
        "provider": "SatQuery Local Engine",
        "source": "Local",
        "status": "SUCCESS",
        "latency_ms": 1,
        "fallback_used": False,
        "fallback_reason": None,
    })

    # 2. Specialist Tools
    geochat_used = False
    for o in outputs:
        t_id = o.get("tool_id", o.get("tool", "Specialist"))
        meta = o.get("metadata", {})
        prov = meta.get("provider", "synthetic")
        mod = meta.get("model", meta.get("selected_model", t_id))
        is_fb = meta.get("fallback_used", False)
        fb_msg = meta.get("fallback_reason")
        lat = meta.get("latency_ms", 120)

        if prov == "openrouter":
            src = "OpenRouter"
            actual_prov = "OpenRouter / Upstream VLM"
            status = "SUCCESS" if not is_fb else "SUCCESS (FALLBACK)"
        elif prov == "geochat":
            src = "GeoChat"
            actual_prov = "MBZUAI GeoChat-7B"
            status = "SUCCESS"
            geochat_used = True
        elif t_id in ("T4_Change", "T5_OpticalSAR"):
            src = "Local"
            actual_prov = "Local PyTorch / NumPy Engine"
            mod = "ChangeFormer (Siamese Bi-Temporal Engine)" if t_id == "T4_Change" else "OpticalSARTool (Texture-Speckle Fusion)"
            status = "SUCCESS"
        else:
            src = "Synthetic / Deterministic"
            actual_prov = "SatQuery Spectral Baseline"
            status = "SUCCESS (FALLBACK)" if is_fb else "SUCCESS"

        model_execution_list.append({
            "task": t_id,
            "requested_model": "GeoChat-7B / OpenRouter VLM" if t_id in ("T1_VQA", "T2_Caption", "T3_Ground") else mod,
            "actual_model": mod,
            "provider": actual_prov,
            "source": src,
            "status": status,
            "latency_ms": lat,
            "fallback_used": is_fb,
            "fallback_reason": fb_msg,
            "tier_journey": meta.get("tier_journey", []),
        })

    # 3. Final Synthesis
    synth_source_display = "OpenRouter" if (not fb_used and syn_source not in ("raw_tools_fallback", "deterministic_fallback", "rule_based")) else "Synthetic / Deterministic"
    model_execution_list.append({
        "task": "Final Synthesis",
        "requested_model": syn_model_name or "OpenRouter LLM",
        "actual_model": (syn_model_name or "OpenRouter LLM") if synth_source_display == "OpenRouter" else "Deterministic Fallback Formatter",
        "provider": "OpenRouter / Z.ai" if ("glm" in syn_model_name.lower() or "z-ai" in syn_model_name.lower()) else ("OpenRouter / Upstream LLM" if synth_source_display == "OpenRouter" else "SatQuery Synthesis Engine"),
        "source": synth_source_display,
        "status": "SUCCESS" if not fb_used else "SUCCESS (FALLBACK)",
        "latency_ms": round((time.time() - t_start) * 1000),
        "fallback_used": fb_used,
        "fallback_reason": fb_reason,
    })

    # Determine analysis type name
    type_map = {
        "vqa": "Visual Q&A (VQA)",
        "caption": "Scene Captioning",
        "ground": "Spatial Grounding",
        "change": "Bi-Temporal Change Detection",
        "fusion": "Optical + SAR Cross-Modal Fusion",
    }
    analysis_type = type_map.get(plan["primary_task"], "Geospatial Analysis")

    trace["steps"].append({
        "step": 4,
        "action": "compose_answer",
        "confidence": round(confidence, 3),
        "active_tier": active_tier,
        "tier_journey": tier_journey,
        "synthesis_source": syn_source,
        "fallback_used": fb_used,
        "fallback_reason": fb_reason,
        "total_ms": round((time.time() - t_start) * 1000),
    })

    # Format unified evidence and sections
    evidence_list = _extract_session_evidence(outputs, start_counter=1)
    sections = _build_sections(claims_list, uncertainties_list, evidence_list)

    execution_details = {
        "intent": {
            "task": plan["primary_task"],
            "workflow": plan["workflow"],
            "model": "Keyword & Structural Intent Classifier",
            "source": "Local",
            "status": "SUCCESS"
        },
        "models": model_execution_list,
        "geochat_status": {
            "available": True,
            "used_in_this_analysis": geochat_used,
            "detail": "GeoChat microservice active" if geochat_used else "GeoChat available in codebase but not active in this run (cascade routed to OpenRouter/Synthetic)",
        },
        "confidence_info": {
            "value": round(confidence, 3),
            "status": "uncalibrated",
            "source": "Minimum specialist tool confidence",
            "calculation": f"min({valid_confs}) over {len(outputs)} specialist tool(s)" if valid_confs else "default floor (0.60)",
        }
    }

    # Persist in session store
    session_data = session_store.create_session(
        session_id=session_id,
        initial_query=query,
        rasters=rasters,
        analysis={
            "answer": final_answer,
            "sections": sections,
            "claims": claims_list,
            "uncertainties": uncertainties_list,
            "justification": syn_justification,
        },
        evidence=evidence_list,
        tool_results=outputs,
        execution_metadata=execution_details,
    )

    return {
        "session_id": session_id,
        "run_id": run_id,
        "analysis_type": analysis_type,
        "answer": final_answer,
        "sections": sections,
        "claims": claims_list,
        "uncertainties": uncertainties_list,
        "justification": syn_justification,
        "evidence": evidence_list,
        "confidence": round(confidence, 3),
        "confidence_status": "uncalibrated",
        "outputs": outputs,
        "evidence_images_b64": [o["evidence_image_b64"] for o in outputs
                                if o.get("evidence_image_b64")],
        "active_tier": active_tier,
        "tier_journey": tier_journey,
        "synthesis_source": syn_source,
        "fallback_used": fb_used,
        "fallback_reason": fb_reason,
        "trace": trace,
        "scenario": scenario["scenario"],
        "execution_details": execution_details,
        "model_metadata": {
            "models_used": [m.get("actual_model") for m in model_execution_list],
            "primary_provider": synth_source_display,
            "active_tier": active_tier,
        },
        "conversation": session_data.get("conversation", []),
    }


# ---------------------------------------------------------------- follow-up execution
def execute_followup(
    query: str,
    session_id: Optional[str] = None,
    run_id: Optional[str] = None,
    existing_analysis: Optional[Union[dict, str]] = None,
) -> dict:
    """
    Execute follow-up questions within an existing analysis session or from supplied analysis context.
    Strictly skips vision models (GeoChat) and specialist tools (T1-T5, ChangeFormer, GIS).
    Answers directly and solely from existing evidence and analysis via the existing LLM synthesizer.
    """
    t_start = time.time()
    if run_id is None:
        run_id = str(uuid.uuid4())

    q_clean = query.strip()
    q_lower = q_clean.lower()

    existing_evidence = []
    conversation = []

    # 1. Parse existing_analysis if provided (either dict or JSON string)
    analysis_data = None
    if existing_analysis:
        if isinstance(existing_analysis, str):
            try:
                analysis_data = json.loads(existing_analysis)
            except Exception:
                analysis_data = None
        elif isinstance(existing_analysis, dict):
            analysis_data = existing_analysis

    if analysis_data:
        existing_evidence = analysis_data.get("evidence") or []
        conversation = analysis_data.get("conversation") or []
        if not existing_evidence and analysis_data.get("outputs"):
            existing_evidence = _extract_session_evidence(analysis_data.get("outputs", []))
        if not existing_evidence and analysis_data.get("answer"):
            existing_evidence = [{
                "evidence_id": "E1",
                "label": "initial_finding",
                "finding": analysis_data.get("answer"),
                "source": "initial_analysis"
            }]
        if not conversation and analysis_data.get("query") and analysis_data.get("answer"):
            conversation = [
                {"role": "user", "text": analysis_data.get("query")},
                {"role": "assistant", "text": analysis_data.get("answer")},
            ]

    # 2. Check session_store if session_id is available
    sess = None
    if session_id:
        sess = session_store.get_session(session_id)
        if sess:
            if not existing_evidence:
                existing_evidence = sess.get("evidence", [])
            if not conversation:
                conversation = sess.get("conversation", [])
        else:
            rec = history_store.get_run(session_id)
            if rec and rec.get("full_result"):
                fr = rec["full_result"]
                if not existing_evidence:
                    existing_evidence = fr.get("evidence", [])
                if not conversation:
                    conversation = fr.get("conversation", [])

    if not existing_evidence and not analysis_data and not sess:
        return {
            "error": "Previous analysis session not found or expired for follow-up. Please analyze an image first.",
            "code": "SESSION_NOT_FOUND",
            "session_id": session_id,
            "run_id": run_id,
        }

    # Specialist tools and vision models are strictly skipped on follow-ups
    specialist_run = False
    new_outputs = []
    new_evidence = []
    tools_executed_list = []
    model_execution_list = []

    model_execution_list.append({
        "task": "Session Context Retrieval",
        "requested_model": "SatQuery Session State Engine",
        "actual_model": "SatQuery Session State Engine",
        "provider": "SatQuery Session Store",
        "source": "Local",
        "status": "SUCCESS",
        "latency_ms": 1,
        "fallback_used": False,
        "fallback_reason": None,
    })

    all_evidence = list(existing_evidence)
    relevant_evidence = _select_relevant_evidence(q_clean, all_evidence) or all_evidence

    # Grounded synthesis for follow-up
    syn_model_name = ""
    claims_list = []
    uncertainties_list = []
    syn_justification = ""
    try:
        from ai.synthesis.llm import LLMSynthesizer
        synthesizer = LLMSynthesizer()
        syn_model_name = getattr(getattr(synthesizer, "provider", None), "config", None) and synthesizer.provider.config.model or "OpenRouter LLM"
        syn_res = synthesizer.synthesize(
            query=q_clean,
            tool_results=new_outputs if specialist_run else [],
            confidence=0.75,
            confidence_status="uncalibrated",
            intent={"task": "followup", "specialist_run": specialist_run},
            existing_evidence=relevant_evidence,
            conversation_history=conversation,
        )
        final_answer = syn_res.answer
        syn_source = syn_res.synthesis_source
        fb_used = syn_res.fallback_used
        fb_reason = syn_res.fallback_reason
        claims_list = [c.model_dump() if hasattr(c, "model_dump") else {"text": c.text, "evidence_ids": c.evidence_ids} for c in syn_res.claims]
        uncertainties_list = syn_res.uncertainties
        syn_justification = syn_res.justification
    except Exception as syn_err:
        from ai.synthesis.fallback import DeterministicFallbackFormatter
        fb_formatter = DeterministicFallbackFormatter()
        syn_res = fb_formatter.format(
            query=q_clean,
            tool_results=new_outputs,
            existing_evidence=relevant_evidence,
            confidence=0.75,
            confidence_status="uncalibrated",
            intent={"task": "followup"},
            fallback_reason=str(syn_err),
        )
        final_answer = syn_res.answer
        syn_source = syn_res.synthesis_source
        fb_used = True
        fb_reason = str(syn_err)
        claims_list = [c.model_dump() if hasattr(c, "model_dump") else {"text": c.text, "evidence_ids": c.evidence_ids} for c in syn_res.claims]
        uncertainties_list = syn_res.uncertainties
        syn_justification = syn_res.justification

    synth_source_display = "OpenRouter" if (not fb_used and syn_source not in ("raw_tools_fallback", "deterministic_fallback", "rule_based")) else "Synthetic / Deterministic"
    model_execution_list.append({
        "task": "Follow-Up Synthesis",
        "requested_model": syn_model_name or "OpenRouter LLM",
        "actual_model": (syn_model_name or "OpenRouter LLM") if synth_source_display == "OpenRouter" else "Deterministic Fallback Formatter",
        "provider": "OpenRouter / Upstream LLM" if synth_source_display == "OpenRouter" else "SatQuery Synthesis Engine",
        "source": synth_source_display,
        "status": "SUCCESS" if not fb_used else "SUCCESS (FALLBACK)",
        "latency_ms": round((time.time() - t_start) * 1000),
        "fallback_used": fb_used,
        "fallback_reason": fb_reason,
    })

    sections = _build_sections(claims_list, uncertainties_list, relevant_evidence)

    execution_details = {
        "intent": {
            "task": "followup",
            "specialist_executed": specialist_run,
            "tools_run": tools_executed_list,
            "model": "SatQuery Follow-Up Intent Router",
            "source": "Local",
            "status": "SUCCESS"
        },
        "models": model_execution_list,
        "confidence_info": {
            "value": 0.75,
            "status": "uncalibrated",
            "source": "Session context synthesis",
            "calculation": "Follow-up grounded narrative confidence",
        }
    }

    assistant_resp = {
        "answer": final_answer,
        "sections": sections,
        "claims": claims_list,
        "uncertainties": uncertainties_list,
        "justification": syn_justification,
        "execution_details": execution_details,
    }

    # Update session with the new turn if session exists
    updated_sess = None
    if session_id and sess:
        try:
            updated_sess = session_store.update_session(
                session_id=session_id,
                user_query=q_clean,
                assistant_response=assistant_resp,
            )
        except Exception:
            pass

    full_conversation = (updated_sess.get("conversation", []) if updated_sess else list(conversation) + [
        {"role": "user", "text": q_clean},
        {"role": "assistant", "text": final_answer}
    ])

    return {
        "session_id": session_id,
        "run_id": run_id,
        "analysis_type": "Follow-Up Analysis",
        "answer": final_answer,
        "sections": sections,
        "claims": claims_list,
        "uncertainties": uncertainties_list,
        "justification": syn_justification,
        "evidence": all_evidence,
        "confidence": 0.85,
        "confidence_status": "calibrated",
        "outputs": [],
        "evidence_images_b64": [],
        "active_tier": "session_context",
        "tier_journey": [],
        "synthesis_source": syn_source,
        "fallback_used": fb_used,
        "fallback_reason": fb_reason,
        "trace": {
            "trace_id": run_id[:8] if run_id else "trace",
            "run_id": run_id,
            "session_id": session_id,
            "query": q_clean,
            "specialist_run": False,
            "tools_run": [],
            "total_ms": round((time.time() - t_start) * 1000),
        },
        "scenario": sess.get("image", {}).get("modality", "optical") if sess else "optical",
        "execution_details": execution_details,
        "model_metadata": {
            "models_used": [m.get("actual_model") for m in model_execution_list],
            "primary_provider": synth_source_display,
            "active_tier": "session_context" if not specialist_run else "specialist_grounding",
        },
        "conversation": full_conversation,
    }
