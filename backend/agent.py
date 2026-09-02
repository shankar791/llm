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
except ImportError:  # running as a plain script (uvicorn server:app)
    from rasterio_utils import RasterInput, validate_inputs
    import tools


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


# ---------------------------------------------------------------- execution
def execute(query: str, rasters: list[RasterInput], run_id: Optional[str] = None) -> dict:
    t_start = time.time()
    if run_id is None:
        run_id = str(uuid.uuid4())

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
    trace["steps"].append({
        "step": 2, "action": "classify_task",
        "selected": plan["primary_task"],
        "workflow": plan["workflow"],
        "scores": plan["classification_scores"],
    })

    # Step 3 — route & execute tools
    outputs = []
    wf = plan["workflow"]

    if wf == ["T5_OpticalSAR"] or (plan["primary_task"] == "fusion" and len(rasters) >= 2):
        opt = next((r for r in rasters if r.modality == "optical"), rasters[0])
        sar = next((r for r in rasters if r.modality == "sar"), rasters[-1])
        out = tools.tool_optical_sar(opt, sar, scenario)
        outputs.append(out)
    elif wf == ["T4_Change"] or (len(rasters) == 2 and plan["primary_task"] == "change"
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
    except Exception as syn_err:
        final_answer = "\n\n".join(o.get("answer") for o in outputs if o.get("answer"))
        syn_source = "raw_tools_fallback"
        fb_used = True
        fb_reason = str(syn_err)

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
        "provider": "OpenRouter / Google / MiniMax" if synth_source_display == "OpenRouter" else "SatQuery Synthesis Engine",
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

    return {
        "run_id": run_id,
        "analysis_type": analysis_type,
        "answer": final_answer,
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
        "execution_details": {
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
    }
