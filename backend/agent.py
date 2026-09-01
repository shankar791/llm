"""
Agentic controller for SatQuery AI.

Interprets the query + input configuration, routes to specialist tools,
fuses outputs, and produces an auditable execution trace.
"""
from __future__ import annotations

import time
import uuid
import re

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
        # VQA is the mandatory default for single images unless the user
        # explicitly asked for grounding; captioning rides along as extra.
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
                vqa_score += 0.5   # still answer via VQA+caption pipeline
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
def execute(query: str, rasters: list[RasterInput]) -> dict:
    t_start = time.time()
    trace_id = str(uuid.uuid4())[:8]

    trace = {
        "trace_id": trace_id,
        "query": query,
        "steps": [],
    }

    # Step 1 — validate inputs
    try:
        scenario = validate_inputs(rasters)
    except ValueError as e:
        return {"error": str(e), "trace": trace}
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

    valid_confs = [o["confidence"] for o in outputs if o.get("confidence") is not None]
    confidence = min(valid_confs) if valid_confs else 0.6

    try:
        from ai.synthesis.llm import LLMSynthesizer
        synthesizer = LLMSynthesizer()
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

    trace["steps"].append({
        "step": 4,
        "action": "compose_answer",
        "confidence": round(confidence, 3),
        "synthesis_source": syn_source,
        "fallback_used": fb_used,
        "fallback_reason": fb_reason,
        "total_ms": round((time.time() - t_start) * 1000),
    })

    return {
        "answer": final_answer,
        "confidence": round(confidence, 3),
        "outputs": outputs,
        "evidence_images_b64": [o["evidence_image_b64"] for o in outputs
                                if o.get("evidence_image_b64")],
        "synthesis_source": syn_source,
        "fallback_used": fb_used,
        "fallback_reason": fb_reason,
        "trace": trace,
        "scenario": scenario["scenario"],
    }

