"""
LangGraph node functions for the SatQuery AI Master Agent.
Implements controlled, deterministic, and auditable orchestration.
"""
from __future__ import annotations
import time
from typing import Any, Dict, List
from .state import AgentState
from ai.intent.classifier import RuleBasedIntentClassifier
from ai.compatibility.router import ToolCompatibilityRouter
from ai.synthesis.llm import LLMSynthesizer
from evidence.confidence import ConfidenceAggregator
from schemas.models import ToolResult, ToolRequest, RasterReference
from tools.registry import ToolRegistry


def validate_inputs_node(state: AgentState) -> dict:
    """
    Node 1: Validate uploaded images and determine input scenario.
    """
    start_time = time.time()
    query = state.get("query", "").strip()
    images = state.get("image_bytes", [])
    modalities = state.get("image_modalities", [])
    n = len(images)

    if not query:
        err = "Query string cannot be empty."
        return {
            "error": err,
            "is_compatible": False,
            "trace": state.get("trace", []) + [{
                "step": 1, "node": "validate_inputs", "status": "error",
                "detail": err, "duration_ms": int((time.time() - start_time) * 1000)
            }]
        }

    if n == 0:
        err = "No images provided for analysis."
        return {
            "error": err,
            "is_compatible": False,
            "trace": state.get("trace", []) + [{
                "step": 1, "node": "validate_inputs", "status": "error",
                "detail": err, "duration_ms": int((time.time() - start_time) * 1000)
            }]
        }

    if n == 1:
        scenario = "single_image"
    elif n == 2:
        if set(modalities) == {"optical", "sar"}:
            scenario = "cross_modal_pair"
        else:
            scenario = "bi_temporal_pair"
    else:
        err = f"Unsupported image count: expected 1 or 2, got {n}."
        return {
            "error": err,
            "is_compatible": False,
            "trace": state.get("trace", []) + [{
                "step": 1, "node": "validate_inputs", "status": "error",
                "detail": err, "duration_ms": int((time.time() - start_time) * 1000)
            }]
        }

    duration_ms = int((time.time() - start_time) * 1000)
    trace_entry = {
        "step": 1,
        "node": "validate_inputs",
        "status": "success",
        "scenario": scenario,
        "n_images": n,
        "duration_ms": duration_ms
    }
    return {
        "scenario": scenario,
        "n_images": n,
        "trace": state.get("trace", []) + [trace_entry]
    }


def classify_intent_node(state: AgentState) -> dict:
    """
    Node 2: Classify user intent into structured IntentSchema.
    """
    start_time = time.time()
    if state.get("error"):
        return {}

    query = state.get("query", "")
    n = state.get("n_images", 1)
    modalities = state.get("image_modalities", ["optical"])

    classifier = RuleBasedIntentClassifier()
    intent_result = classifier.classify(query=query, n_images=n, modalities=modalities)
    intent_schema = intent_result.to_schema()

    duration_ms = int((time.time() - start_time) * 1000)
    trace_entry = {
        "step": 2,
        "node": "classify_intent",
        "status": "success",
        "intent": intent_result.primary_task,
        "target": intent_result.target,
        "workflow": intent_result.workflow,
        "duration_ms": duration_ms
    }
    return {
        "intent": intent_result.primary_task,
        "intent_target": intent_result.target,
        "intent_schema": intent_schema.model_dump(),
        "intent_scores": intent_result.scores,
        "workflow": intent_result.workflow,
        "trace": state.get("trace", []) + [trace_entry]
    }


def compatibility_check_node(state: AgentState) -> dict:
    """
    Node 3: Validate whether query requirements are satisfied by available raster data.
    Runs BEFORE Master Agent tool routing.
    """
    start_time = time.time()
    if state.get("error"):
        return {}

    classifier = RuleBasedIntentClassifier()
    intent_result = classifier.classify(
        query=state.get("query", ""),
        n_images=state.get("n_images", 1),
        modalities=state.get("image_modalities", ["optical"])
    )

    router = ToolCompatibilityRouter()
    compat_res = router.check_compatibility(
        intent=intent_result,
        n_images=state.get("n_images", 1),
        modalities=state.get("image_modalities", ["optical"])
    )
    compat_schema = compat_res.to_schema()

    duration_ms = int((time.time() - start_time) * 1000)
    trace_entry = {
        "step": 3,
        "node": "compatibility_check",
        "status": "success" if compat_res.compatible else "incompatible",
        "explanation": compat_res.explanation,
        "duration_ms": duration_ms
    }

    out: Dict[str, Any] = {
        "compatibility": compat_schema.model_dump(),
        "is_compatible": compat_res.compatible,
        "trace": state.get("trace", []) + [trace_entry]
    }

    if not compat_res.compatible:
        out["error"] = compat_res.explanation

    return out


def master_router_node(state: AgentState) -> dict:
    """
    Node 4: Master Agent Routing & Decision Engine.
    Given verified intent, data compatibility, and state, decides which approved tool executes.
    Generates a structured Master Agent Decision Record.
    """
    start_time = time.time()
    if not state.get("is_compatible", True) or state.get("error"):
        return {"selected_tool": None}

    workflow = state.get("workflow", ["T1_VQA"])
    candidate_tool = workflow[0] if workflow else "T1_VQA"

    # Strict allowlist verification via ToolRegistry
    if not ToolRegistry.is_allowed(candidate_tool):
        err = f"Master Agent rejected unapproved tool request: {candidate_tool!r}."
        return {
            "error": err,
            "selected_tool": None,
            "trace": state.get("trace", []) + [{
                "step": 4, "node": "master_router", "status": "rejected",
                "detail": err, "duration_ms": int((time.time() - start_time) * 1000)
            }]
        }

    tool_def = ToolRegistry.get(candidate_tool)
    intent = state.get("intent", "vqa")
    intent_target = state.get("intent_target")

    decision_log = {
        "decision": "route",
        "selected_tool": candidate_tool,
        "reason": f"Structured intent task '{intent}' matches approved capability '{candidate_tool}' ({tool_def.name}). Data inputs verified compatible.",
        "inputs_verified": True,
        "timestamp": time.time()
    }

    # Construct standard ToolRequest
    filenames = state.get("image_filenames", [])
    modalities = state.get("image_modalities", [])
    raster_refs = [
        RasterReference(filename=fn, modality=mod).model_dump()
        for fn, mod in zip(filenames, modalities)
    ]

    tool_request = ToolRequest(
        tool_id=candidate_tool,
        query=state.get("query", ""),
        rasters=[RasterReference(**r) for r in raster_refs],
        parameters={"intent": intent, "target": intent_target},
        metadata=state.get("metadata", {})
    ).model_dump()

    duration_ms = int((time.time() - start_time) * 1000)
    trace_entry = {
        "step": 4,
        "node": "master_router",
        "status": "success",
        "decision": decision_log,
        "duration_ms": duration_ms
    }

    return {
        "selected_tool": candidate_tool,
        "tool_request": tool_request,
        "decision_log": decision_log,
        "tool_results": [],
        "trace": state.get("trace", []) + [trace_entry]
    }


def execute_specialist_tool_node(state: AgentState) -> dict:
    """
    Node 5: Execute the selected specialist tool via ToolRegistry.
    """
    start_time = time.time()
    selected_tool = state.get("selected_tool")
    if not selected_tool or state.get("error"):
        return {}

    query = state.get("query", "")
    images = state.get("image_bytes", [])
    modalities = state.get("image_modalities", ["optical"])
    results = list(state.get("tool_results", []))

    try:
        tool_instance = ToolRegistry.instantiate(selected_tool)
        if selected_tool == "T1_VQA":
            res = tool_instance.run(query=query, image_bytes=images, modalities=modalities)
        elif selected_tool == "T2_Caption":
            img = images[0] if images else b""
            mod = modalities[0] if modalities else "optical"
            res = tool_instance.run(image_bytes=img, modality=mod)
        elif selected_tool == "T3_Ground":
            img = images[0] if images else b""
            mod = modalities[0] if modalities else "optical"
            res = tool_instance.run(query=query, image_bytes=img, modality=mod)
        elif selected_tool == "T4_Change":
            img_t0 = images[0] if len(images) > 0 else b""
            img_t1 = images[1] if len(images) > 1 else b""
            res = tool_instance.run(image_bytes_t0=img_t0, image_bytes_t1=img_t1)
        elif selected_tool == "T5_OpticalSAR":
            img_opt = images[0] if len(images) > 0 else b""
            img_sar = images[1] if len(images) > 1 else b""
            res = tool_instance.run(optical_bytes=img_opt, sar_bytes=img_sar)
        else:
            raise ValueError(f"No execution handler for tool {selected_tool}")

        # Ensure mock flag is explicit
        if "metadata" not in res:
            res["metadata"] = {}
        res["metadata"]["mock"] = True
        res["metadata"]["status"] = "success"

        results.append(res)
        status = "success"
        error_msg = None

    except Exception as e:
        status = "error"
        error_msg = f"Specialist tool '{selected_tool}' execution failed: {e}"
        results.append({
            "tool_id": selected_tool,
            "answer": error_msg,
            "confidence": 0.0,
            "evidence": [],
            "metadata": {"mock": True, "status": "error", "error": str(e)}
        })

    duration_ms = int((time.time() - start_time) * 1000)
    trace_entry = {
        "step": 5,
        "node": "execute_specialist_tool",
        "status": status,
        "tool_id": selected_tool,
        "duration_ms": duration_ms
    }

    out: Dict[str, Any] = {
        "tool_results": results,
        "trace": state.get("trace", []) + [trace_entry]
    }
    if error_msg:
        out["error"] = error_msg
    return out


def standardize_results_node(state: AgentState) -> dict:
    """
    Node 6: Validate all raw tool results against ToolResult schema.
    """
    start_time = time.time()
    raw_results = state.get("tool_results", [])
    standardized: List[Dict[str, Any]] = []

    for r in raw_results:
        validated = ToolResult.model_validate(r)
        standardized.append(validated.model_dump())

    duration_ms = int((time.time() - start_time) * 1000)
    trace_entry = {
        "step": 6,
        "node": "standardize_results",
        "status": "success",
        "count": len(standardized),
        "duration_ms": duration_ms
    }
    return {"tool_results": standardized, "trace": state.get("trace", []) + [trace_entry]}


def gis_processor_node(state: AgentState) -> dict:
    """
    Node 7: Deterministic geospatial processing assembling canonical GeoJSON.
    """
    start_time = time.time()
    results = state.get("tool_results", [])
    features = []

    for res in results:
        for ev in res.get("evidence", []):
            if ev.get("geojson_feature"):
                features.append(ev["geojson_feature"])

    geojson = {
        "type": "FeatureCollection",
        "features": features
    } if features else None

    duration_ms = int((time.time() - start_time) * 1000)
    trace_entry = {
        "step": 7,
        "node": "gis_processor",
        "status": "success",
        "features_assembled": len(features),
        "duration_ms": duration_ms
    }
    return {"geojson": geojson, "trace": state.get("trace", []) + [trace_entry]}


def evidence_confidence_node(state: AgentState) -> dict:
    """
    Node 8: Aggregate calibrated confidence scores via ConfidenceAggregator.
    """
    start_time = time.time()
    results = state.get("tool_results", [])

    if not results or state.get("error"):
        confidence = 0.0
    else:
        aggregator = ConfidenceAggregator()
        confidence = aggregator.aggregate(results)

    duration_ms = int((time.time() - start_time) * 1000)
    trace_entry = {
        "step": 8,
        "node": "evidence_confidence",
        "status": "success",
        "confidence": round(confidence, 3),
        "duration_ms": duration_ms
    }
    return {"confidence": round(confidence, 3), "trace": state.get("trace", []) + [trace_entry]}


def llm_synthesis_node(state: AgentState) -> dict:
    """
    Node 9: Synthesize final grounded natural-language response.
    """
    start_time = time.time()
    synthesizer = LLMSynthesizer()
    final_answer = synthesizer.synthesize(
        query=state.get("query", ""),
        tool_results=state.get("tool_results", []),
        confidence=state.get("confidence", 0.0),
        geojson=state.get("geojson"),
        error=state.get("error")
    )

    duration_ms = int((time.time() - start_time) * 1000)
    trace_entry = {
        "step": 9,
        "node": "llm_synthesis",
        "status": "success",
        "answer_len": len(final_answer),
        "duration_ms": duration_ms
    }
    return {"final_answer": final_answer, "trace": state.get("trace", []) + [trace_entry]}
