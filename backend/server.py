"""FastAPI backend for SatQuery AI with 3D Experience, Mission Dashboard & Live Pipeline Monitor."""
from __future__ import annotations

import io
import json
import mimetypes
import sys
import uuid
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Ensure MIME types for 3D engine assets
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".basis")
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from .rasterio_utils import RasterInput
    from .agent import execute, execute_followup
    from .history import history_store
    from .session import session_store
    from .chat import execute_chat_turn
except ImportError:
    # Running directly as uvicorn server:app inside backend/
    sys.path.append(str(Path(__file__).parent))
    from rasterio_utils import RasterInput
    from agent import execute, execute_followup
    from history import history_store
    from session import session_store
    from chat import execute_chat_turn

app = FastAPI(title="SatQuery AI", version="1.0")

# Enable CORS for local dev / browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_3D = ROOT_DIR / "frontend"
DASHBOARD_DIR = ROOT_DIR / "satquery-frontend-dashboard"
STATIC = Path(__file__).resolve().parent / "static"
REAL_DATA = Path(__file__).resolve().parent / "real_data"
TEST_IMAGES = Path(__file__).resolve().parent / "test_images"

# Mount 3D assets subfolder if present
if (FRONTEND_3D / "files").exists():
    app.mount("/files", StaticFiles(directory=str(FRONTEND_3D / "files")), name="frontend_3d_files")


# ---------------------------------------------------------------- Unified Web Interfaces
@app.get("/", response_class=HTMLResponse)
async def entry_3d_experience():
    """Serve the 3D Satellite Orbital Entry experience."""
    target = FRONTEND_3D / "index.html"
    if target.exists():
        return target.read_text(encoding="utf-8")
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/mission", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
async def mission_dashboard():
    """Serve the imported high-performance SatQuery AI Geospatial Dashboard."""
    target = DASHBOARD_DIR / "index.html"
    if target.exists():
        return target.read_text(encoding="utf-8")
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/monitor", response_class=HTMLResponse)
async def pipeline_monitor():
    """Serve the legacy 11-stage vertical execution timeline monitor."""
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "satquery-ai"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Prevent 404 logs for browser favicon requests."""
    return Response(status_code=204)


# ---------------------------------------------------------------- Sample Imagery
@app.get("/api/samples/{filename}")
async def get_sample(filename: str):
    """Serve sample satellite imagery for one-click testing in UI."""
    for folder in [REAL_DATA, TEST_IMAGES]:
        target = folder / filename
        if target.exists() and target.is_file():
            return FileResponse(target, media_type="image/png")
    return {"error": f"Sample image '{filename}' not found"}


# ---------------------------------------------------------------- Core SatQuery AI Analysis Endpoint
@app.post("/api/query")
async def query_api(
    request: Request,
    query: Optional[str] = Form(None),
    files: Optional[list[UploadFile]] = File(None),
    run_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    mode: Optional[str] = Form(None),
    existing_analysis: Optional[str] = Form(None),
):
    # Support direct JSON payload for follow-up questions
    content_type = request.headers.get("content-type", "")
    existing_analysis_data = existing_analysis
    if content_type.startswith("application/json"):
        try:
            body = await request.json()
            if not query:
                query = body.get("query")
            if not run_id:
                run_id = body.get("run_id")
            if not session_id:
                session_id = body.get("session_id")
            if not mode:
                mode = body.get("mode")
            if not existing_analysis_data:
                existing_analysis_data = body.get("existing_analysis")
        except Exception:
            pass

    if not run_id:
        run_id = str(uuid.uuid4())

    if not query or not query.strip():
        query = "Analyze this satellite imagery and describe major land-cover or surface patterns."

    # -----------------------------------------------------------
    # CASE 1: FOLLOW-UP QUERY (session_id or existing_analysis provided and no new files uploaded)
    # -----------------------------------------------------------
    if (session_id or existing_analysis_data) and (not files or len(files) == 0):
        # Create history entry in RUNNING state
        history_store.create_entry(
            run_id=run_id,
            query=query,
            image_names=[],
            analysis_type="Follow-Up Synthesis...",
            status="RUNNING",
        )

        try:
            result = execute_followup(
                query,
                session_id=session_id,
                run_id=run_id,
                existing_analysis=existing_analysis_data,
            )
            if "error" in result and not result.get("answer"):
                history_store.update_entry(
                    run_id=run_id,
                    status="FAILED",
                    error=result["error"],
                    full_result=result,
                )
                if result.get("code") == "SESSION_NOT_FOUND":
                    from fastapi.responses import JSONResponse
                    return JSONResponse(status_code=404, content=result)
                return result

            # Update history entry to SUCCESS
            tools_run = [o.get("tool_id", o.get("tool")) for o in result.get("outputs", [])]
            models_used = result.get("execution_details", {}).get("models", [])
            history_store.update_entry(
                run_id=run_id,
                status="SUCCESS",
                tools_executed=tools_run,
                models_used=models_used,
                result_summary=result.get("answer", "")[:160] + "..." if len(result.get("answer", "")) > 160 else result.get("answer", ""),
                full_result=result,
                analysis_type=result.get("analysis_type", "Follow-Up Analysis"),
            )
            return result

        except HTTPException:
            raise
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            history_store.update_entry(
                run_id=run_id,
                status="FAILED",
                error=err_msg,
                full_result={"error": err_msg, "run_id": run_id, "session_id": session_id},
            )
            return {"error": err_msg, "run_id": run_id, "session_id": session_id}

    # -----------------------------------------------------------
    # CASE 2: FRESH TEXT PROMPT (no files uploaded and no previous session/analysis)
    # -----------------------------------------------------------
    if not files or len(files) == 0:
        result = followup_api({
            "question": query,
            "context": None,
            "run_id": run_id,
        })
        result["run_id"] = run_id
        result["session_id"] = run_id
        return result

    # -----------------------------------------------------------
    # CASE 3: INITIAL IMAGE ANALYSIS (new files uploaded)
    # -----------------------------------------------------------
    if not session_id:
        session_id = run_id

    filenames = [f.filename or f"image_{i}.png" for i, f in enumerate(files)]
    rasters = []
    for i, f in enumerate(files):
        data = await f.read()
        rasters.append(RasterInput(f.filename or f"image_{i}.png", data))

        if len(rasters) == 2:
            if mode in ("sar", "fusion"):
                # MODE 2: Optical + SAR Pair
                # Image 1 = OPTICAL, Image 2 = SAR
                rasters[0].modality = "optical"
                rasters[1].modality = "sar"
            elif mode in ("multi", "bitemporal", "change"):
                # MODE 1: Before / After Pair
                # Image 1 = BEFORE / T0, Image 2 = AFTER / T1
                rasters[0].modality = "optical"
                rasters[1].modality = "optical"

    # 1. Create history entry in RUNNING state
    history_store.create_entry(
        run_id=run_id,
        query=query,
        image_names=filenames,
        analysis_type="Processing...",
        status="RUNNING",
    )

    try:
        result = execute(query, rasters, run_id=run_id, session_id=session_id)
        result["run_id"] = run_id
        result["session_id"] = session_id
        if "error" in result and not result.get("answer"):
            history_store.update_entry(
                run_id=run_id,
                status="FAILED",
                error=result["error"],
                full_result=result,
            )
            return result

        # 2. Update history record to SUCCESS
        analysis_type = result.get("analysis_type", "Geospatial Analysis")
        tools_run = [o.get("tool_id", o.get("tool")) for o in result.get("outputs", [])]
        models_used = result.get("execution_details", {}).get("models", [])

        history_store.update_entry(
            run_id=run_id,
            status="SUCCESS",
            tools_executed=tools_run,
            models_used=models_used,
            result_summary=result.get("answer", "")[:160] + "..." if len(result.get("answer", "")) > 160 else result.get("answer", ""),
            full_result=result,
            analysis_type=analysis_type,
        )
        return result

    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        history_store.update_entry(
            run_id=run_id,
            status="FAILED",
            error=err_msg,
            full_result={"error": err_msg, "run_id": run_id, "session_id": session_id},
        )
        return {"error": err_msg, "run_id": run_id, "session_id": session_id}


# ---------------------------------------------------------------- Direct Follow-Up & Text LLM Endpoint
def _get_general_text_fallback(query: str, fallback_reason: str = "") -> str:
    """Factual remote-sensing domain fallback when upstream LLM is unreachable."""
    q_low = query.lower()
    if "ndvi" in q_low:
        return (
            "NDVI (Normalized Difference Vegetation Index) is a standardized spectral index used to quantify "
            "vegetation greenness, density, and plant health from satellite imagery. It is calculated using the formula:\n\n"
            "$$\\text{NDVI} = \\frac{\\text{NIR} - \\text{Red}}{\\text{NIR} + \\text{Red}}$$\n\n"
            "where NIR is near-infrared reflectance (strongly reflected by healthy plant cell structures) "
            "and Red is visible red reflectance (absorbed by chlorophyll for photosynthesis). Typical values range from -1.0 to +1.0:\n"
            "- 0.4 to 0.8: Dense, healthy green vegetation (forests, active cropland)\n"
            "- 0.2 to 0.4: Sparse or senescent vegetation, shrublands, grasslands\n"
            "- 0.0 to 0.2: Bare soil, rock, sand, or urban surfaces\n"
            "- Negative values (< 0): Open water bodies, snow, or clouds."
        )
    elif "ndwi" in q_low:
        return (
            "NDWI (Normalized Difference Water Index) is a remote sensing index used to delineate open water bodies "
            "and monitor water content in surface features. The McFeeters index is defined as:\n\n"
            "$$\\text{NDWI} = \\frac{\\text{Green} - \\text{NIR}}{\\text{Green} + \\text{NIR}}$$\n\n"
            "Positive values typically represent water surfaces, while terrestrial vegetation and soil produce negative values."
        )
    elif "sar" in q_low or "radar" in q_low:
        return (
            "SAR (Synthetic Aperture Radar) is an active microwave remote sensing system that transmits radar pulses "
            "and measures backscatter intensity and phase. Because microwave signals penetrate clouds, rain, and haze, "
            "SAR operates effectively day and night. Common polarizations include VV (co-polarized) and VH (cross-polarized), "
            "providing key insights into surface roughness, structure, and dielectric properties."
        )
    else:
        return (
            f"SatQuery AI Earth Observation Assistant: Received query '{query}'. "
            "No satellite imagery context was attached. Upstream TokenRouter LLM offline fallback engaged."
        )


@app.post("/api/followup")
def followup_api(payload: dict):
    """
    Direct LLM follow-up reasoning over existing image analysis context (Scenario 1)
    or direct fresh text query (Scenario 2).
    Executes ONLY the TokenRouter LLM without re-running any specialist vision tools or re-uploading images.
    """
    question = (payload.get("question") or payload.get("query") or "").strip()
    context = payload.get("context") or {}
    run_id = payload.get("run_id") or str(uuid.uuid4())
    session_id = payload.get("session_id")

    if not question:
        raise HTTPException(status_code=400, detail="Missing question.")

    # Determine whether previous image analysis context is present (Scenario 1 vs Scenario 2)
    has_image_context = bool(
        context and isinstance(context, dict) and any(
            context.get(k) for k in ("evidence", "answer", "outputs", "sections", "claims", "image")
        )
    )

    if has_image_context:
        # SCENARIO 1: Previous Image Analysis Exists
        prompt = f"""You are answering a follow-up question about a satellite image that has already been analyzed.

Use ONLY the supplied existing analysis/evidence.

Existing analysis:
{json.dumps(context, indent=2, default=str)}

Follow-up question:
{question}

Answer the question directly.

Do not invent:
- coordinates
- areas
- percentages
- confidence values
- detected objects
- measurements
- change statistics

If the existing analysis does not contain enough information to answer the question, say so clearly.

Do not claim that a new image analysis was performed."""

        system_prompt = (
            "You are the SatQuery AI Synthesis Engine answering follow-up questions about satellite imagery. "
            "CRITICAL INSTRUCTIONS & ZERO-REASONING POLICY:\n"
            "- Output ONLY the final response intended for the user. NEVER output internal reasoning, planning steps, or chain-of-thought.\n"
            "- NEVER write 'Here\'s a thinking process:', 'Thinking:', 'Analyze User Input', 'Draft Response', or numbered steps.\n"
            "- Base your answers strictly on the supplied existing analysis and evidence. Never invent statistics or claim new analysis was run.\n"
            "- Format responses cleanly with short headings, bullet points for observations, and bold findings."
        )
        analysis_type = "Follow-Up Analysis"
        task_name = "Follow-Up Synthesis"
        justification = "Direct LLM synthesis using existing analysis context."
    else:
        # SCENARIO 2: Fresh Text Prompt (no image context)
        prompt = question
        system_prompt = (
            "You are SatQuery AI, an expert conversational assistant specializing in Earth observation, "
            "satellite remote sensing, GIS, and geospatial intelligence.\n"
            "CRITICAL INSTRUCTIONS & ZERO-REASONING POLICY:\n"
            "- Output ONLY the direct final response intended for the user. NEVER output reasoning, planning, or chain-of-thought.\n"
            "- NEVER write 'Here\'s a thinking process:', 'Thinking:', 'Analyze User Input', 'Determine Appropriate Response', 'Draft Response', or numbered planning steps.\n"
            "- Begin immediately with the user-facing message.\n"
            "- When the user introduces themselves (e.g. 'my name is shankar'), greet them warmly by name (e.g. 'Nice to meet you, Shankar! 👋') and state how you can help them with satellite imagery, remote sensing, or GIS.\n"
            "- Answer factually without inventing measurements or coordinates."
        )
        analysis_type = "Direct LLM Query"
        task_name = "Direct LLM Query"
        justification = "Direct TokenRouter LLM reasoning with no image analysis context."

    # Call existing TokenRouter LLM directly (NO GeoChat, NO VQA, NO Caption, NO ChangeFormer, NO GIS)
    answer_text = ""
    llm_source = "TokenRouter GLM-5.3-Free"
    fallback_used = False
    fallback_reason = None

    try:
        from ai.llm.provider import get_llm_provider
        provider = get_llm_provider()
        model_name = getattr(provider.config, "model", "z-ai/glm-5.3-free")
        llm_source = f"TokenRouter ({model_name})"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        resp = provider.generate_sync(messages, temperature=0.0, max_tokens=1024)
        raw_content = (resp.content or "").strip()

        try:
            from .chat import clean_chat_response
        except ImportError:
            try:
                from chat import clean_chat_response
            except ImportError:
                from backend.chat import clean_chat_response

        answer_text = clean_chat_response(raw_content)

    except Exception as e:
        fallback_used = True
        fallback_reason = f"{type(e).__name__}: {e}"
        if has_image_context:
            from ai.synthesis.fallback import DeterministicFallbackFormatter
            fb = DeterministicFallbackFormatter()
            fb_res = fb.format(
                query=question,
                tool_results=context.get("outputs") or [],
                existing_evidence=context.get("evidence") or [],
                fallback_reason=str(e),
            )
            answer_text = fb_res.answer
        else:
            answer_text = _get_general_text_fallback(question, str(e))

    if not answer_text:
        if has_image_context:
            prev_ans = context.get("answer") or "Previous scene observation recorded."
            answer_text = f"Based on the previous analysis:\n{prev_ans}"
        else:
            answer_text = _get_general_text_fallback(question)

    if has_image_context:
        from ai.synthesis.formatter import format_vlm_presentation
        answer_text = format_vlm_presentation(answer_text, query=question)

    result_payload = {
        "run_id": run_id,
        "session_id": session_id or (context.get("session_id") if has_image_context else run_id),
        "answer": answer_text,
        "sections": context.get("sections") or [] if has_image_context else [],
        "claims": context.get("claims") or [] if has_image_context else [],
        "uncertainties": (context.get("uncertainties") or [
            "Follow-up derived from prior analysis context without re-executing vision models."
        ]) if has_image_context else [],
        "justification": justification,
        "analysis_type": analysis_type,
        "evidence": context.get("evidence") or [] if has_image_context else [],
        "execution_details": {
            "models": [{
                "task": task_name,
                "actual_model": llm_source,
                "source": "TokenRouter" if not fallback_used else "Deterministic Fallback",
                "status": "SUCCESS" if not fallback_used else "FALLBACK",
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
            }],
            "intent": {
                "specialist_executed": False,
                "tools_run": []
            }
        }
    }

    try:
        history_store.create_entry(
            run_id=run_id,
            query=question,
            image_names=[],
            analysis_type=analysis_type,
            status="SUCCESS" if not fallback_used else "SUCCESS (FALLBACK)",
            tools_executed=[],
            models_used=[{
                "task": task_name,
                "actual_model": llm_source,
                "source": "TokenRouter" if not fallback_used else "Deterministic Fallback",
                "status": "SUCCESS" if not fallback_used else "FALLBACK",
            }],
            result_summary=answer_text[:160] + "..." if len(answer_text) > 160 else answer_text,
            full_result=result_payload,
        )
    except Exception:
        pass

    return result_payload


# ---------------------------------------------------------------- Step 3 Chat Endpoint
@app.post("/api/chat")
def chat_endpoint(payload: dict):
    """
    Step 3 Chat Endpoint:
    Connects session context to existing LLM provider.
    Executes a chat turn using session_id, user query, and context builder.
    """
    session_id = payload.get("session_id")
    query = (payload.get("query") or payload.get("message") or payload.get("question") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id.")
    if not query:
        raise HTTPException(status_code=400, detail="Missing query.")

    try:
        from backend.chat import execute_chat_turn
    except ImportError:
        from chat import execute_chat_turn

    try:
        result = execute_chat_turn(session_id=session_id, query=query)
        return result
    except Exception as e:
        sess = session_store.get_session(session_id)
        has_context = bool(sess and (sess.get("evidence") or sess.get("last_analysis")))
        if has_context:
            from ai.synthesis.fallback import DeterministicFallbackFormatter
            fb = DeterministicFallbackFormatter()
            fb_res = fb.format(
                query=query,
                tool_results=(sess.get("last_analysis") or {}).get("outputs", []),
                existing_evidence=sess.get("evidence", []),
                fallback_reason=str(e),
            )
            answer_text = fb_res.answer
        else:
            answer_text = _get_general_text_fallback(query, str(e))

        from ai.synthesis.formatter import format_vlm_presentation
        answer_text = format_vlm_presentation(answer_text, query=query)
        session_store.add_assistant_message(session_id, answer_text)

        return {
            "session_id": session_id,
            "query": query,
            "answer": answer_text,
            "response": answer_text,
            "model": "Deterministic Fallback",
            "provider": "local_fallback",
            "fallback": True,
            "error": str(e)
        }


# ---------------------------------------------------------------- Analysis Session Endpoints
@app.get("/api/session/{session_id}")
async def get_session_endpoint(session_id: str):
    """Retrieve an active analysis session including its conversation thread and evidence."""
    sess = session_store.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail=f"Analysis session '{session_id}' not found")
    return sess


@app.delete("/api/session/{session_id}")
async def delete_session_endpoint(session_id: str):
    """Delete an analysis session and its cached imagery."""
    deleted = session_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Analysis session '{session_id}' not found")
    return {"deleted": True, "session_id": session_id}


# ---------------------------------------------------------------- Analysis History Endpoints
@app.get("/api/history")
async def get_history(
    q: Optional[str] = Query(None, description="Search term for filtering history"),
    limit: int = Query(50, ge=1, le=200),
):
    """List recent analysis history sessions with optional search filtering."""
    return history_store.get_history(search=q, limit=limit)


@app.get("/api/history/{run_id}")
async def get_history_by_id(run_id: str):
    """Retrieve full cached analysis result for a specific run without re-running."""
    record = history_store.get_run(run_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Analysis session with ID '{run_id}' not found")
    return record


@app.delete("/api/history/{run_id}")
async def delete_history_item(run_id: str):
    """Delete a specific analysis run from history."""
    deleted = history_store.delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Analysis session with ID '{run_id}' not found")
    return {"deleted": True, "run_id": run_id}


@app.delete("/api/history")
async def clear_all_history():
    """Clear all analysis history records."""
    history_store.clear()
    return {"cleared": True}


# ---------------------------------------------------------------- Static Asset Fallback (3D & Dashboard files)
@app.get("/{filename:path}")
async def serve_static_root_assets(filename: str):
    """Serve root-level static scripts, 3D configs, styles, and assets."""
    if filename.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found")

    # Check in 3D frontend directory
    target_3d = FRONTEND_3D / filename
    if target_3d.exists() and target_3d.is_file():
        media_type, _ = mimetypes.guess_type(str(target_3d))
        if not media_type:
            if target_3d.suffix == ".wasm":
                media_type = "application/wasm"
            elif target_3d.suffix == ".basis":
                media_type = "application/octet-stream"
            elif target_3d.suffix == ".glb":
                media_type = "model/gltf-binary"
        return FileResponse(target_3d, media_type=media_type)

    # Check in Dashboard directory
    dash_target = DASHBOARD_DIR / filename
    if dash_target.exists() and dash_target.is_file():
        media_type, _ = mimetypes.guess_type(str(dash_target))
        return FileResponse(dash_target, media_type=media_type)

    raise HTTPException(status_code=404, detail=f"Resource '{filename}' not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
