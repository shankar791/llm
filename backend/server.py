"""FastAPI backend for SatQuery AI with 3D Experience, Mission Dashboard & Live Pipeline Monitor."""
from __future__ import annotations

import io
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
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Ensure MIME types for 3D engine assets
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".basis")
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

try:
    from .rasterio_utils import RasterInput
    from .agent import execute, execute_followup
    from .history import history_store
    from .session import session_store
except ImportError:
    # Running directly as uvicorn server:app inside backend/
    sys.path.append(str(Path(__file__).parent))
    from rasterio_utils import RasterInput
    from agent import execute, execute_followup
    from history import history_store
    from session import session_store

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
):
    # Support direct JSON payload for follow-up questions
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        try:
            body = await request.json()
            if not query:
                query = body.get("query")
            if not run_id:
                run_id = body.get("run_id")
            if not session_id:
                session_id = body.get("session_id")
        except Exception:
            pass

    if not run_id:
        run_id = str(uuid.uuid4())

    if not query or not query.strip():
        query = "Analyze this satellite imagery and describe major land-cover or surface patterns."

    # -----------------------------------------------------------
    # CASE 1: FOLLOW-UP QUERY (session_id provided and no new files uploaded)
    # -----------------------------------------------------------
    if session_id and (not files or len(files) == 0):
        sess = session_store.get_session(session_id)
        if not sess:
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"Analysis session '{session_id}' not found or expired.",
                    "detail": f"Analysis session '{session_id}' not found or expired. Please start a new analysis.",
                    "code": "SESSION_NOT_FOUND",
                    "session_id": session_id,
                    "run_id": run_id,
                },
            )

        # Create history entry in RUNNING state
        history_store.create_entry(
            run_id=run_id,
            query=query,
            image_names=sess.get("image", {}).get("filenames", []),
            analysis_type="Follow-Up Processing...",
            status="RUNNING",
        )

        try:
            result = execute_followup(query, session_id=session_id, run_id=run_id)
            if "error" in result and not result.get("answer"):
                history_store.update_entry(
                    run_id=run_id,
                    status="FAILED",
                    error=result["error"],
                    full_result=result,
                )
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
    # CASE 2: INITIAL QUERY (or starting a new session with new files)
    # -----------------------------------------------------------
    if not session_id:
        session_id = run_id

    if not files or len(files) == 0:
        sample_path = REAL_DATA / "opt_0611.png"
        if not sample_path.exists():
            sample_path = TEST_IMAGES / "optical_t0.png"
        if sample_path.exists():
            data = sample_path.read_bytes()
            rasters = [RasterInput("opt_0611.png", data)]
            filenames = ["opt_0611.png"]
        else:
            return {"error": "No image files provided and default sample imagery not found.", "run_id": run_id}
    else:
        filenames = [f.filename or "image.png" for f in files]
        rasters = []
        for f in files:
            data = await f.read()
            rasters.append(RasterInput(f.filename or "image.png", data))

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
