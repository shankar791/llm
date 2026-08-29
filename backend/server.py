"""FastAPI backend for SatQuery AI."""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from rasterio_utils import RasterInput
from agent import execute

app = FastAPI(title="SatQuery AI", version="1.0")

STATIC = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/api/query")
async def query_api(query: str = Form(...), files: list[UploadFile] = File(...)):
    rasters = []
    for f in files:
        data = await f.read()
        rasters.append(RasterInput(f.filename, data))
    try:
        result = execute(query, rasters)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
