"""FastAPI backend for SatQuery AI."""
from __future__ import annotations

import io
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from .rasterio_utils import RasterInput
    from .agent import execute
except ImportError:
    # Running directly as uvicorn server:app inside backend/
    sys.path.append(str(Path(__file__).parent))
    from rasterio_utils import RasterInput
    from agent import execute

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SatQuery AI", version="1.0")

# Enable CORS for local dev / browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC = Path(__file__).parent / "static"
REAL_DATA = Path(__file__).parent / "real_data"
TEST_IMAGES = Path(__file__).parent / "test_images"


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/samples/{filename}")
async def get_sample(filename: str):
    """Serve sample satellite imagery for one-click testing in UI."""
    for folder in [REAL_DATA, TEST_IMAGES]:
        target = folder / filename
        if target.exists() and target.is_file():
            return FileResponse(target, media_type="image/png")
    return {"error": f"Sample image '{filename}' not found"}


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
