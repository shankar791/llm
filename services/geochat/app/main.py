"""
GeoChat Microservice — FastAPI application for Hugging Face Inference Endpoints.

Exposes:
  GET  /health  — HF Endpoint readiness probe
  POST /vqa     — Real multimodal Visual Question Answering

Model is loaded ONCE at startup via lifespan context manager.
All requests reuse the single loaded model instance.
"""
from __future__ import annotations
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse

from .runtime import GeoChatRuntime
from .inference import run_vqa
from .schemas import VQAResponse, HealthResponse, ErrorResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("geochat.service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the GeoChat model once at startup, keep it resident in GPU memory."""
    logger.info("=== GeoChat Service Starting ===")
    runtime = GeoChatRuntime.get_instance()
    try:
        load_info = runtime.load()
        logger.info(f"Model loaded successfully: {load_info}")
    except Exception as e:
        logger.error(f"FATAL: Model failed to load at startup: {e}")

    yield  # Service is running

    logger.info("=== GeoChat Service Shutting Down ===")


app = FastAPI(
    title="SatQuery AI — GeoChat VQA Service",
    description="Real multimodal Visual Question Answering for satellite imagery using GeoChat-7B on Hugging Face Inference Endpoints.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check whether the GeoChat model is loaded and ready for HF endpoint traffic."""
    runtime = GeoChatRuntime.get_instance()
    if not runtime.is_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "model_loaded": False,
                "model_class": "",
                "model_name": runtime.model_path,
                "device": runtime.device,
                "parameter_count": 0,
            },
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_class=runtime.load_info.get("model_class", ""),
        model_name=runtime.load_info.get("model_name", ""),
        device=runtime.load_info.get("device", ""),
        parameter_count=runtime.load_info.get("parameter_count", 0),
    )


@app.post("/vqa", response_model=VQAResponse, responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def vqa(
    image: UploadFile = File(..., description="Satellite image (PNG/JPEG/TIFF)"),
    question: str = Form(..., description="Natural-language question about the image"),
):
    """
    Execute real GeoChat VQA inference.

    Accepts a satellite image and a question, returns the model's generated answer.
    The image is preprocessed to 504x504 and passed as pixel_values to the multimodal model.
    """
    runtime = GeoChatRuntime.get_instance()
    if not runtime.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GeoChat model is not loaded. The service may still be initializing.",
        )

    if not question or not question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must not be empty.",
        )

    try:
        image_bytes = await image.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded image: {e}",
        )

    if not image_bytes or len(image_bytes) < 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or empty image file.",
        )

    # Execute real inference — no mock fallback
    try:
        result = run_vqa(image_bytes=image_bytes, question=question.strip())
    except Exception as e:
        logger.error(f"VQA inference failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {e}",
        )

    return VQAResponse(**result)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)},
    )
