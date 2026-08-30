"""
GeoChat Service — Request/Response schemas for Hugging Face Inference Endpoint.
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class VQAResponse(BaseModel):
    """Canonical VQA response from the GeoChat service."""
    task: str = "vqa"
    model: str = "GeoChat-7B"
    answer: str
    latency_ms: float
    image_size: List[int] = Field(description="Original [width, height]")
    processed_size: List[int] = Field(default=[504, 504], description="Model input [width, height]")
    mode: str = "real"


class HealthResponse(BaseModel):
    """Health check response conforming to HF endpoint monitoring."""
    status: str = "ok"
    model_loaded: bool = False
    model_class: str = ""
    model_name: str = ""
    device: str = ""
    parameter_count: int = 0


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
