"""
Base interfaces and data structures for the SatQuery AI Vision subsystem.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable
from pydantic import BaseModel, Field, model_validator

TaskType = Literal["vqa", "caption", "ground"]


class GroundingBox(BaseModel):
    """
    A single localized object bounding box.
    Box format: [x0, y0, x1, y1] normalized to [0.0, 1.0] or in image pixels.
    """
    label: str = Field(..., description="Semantic entity class or object label")
    box: List[float] = Field(..., min_length=4, max_length=4, description="Bounding coordinates [x0, y0, x1, y1]")
    confidence: Optional[float] = Field(default=None, description="Optional detection confidence")

    @model_validator(mode="after")
    def validate_coordinates(self) -> GroundingBox:
        x0, y0, x1, y1 = self.box
        if not self.label or not self.label.strip():
            raise ValueError("GroundingBox label cannot be empty")
        if x0 > x1:
            raise ValueError(f"Invalid x coordinates: x0 ({x0}) > x1 ({x1})")
        if y0 > y1:
            raise ValueError(f"Invalid y coordinates: y0 ({y0}) > y1 ({y1})")
        return self

    def to_pixel_box(self, width: int, height: int) -> List[int]:
        """
        Convert coordinates to integer [ymin, xmin, ymax, xmax] pixel format.
        Handles both normalized [0.0, 1.0] and raw pixel inputs safely.
        """
        x0, y0, x1, y1 = self.box
        
        # If coordinates are normalized in [0.0, 1.0] (with small margin for float noise)
        if max(x0, y0, x1, y1) <= 1.05:
            xmin = int(round(max(0.0, min(1.0, x0)) * width))
            ymin = int(round(max(0.0, min(1.0, y0)) * height))
            xmax = int(round(max(0.0, min(1.0, x1)) * width))
            ymax = int(round(max(0.0, min(1.0, y1)) * height))
        else:
            # Already in pixel space
            xmin = int(round(max(0, min(width, x0))))
            ymin = int(round(max(0, min(height, y0))))
            xmax = int(round(max(0, min(width, x1))))
            ymax = int(round(max(0, min(height, y1))))

        return [ymin, xmin, ymax, xmax]


class GroundingResult(BaseModel):
    """Container for structured object localization results."""
    objects: List[GroundingBox] = Field(default_factory=list, description="List of detected object bounding boxes")


@dataclass
class VisionResponse:
    """Standardized response from any VisionProvider."""
    text: str
    grounding: Optional[GroundingResult] = None
    raw_json: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    provider: str = ""
    model: str = ""
    selected_model: str = ""
    attempted_models: List[str] = field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: Optional[str] = None


@runtime_checkable
class VisionProvider(Protocol):
    """
    Protocol defining the vendor-agnostic remote vision interface.
    Accepts raw images + prompts, returns canonical VisionResponse.
    """

    async def analyze_image(
        self,
        image_input: Any,
        prompt: str,
        *,
        task: TaskType = "vqa",
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> VisionResponse:
        """Analyze an image asynchronously."""
        ...

    def analyze_image_sync(
        self,
        image_input: Any,
        prompt: str,
        *,
        task: TaskType = "vqa",
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> VisionResponse:
        """Analyze an image synchronously."""
        ...
