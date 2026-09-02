"""
GeoChat Vision Provider implementation for SatQuery AI.
Connects to the live GeoChat microservice (POST /chat) for remote-sensing VQA and captioning.
Strictly implements the VisionProvider protocol.
"""
from __future__ import annotations
import asyncio
import io
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import requests

from .base import TaskType, VisionProvider, VisionResponse
from .config import VisionConfig
from .errors import (
    VisionAuthenticationError,
    VisionError,
    VisionNetworkError,
    VisionRateLimitError,
    VisionResponseError,
    VisionTimeoutError,
)

logger = logging.getLogger("satquery.vision.geochat")


def _image_to_bytes(image_input: Union[bytes, np.ndarray, Image.Image]) -> bytes:
    """Normalize input image to PNG/JPEG bytes for multipart HTTP transmission."""
    if isinstance(image_input, bytes):
        if not image_input:
            raise ValueError("Input image bytes cannot be empty.")
        return image_input

    if isinstance(image_input, Image.Image):
        buf = io.BytesIO()
        img = image_input.convert("RGB")
        img.save(buf, format="PNG")
        return buf.getvalue()

    if isinstance(image_input, np.ndarray):
        arr = image_input
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[2] not in {1, 3, 4}:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            a = arr.astype(np.float32)
            lo, hi = np.percentile(a, [2, 98])
            if hi <= lo:
                hi = lo + 1.0
            a = np.clip((a - lo) / (hi - lo), 0, 1)
            arr = (a * 255).astype(np.uint8)
        img = Image.fromarray(arr[:, :, :3]).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    raise ValueError(f"Unsupported image input type: {type(image_input)}")


class GeoChatVisionProvider:
    """
    Vendor-agnostic VisionProvider adapter for the external GeoChat remote-sensing VLM microservice.
    Dispatches multipart requests to POST {base_url}/chat and normalizes responses into VisionResponse.
    """

    def __init__(
        self,
        config: Optional[VisionConfig] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.config = config or VisionConfig.from_env()
        self.base_url = (
            base_url
            or getattr(self.config, "geochat_base_url", None)
            or os.environ.get("GEOCHAT_BASE_URL", "http://172.25.32.36:8000")
        ).rstrip("/")
        self.api_key = (
            api_key
            or getattr(self.config, "geochat_api_key", None)
            or os.environ.get("GEOCHAT_API_KEY", "")
        )
        self.timeout = timeout or getattr(self.config, "timeout", 120.0)
        self._endpoint = f"{self.base_url}/chat"

    def analyze_image_sync(
        self,
        image_input: Any,
        prompt: str,
        *,
        task: TaskType = "vqa",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> VisionResponse:
        """
        Synchronously dispatch an image and natural-language question to the GeoChat microservice.
        """
        start_time = time.perf_counter()
        img_bytes = _image_to_bytes(image_input)
        question = prompt or "Describe what is visible in this satellite image."

        headers: Dict[str, str] = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        files = {
            "image": ("image.png", img_bytes, "image/png"),
        }
        data = {
            "question": question,
        }

        logger.info(f"Dispatching to GeoChat: {self._endpoint} (task={task})")

        try:
            resp = requests.post(
                self._endpoint,
                headers=headers,
                files=files,
                data=data,
                timeout=(1.5, self.timeout),
            )
        except requests.exceptions.Timeout as e:
            raise VisionTimeoutError(
                f"GeoChat service timed out after {self.timeout}s at {self._endpoint}: {e}",
                provider="geochat",
            ) from e
        except (requests.exceptions.ConnectionError, requests.exceptions.RequestException) as e:
            raise VisionNetworkError(
                f"Network error connecting to GeoChat at {self._endpoint}: {e}",
                provider="geochat",
            ) from e

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Handle HTTP Errors
        if resp.status_code in {401, 403}:
            raise VisionAuthenticationError(
                f"HTTP {resp.status_code} Unauthorized from GeoChat endpoint: {resp.text}",
                provider="geochat",
                status_code=resp.status_code,
            )

        if resp.status_code == 429:
            raise VisionRateLimitError(
                f"HTTP 429 Rate Limit from GeoChat endpoint: {resp.text}",
                provider="geochat",
                status_code=429,
            )

        if resp.status_code >= 400:
            raise VisionResponseError(
                f"HTTP {resp.status_code} from GeoChat endpoint: {resp.text}",
                provider="geochat",
                status_code=resp.status_code,
            )

        # Parse Successful Response
        try:
            body = resp.json()
        except Exception:
            body = resp.text

        if isinstance(body, dict):
            answer = (
                body.get("response")
                or body.get("answer")
                or body.get("text")
                or body.get("generated_text")
                or body.get("output")
                or body.get("message")
                or body.get("result")
                or ""
            )
            if not answer and "detail" in body:
                answer = str(body["detail"])
        elif isinstance(body, str):
            answer = body
        else:
            answer = str(body)

        return VisionResponse(
            text=str(answer).strip(),
            grounding=None,
            raw_json=body if isinstance(body, dict) else {"raw": str(body)},
            latency_ms=latency_ms,
            provider="geochat",
            model="GeoChat-7B",
            selected_model="GeoChat-7B",
            attempted_models=["GeoChat-7B"],
            fallback_used=False,
            fallback_reason=None,
        )

    async def analyze_image(
        self,
        image_input: Any,
        prompt: str,
        *,
        task: TaskType = "vqa",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> VisionResponse:
        """Asynchronously dispatch an image to GeoChat."""
        return await asyncio.to_thread(
            self.analyze_image_sync,
            image_input,
            prompt,
            task=task,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
