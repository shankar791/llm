"""
OpenRouter Qwen2.5-VL Vision Provider implementation for SatQuery AI.
Interacts with the OpenRouter multimodal chat completions API for VQA, Captioning, and Grounding.
"""
from __future__ import annotations
import base64
import io
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from .base import GroundingBox, GroundingResult, TaskType, VisionProvider, VisionResponse
from .config import VisionConfig
from .errors import (
    GroundingParseError,
    VisionAuthenticationError,
    VisionConfigurationError,
    VisionError,
    VisionNetworkError,
    VisionRateLimitError,
    VisionResponseError,
    VisionTimeoutError,
)

logger = logging.getLogger("satquery.vision.qwen")


def _encode_image_to_data_url(
    image_input: Union[bytes, np.ndarray, Image.Image]
) -> Tuple[str, Tuple[int, int]]:
    """
    Encode an image to a Base64 data URL and return its pixel (width, height) dimensions.
    """
    if isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
        width, height = img.size
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}", (width, height)

    elif isinstance(image_input, np.ndarray):
        # Handle numpy arrays [H, W, C] or [H, W]
        if image_input.ndim == 2:
            img = Image.fromarray(image_input).convert("RGB")
        elif image_input.ndim == 3:
            if image_input.shape[0] in {1, 3, 4} and image_input.shape[2] not in {1, 3, 4}:
                # Transpose [C, H, W] -> [H, W, C]
                image_input = np.transpose(image_input, (1, 2, 0))
            if image_input.shape[2] == 1:
                img = Image.fromarray(image_input[:, :, 0]).convert("RGB")
            else:
                img = Image.fromarray(image_input[:, :, :3]).convert("RGB")
        else:
            raise VisionError(f"Unsupported numpy array dimensions: {image_input.shape}")

        width, height = img.size
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}", (width, height)

    elif isinstance(image_input, bytes):
        try:
            img = Image.open(io.BytesIO(image_input)).convert("RGB")
            width, height = img.size
            # If already valid bytes, inspect MIME or re-encode clean JPEG
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=90)
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}", (width, height)
        except Exception as e:
            # Fallback direct b64 if PIL fails
            b64 = base64.b64encode(image_input).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}", (512, 512)

    else:
        raise VisionError(f"Unsupported image input type: {type(image_input)}")


class OpenRouterQwenVisionProvider(VisionProvider):
    """
    Multimodal Vision Provider targeting Qwen2.5-VL via OpenRouter.
    """

    SYSTEM_PROMPT_VQA = (
        "You are an expert remote sensing and Earth observation vision assistant. "
        "Analyze the provided satellite/aerial imagery and answer the user's question directly, concisely, and factually."
    )

    SYSTEM_PROMPT_CAPTION = (
        "You are an expert remote sensing analyst. "
        "Provide a clear, detailed, and concise overview describing the land cover, structures, terrain, and features in this satellite scene."
    )

    SYSTEM_PROMPT_GROUND = (
        "You are an object detection and spatial grounding engine for satellite imagery.\n"
        "Detect and locate the requested objects or features in the image.\n"
        "Respond with a JSON object strictly conforming to:\n"
        "{\n"
        '  "objects": [\n'
        '    {"label": "object_name", "box": [x0, y0, x1, y1]}\n'
        "  ]\n"
        "}\n"
        "Coordinates MUST be normalized floats in [0.0, 1.0] where (x0, y0) is top-left and (x1, y1) is bottom-right."
    )

    def __init__(self, config: Optional[VisionConfig] = None):
        self.config = config or VisionConfig.from_env()
        self._endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"

    def _execute_http_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HTTP request with transient retry logic."""
        if not self.config.api_key:
            raise VisionAuthenticationError(
                "OPENROUTER_API_KEY is not set. Please configure OPENROUTER_API_KEY or VISION_API_KEY in the environment.",
                provider=self.config.provider,
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
            "HTTP-Referer": "https://github.com/satquery-ai",
            "X-Title": "SatQuery AI",
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self._endpoint, data=data_bytes, headers=headers, method="POST")

        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                    resp_body = resp.read().decode("utf-8")
                    return json.loads(resp_body)

            except urllib.error.HTTPError as e:
                status_code = e.code
                error_body = e.read().decode("utf-8", errors="replace")

                if status_code in {401, 403}:
                    raise VisionAuthenticationError(
                        f"HTTP {status_code} from OpenRouter: {error_body}",
                        provider=self.config.provider,
                        status_code=status_code,
                    )
                elif status_code == 429:
                    last_error = VisionRateLimitError(
                        f"HTTP 429 Rate Limit from OpenRouter ({self.config.model}): {error_body}",
                        provider=self.config.provider,
                        status_code=429,
                    )
                    if attempt < self.config.max_retries:
                        time.sleep(1.5 * (2 ** attempt))
                        continue
                    raise last_error
                elif status_code in {500, 502, 503, 504}:
                    last_error = VisionResponseError(
                        f"HTTP {status_code} Server Error from OpenRouter: {error_body}",
                        provider=self.config.provider,
                        status_code=status_code,
                    )
                    if attempt < self.config.max_retries:
                        time.sleep(1.0 * (2 ** attempt))
                        continue
                    raise last_error
                else:
                    raise VisionResponseError(
                        f"HTTP {status_code} Error from OpenRouter: {error_body}",
                        provider=self.config.provider,
                        status_code=status_code,
                    )

            except urllib.error.URLError as e:
                if "timed out" in str(e).lower():
                    last_error = VisionTimeoutError(
                        f"OpenRouter request timed out after {self.config.timeout}s: {e}",
                        provider=self.config.provider,
                    )
                else:
                    last_error = VisionNetworkError(
                        f"Network error connecting to OpenRouter: {e}",
                        provider=self.config.provider,
                    )
                if attempt < self.config.max_retries:
                    time.sleep(1.0 * (2 ** attempt))
                    continue
                raise last_error

            except TimeoutError as e:
                last_error = VisionTimeoutError(
                    f"OpenRouter socket timed out after {self.config.timeout}s: {e}",
                    provider=self.config.provider,
                )
                if attempt < self.config.max_retries:
                    time.sleep(1.0 * (2 ** attempt))
                    continue
                raise last_error

        raise last_error or VisionError("Failed to execute OpenRouter request", provider=self.config.provider)

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
        """
        Synchronously analyze an image with Qwen2.5-VL via OpenRouter.
        """
        start_time = time.perf_counter()

        # 1. Encode image to data URL and retrieve dimensions
        data_url, (img_w, img_h) = _encode_image_to_data_url(image_input)

        # 2. Configure system prompt and user query per task
        if task == "caption":
            system_prompt = self.SYSTEM_PROMPT_CAPTION
            task_prompt = prompt or "Describe the satellite scene."
        elif task == "ground":
            system_prompt = self.SYSTEM_PROMPT_GROUND
            task_prompt = f"Locate: {prompt}"
        else:  # vqa
            system_prompt = self.SYSTEM_PROMPT_VQA
            task_prompt = prompt

        target_model = kwargs.get("model") or self.config.get_model_for_task(task)

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if task == "ground":
            payload["response_format"] = {"type": "json_object"}

        # 3. Dispatch HTTP request
        raw_resp = self._execute_http_request(payload)

        # 4. Extract generated text
        try:
            content_text = raw_resp["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise VisionResponseError(
                f"Malformed response structure from OpenRouter: {raw_resp}",
                provider=self.config.provider,
            ) from e

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # 5. Process Grounding task if applicable
        grounding_result = None
        if task == "ground":
            try:
                parsed_json = json.loads(content_text)
                grounding_result = GroundingResult.model_validate(parsed_json)
            except Exception as e:
                raise GroundingParseError(
                    f"Failed to parse structured grounding coordinates from Qwen response: {content_text}. Error: {e}",
                    provider=self.config.provider,
                ) from e

        return VisionResponse(
            text=content_text,
            grounding=grounding_result,
            raw_json=raw_resp,
            latency_ms=latency_ms,
            provider="openrouter",
            model=target_model,
        )

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
        """
        Asynchronously analyze an image (wraps synchronous execution in thread).
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.analyze_image_sync(
                image_input=image_input,
                prompt=prompt,
                task=task,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            ),
        )
