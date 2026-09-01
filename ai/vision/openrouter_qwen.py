"""
OpenRouter Vision Provider implementation for SatQuery AI.
Interacts with the OpenRouter multimodal chat completions API for VQA, Captioning, and Grounding.
Supports multi-model routing across primary, secondary, and tertiary candidates with transient error fallback.
"""
from __future__ import annotations
import base64
import io
import json
import logging
import re
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

logger = logging.getLogger("satquery.vision.openrouter")


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
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=90)
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}", (width, height)
        except Exception:
            b64 = base64.b64encode(image_input).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}", (512, 512)

    else:
        raise VisionError(f"Unsupported image input type: {type(image_input)}")


def _is_account_level_rate_limit(error_body: str) -> bool:
    """Check if the rate limit response indicates global account quota exhaustion."""
    if not error_body:
        return False
    lower = error_body.lower()
    quota_indicators = [
        "free-models-per-day",
        "free models per day",
        "rate limit exceeded for free models",
        "daily rate limit",
        "account quota exceeded",
        "free tier limit",
        "daily request limit",
    ]
    return any(ind in lower for ind in quota_indicators)


def _parse_and_validate_grounding(content_text: str, img_w: int, img_h: int) -> Optional[GroundingResult]:
    """
    Parse and strictly validate structured bounding boxes from model output.
    Returns GroundingResult if valid machine-readable boxes exist, or None if unsupported / text-only.
    """
    if not content_text or not content_text.strip():
        return None

    # 1. Attempt extracting JSON from raw text or markdown code fence
    json_str = None
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content_text, re.DOTALL)
    if fence_match:
        json_str = fence_match.group(1).strip()
    else:
        brace_start = content_text.find("{")
        brace_end = content_text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            json_str = content_text[brace_start:brace_end + 1].strip()

    if not json_str:
        return None

    try:
        data = json.loads(json_str)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    raw_objects = data.get("objects")
    if not isinstance(raw_objects, list):
        return None

    valid_boxes: List[GroundingBox] = []
    for item in raw_objects:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        box = item.get("box")
        confidence = item.get("confidence")

        if not label or not isinstance(label, str) or not label.strip():
            continue
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue

        try:
            x0, y0, x1, y1 = [float(c) for c in box]
        except (ValueError, TypeError):
            continue

        # Coordinate geometric sanity checks
        if x0 > x1 or y0 > y1:
            continue

        try:
            gbox = GroundingBox(label=label.strip(), box=[x0, y0, x1, y1], confidence=confidence)
            valid_boxes.append(gbox)
        except Exception:
            continue

    if not valid_boxes:
        return None

    return GroundingResult(objects=valid_boxes)


class OpenRouterVisionProvider(VisionProvider):
    """
    Resilient Multimodal Vision Provider targeting OpenRouter models:
    - Primary: google/gemma-4-26b-a4b-it:free
    - Secondary: google/gemma-4-31b-it:free
    - Tertiary: nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
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

    def _execute_single_request(self, payload: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Execute a single HTTP request for a specified model with bounded transient retry."""
        if not self.config.api_key:
            raise VisionAuthenticationError(
                "OPENROUTER_API_KEY is not set. Please configure OPENROUTER_API_KEY in the environment.",
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
                        f"HTTP {status_code} from OpenRouter ({model_name}): {error_body}",
                        provider=self.config.provider,
                        status_code=status_code,
                    )
                elif status_code == 400:
                    raise VisionResponseError(
                        f"HTTP 400 Bad Request from OpenRouter ({model_name}): {error_body}",
                        provider=self.config.provider,
                        status_code=400,
                    )
                elif status_code == 429:
                    is_account_limit = _is_account_level_rate_limit(error_body)
                    retry_after = None
                    ra_hdr = e.headers.get("Retry-After") if hasattr(e, "headers") else None
                    if ra_hdr:
                        try:
                            retry_after = float(ra_hdr)
                        except ValueError:
                            pass

                    last_error = VisionRateLimitError(
                        f"HTTP 429 Rate Limit from OpenRouter ({model_name}): {error_body}",
                        provider=self.config.provider,
                        status_code=429,
                        is_account_limit=is_account_limit,
                        retry_after=retry_after,
                    )

                    if is_account_limit:
                        # Account-level daily limit: Fail fast, do not burn other model calls
                        logger.error(f"Account-level rate limit detected: {error_body}")
                        raise last_error

                    if attempt < self.config.max_retries:
                        sleep_s = retry_after if (retry_after is not None and retry_after <= 5.0) else (1.5 * (2 ** attempt))
                        time.sleep(sleep_s)
                        continue
                    raise last_error

                elif status_code in {500, 502, 503, 504}:
                    last_error = VisionResponseError(
                        f"HTTP {status_code} Upstream Server Error from OpenRouter ({model_name}): {error_body}",
                        provider=self.config.provider,
                        status_code=status_code,
                    )
                    if attempt < self.config.max_retries:
                        time.sleep(1.0 * (2 ** attempt))
                        continue
                    raise last_error
                else:
                    raise VisionResponseError(
                        f"HTTP {status_code} Error from OpenRouter ({model_name}): {error_body}",
                        provider=self.config.provider,
                        status_code=status_code,
                    )

            except urllib.error.URLError as e:
                if "timed out" in str(e).lower():
                    last_error = VisionTimeoutError(
                        f"OpenRouter request timed out after {self.config.timeout}s for model {model_name}: {e}",
                        provider=self.config.provider,
                    )
                else:
                    last_error = VisionNetworkError(
                        f"Network error connecting to OpenRouter for model {model_name}: {e}",
                        provider=self.config.provider,
                    )
                if attempt < self.config.max_retries:
                    time.sleep(1.0 * (2 ** attempt))
                    continue
                raise last_error

            except TimeoutError as e:
                last_error = VisionTimeoutError(
                    f"OpenRouter socket timed out after {self.config.timeout}s for model {model_name}: {e}",
                    provider=self.config.provider,
                )
                if attempt < self.config.max_retries:
                    time.sleep(1.0 * (2 ** attempt))
                    continue
                raise last_error

    def _execute_http_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Backward-compatible HTTP execution wrapper."""
        model_name = payload.get("model", self.config.model)
        return self._execute_single_request(payload, model_name)

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
        Synchronously analyze an image with resilient multi-model routing and fallbacks.
        """
        start_total = time.perf_counter()

        # Determine effective token budget per task
        if max_tokens is not None:
            effective_max_tokens = max_tokens
        elif task == "caption":
            effective_max_tokens = 768
        else:
            effective_max_tokens = 512

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

        # 3. Determine candidate models list
        explicit_model = kwargs.get("model")
        if explicit_model:
            candidate_models = [explicit_model]
        else:
            candidate_models = self.config.get_candidate_models_for_task(task)

        attempted_models: List[str] = []
        last_error: Optional[Exception] = None
        fallback_reason: Optional[str] = None

        for idx, current_model in enumerate(candidate_models):
            attempted_models.append(current_model)

            payload: Dict[str, Any] = {
                "model": current_model,
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
                "max_tokens": effective_max_tokens,
            }

            if task in {"vqa", "caption"}:
                payload["reasoning"] = {"effort": "low"}

            if task == "ground":
                payload["response_format"] = {"type": "json_object"}

            try:
                raw_resp = self._execute_http_request(payload)
            except VisionRateLimitError as e:
                if e.is_account_limit:
                    # Fail immediately on account limit; do not try further models
                    raise
                last_error = e
                fallback_reason = "upstream_rate_limit"
                logger.warning(f"Model {current_model} rate limited (429). Attempting fallback...")
                continue
            except (VisionTimeoutError, VisionNetworkError) as e:
                last_error = e
                fallback_reason = "provider_timeout" if isinstance(e, VisionTimeoutError) else "network_error"
                logger.warning(f"Model {current_model} hit network/timeout error. Attempting fallback...")
                continue
            except VisionResponseError as e:
                if e.status_code and e.status_code in {400, 404, 429, 500, 502, 503, 504}:
                    last_error = e
                    fallback_reason = f"upstream_error_{e.status_code}"
                    logger.warning(f"Model {current_model} returned {e.status_code}. Attempting fallback...")
                    continue
                else:
                    # Non-transient error
                    raise
            except VisionAuthenticationError:
                # Auth error: fail immediately
                raise

            # 4. Extract generated text strictly from choices[0].message.content
            try:
                choice = raw_resp["choices"][0]
                choice_msg = choice.get("message", {})
                finish_reason = choice.get("finish_reason")
                raw_content = choice_msg.get("content") or ""
                content_text = raw_content.strip()
            except (KeyError, IndexError, TypeError) as e:
                last_error = VisionResponseError(
                    f"Malformed response structure from OpenRouter ({current_model}): {raw_resp}",
                    provider=self.config.provider,
                )
                fallback_reason = "malformed_response"
                continue

            # Ensure choices[0].message.content is non-empty; NEVER expose internal reasoning fields as final text
            if not content_text:
                if finish_reason == "length":
                    fallback_reason = "token_limit_exceeded"
                    last_error = VisionResponseError(
                        f"Model {current_model} returned empty content with finish_reason='length' (reasoning budget exhausted)",
                        provider=self.config.provider,
                        status_code=200,
                    )
                else:
                    fallback_reason = "empty_content"
                    last_error = VisionResponseError(
                        f"Model {current_model} returned empty content (finish_reason='{finish_reason}')",
                        provider=self.config.provider,
                        status_code=200,
                    )
                logger.warning(
                    f"Model {current_model} returned empty content (finish_reason={finish_reason}). Triggering fallback..."
                )
                continue

            # 5. Process and validate Grounding if applicable
            grounding_result = None
            if task == "ground":
                grounding_result = _parse_and_validate_grounding(content_text, img_w, img_h)
                if grounding_result is None or len(grounding_result.objects) == 0:
                    # Model returned text-only or unsupported grounding format
                    if idx < len(candidate_models) - 1:
                        fallback_reason = "grounding_unsupported"
                        logger.warning(
                            f"Model {current_model} returned unsupported/text-only grounding. Attempting next candidate..."
                        )
                        continue
                    else:
                        # All candidates attempted, return empty GroundingResult rather than hallucinating
                        grounding_result = GroundingResult(objects=[])

            latency_ms = round((time.perf_counter() - start_total) * 1000.0, 2)
            fallback_used = len(attempted_models) > 1

            return VisionResponse(
                text=content_text,
                grounding=grounding_result,
                raw_json=raw_resp,
                latency_ms=latency_ms,
                provider="openrouter",
                model=current_model,
                selected_model=current_model,
                attempted_models=attempted_models,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason if fallback_used else None,
            )

        raise last_error or VisionError(
            f"All candidate vision models failed for task '{task}': {attempted_models}",
            provider=self.config.provider,
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


# Backward-compatible alias
OpenRouterQwenVisionProvider = OpenRouterVisionProvider
