"""
GeoChat Inference — Real multimodal VQA execution using the verified 504x504 pipeline.

Preserves the exact inference path verified in Colab:
  PIL image → GeoChat image processor → 504x504 → image token handling
  → multimodal model → generate() → decode → answer
"""
from __future__ import annotations
import io
import time
import logging
from typing import Dict, Any, Tuple

import torch
from PIL import Image

from .runtime import GeoChatRuntime

logger = logging.getLogger("geochat.inference")

# GeoChat uses a 504x504 image size (not standard 336x336 CLIP)
GEOCHAT_IMAGE_SIZE = 504
DEFAULT_IMAGE_TOKEN = "<image>"


def _prepare_image(
    image_bytes: bytes,
    runtime: GeoChatRuntime,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Preprocess image bytes into the model's expected pixel tensor.

    Returns:
        pixel_values: torch.Tensor of shape [1, 3, 504, 504]
        original_size: (width, height) of the original image
    """
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = pil_image.size  # (width, height)

    # Use the GeoChat image processor (loaded with the model)
    image_tensor = runtime.image_processor.preprocess(
        pil_image, return_tensors="pt"
    )["pixel_values"]

    # Move to model device with correct dtype
    try:
        first_param = next(runtime.model.parameters())
        device = first_param.device
        dtype = first_param.dtype
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16

    image_tensor = image_tensor.to(device=device, dtype=dtype)

    return image_tensor, original_size


def _build_prompt(question: str) -> str:
    """
    Build the GeoChat vicuna_v1 conversation prompt with image token.
    """
    return f"USER: {DEFAULT_IMAGE_TOKEN}\n{question}\nASSISTANT:"


def run_vqa(
    image_bytes: bytes,
    question: str,
) -> Dict[str, Any]:
    """
    Execute real GeoChat VQA inference.

    This is the verified inference path:
      image bytes → PIL → image_processor → pixel_values [1, 3, 504, 504]
      question → tokenizer → input_ids
      model.generate(input_ids, images=pixel_values) → decode → answer

    No mock fallback. No hardcoded answers.
    """
    runtime = GeoChatRuntime.get_instance()

    if not runtime.is_loaded:
        raise RuntimeError("GeoChat model is not loaded. Service startup failed.")

    start_time = time.perf_counter()

    # 1. Preprocess image to 504x504 tensor
    image_tensor, original_size = _prepare_image(image_bytes, runtime)

    # 2. Build prompt with image token
    prompt = _build_prompt(question)

    # 3. Tokenize
    input_ids = runtime.tokenizer(prompt, return_tensors="pt").input_ids
    try:
        first_param = next(runtime.model.parameters())
        device = first_param.device
    except StopIteration:
        device = torch.device("cuda")

    input_ids = input_ids.to(device)

    # 4. Generate with multimodal inputs
    with torch.no_grad():
        try:
            # GeoChat/LLaVA uses 'images' kwarg for pixel values
            output_ids = runtime.model.generate(
                input_ids=input_ids,
                images=image_tensor,
                max_new_tokens=runtime.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        except TypeError:
            # Fallback to pixel_values kwarg if model API differs
            output_ids = runtime.model.generate(
                input_ids=input_ids,
                pixel_values=image_tensor,
                max_new_tokens=runtime.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

    # 5. Decode — slice off the input tokens to get only generated tokens
    generated_ids = output_ids[0][input_ids.shape[1]:]
    answer = runtime.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    logger.info(
        f"VQA completed: question='{question[:60]}...', "
        f"answer='{answer[:80]}...', latency={elapsed_ms:.0f}ms"
    )

    return {
        "task": "vqa",
        "model": "GeoChat-7B",
        "answer": answer,
        "latency_ms": elapsed_ms,
        "image_size": list(original_size),
        "processed_size": [GEOCHAT_IMAGE_SIZE, GEOCHAT_IMAGE_SIZE],
        "mode": "real",
    }
