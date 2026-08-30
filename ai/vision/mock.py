"""
Mock Vision Provider for offline unit testing and development.
"""
from __future__ import annotations
import json
from typing import Any, Callable, Dict, List, Optional

from .base import GroundingBox, GroundingResult, TaskType, VisionProvider, VisionResponse


class MockVisionProvider(VisionProvider):
    """
    Mock implementation of VisionProvider.
    Enables 100% offline deterministic test suites without external network calls.
    """

    def __init__(
        self,
        default_vqa_response: str = "[MOCK QWEN] The image shows an urban residential district with multiple buildings and roads.",
        default_caption_response: str = "[MOCK QWEN] High-resolution satellite view of mixed agricultural fields and water bodies.",
        default_ground_objects: Optional[List[Dict[str, Any]]] = None,
        custom_handler: Optional[Callable[..., VisionResponse]] = None,
    ):
        self.default_vqa = default_vqa_response
        self.default_caption = default_caption_response
        self.default_ground_objects = default_ground_objects or [
            {"label": "building", "box": [0.1, 0.2, 0.4, 0.5]},
            {"label": "water_body", "box": [0.6, 0.7, 0.9, 0.95]},
        ]
        self.custom_handler = custom_handler
        self.calls: List[Dict[str, Any]] = []

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
        self.calls.append({
            "prompt": prompt,
            "task": task,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "kwargs": kwargs,
        })

        if self.custom_handler:
            return self.custom_handler(image_input=image_input, prompt=prompt, task=task, **kwargs)

        if task == "caption":
            return VisionResponse(
                text=self.default_caption,
                grounding=None,
                latency_ms=1.5,
                provider="mock_vision",
                model="qwen/qwen-2.5-vl-7b-instruct:free",
            )
        elif task == "ground":
            boxes = [GroundingBox.model_validate(obj) for obj in self.default_ground_objects]
            result = GroundingResult(objects=boxes)
            raw_text = json.dumps({"objects": [b.model_dump() for b in boxes]})
            return VisionResponse(
                text=raw_text,
                grounding=result,
                latency_ms=2.0,
                provider="mock_vision",
                model="qwen/qwen-2.5-vl-7b-instruct:free",
            )
        else:  # vqa
            return VisionResponse(
                text=self.default_vqa,
                grounding=None,
                latency_ms=1.0,
                provider="mock_vision",
                model="qwen/qwen-2.5-vl-7b-instruct:free",
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
        return self.analyze_image_sync(
            image_input=image_input,
            prompt=prompt,
            task=task,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
