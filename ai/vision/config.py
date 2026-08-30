"""
Configuration management for the SatQuery AI Vision subsystem.
Supports Qwen2.5-VL and Qwen3-VL models with task-level selection.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Optional


# Canonical OpenRouter Model Slugs for Qwen VL series
MODEL_SLUGS: Dict[str, str] = {
    "qwen25_free": "qwen/qwen-2.5-vl-7b-instruct:free",
    "qwen25": "qwen/qwen-2.5-vl-7b-instruct",
    "qwen3": "qwen/qwen3-vl-8b-instruct",
    "qwen3_free": "qwen/qwen3-vl-8b-instruct:free",
    "qwen3_30b": "qwen/qwen3-vl-30b-a3b-instruct",
    "qwen3_235b": "qwen/qwen3-vl-235b-a22b-instruct",
}

# User-friendly short aliases
MODEL_ALIASES: Dict[str, str] = {
    "qwen25": MODEL_SLUGS["qwen25_free"],
    "qwen2.5": MODEL_SLUGS["qwen25_free"],
    "qwen2.5-vl": MODEL_SLUGS["qwen25_free"],
    "qwen2.5-vl-7b": MODEL_SLUGS["qwen25_free"],
    "qwen3": MODEL_SLUGS["qwen3"],
    "qwen3-vl": MODEL_SLUGS["qwen3"],
    "qwen3-vl-8b": MODEL_SLUGS["qwen3"],
    "qwen3_free": MODEL_SLUGS["qwen3_free"],
}


def resolve_model_slug(raw_name: Optional[str], default: str = MODEL_SLUGS["qwen25_free"]) -> str:
    """Resolve a model name or alias to its canonical OpenRouter slug."""
    if not raw_name:
        return default
    normalized = raw_name.strip().lower()
    return MODEL_ALIASES.get(normalized, raw_name.strip())


@dataclass(frozen=True)
class VisionConfig:
    """
    Configuration for remote multimodal vision providers.
    Secrets are automatically masked in logging and string representations.
    """
    provider: str = "qwen_openrouter"
    model: str = MODEL_SLUGS["qwen25_free"]
    vqa_model: Optional[str] = None
    caption_model: Optional[str] = None
    ground_model: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: Optional[str] = None
    timeout: float = 45.0
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> VisionConfig:
        """Load vision configuration from environment variables."""
        provider = os.environ.get("VISION_PROVIDER", "qwen_openrouter").lower()
        
        raw_model = os.environ.get("VISION_MODEL", MODEL_SLUGS["qwen25_free"])
        resolved_model = resolve_model_slug(raw_model)

        raw_vqa = os.environ.get("VISION_VQA_MODEL")
        vqa_model = resolve_model_slug(raw_vqa, default=resolved_model) if raw_vqa else resolved_model

        raw_cap = os.environ.get("VISION_CAPTION_MODEL")
        caption_model = resolve_model_slug(raw_cap, default=resolved_model) if raw_cap else resolved_model

        raw_grd = os.environ.get("VISION_GROUND_MODEL")
        ground_model = resolve_model_slug(raw_grd, default=resolved_model) if raw_grd else resolved_model

        base_url = os.environ.get("VISION_BASE_URL", os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
        
        # Check OPENROUTER_API_KEY, fallback to VISION_API_KEY
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("VISION_API_KEY")

        timeout_str = os.environ.get("VISION_TIMEOUT", "45.0")
        try:
            timeout = float(timeout_str)
        except ValueError:
            timeout = 45.0

        retries_str = os.environ.get("VISION_MAX_RETRIES", "2")
        try:
            max_retries = int(retries_str)
        except ValueError:
            max_retries = 2

        return cls(
            provider=provider,
            model=resolved_model,
            vqa_model=vqa_model,
            caption_model=caption_model,
            ground_model=ground_model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def get_model_for_task(self, task: str) -> str:
        """Return the resolved model slug for a given vision task."""
        if task == "vqa" and self.vqa_model:
            return self.vqa_model
        elif task == "caption" and self.caption_model:
            return self.caption_model
        elif task == "ground" and self.ground_model:
            return self.ground_model
        return self.model

    def __repr__(self) -> str:
        masked = "***" if self.api_key else "None"
        return (
            f"VisionConfig(provider='{self.provider}', model='{self.model}', "
            f"vqa_model='{self.vqa_model}', ground_model='{self.ground_model}', "
            f"base_url='{self.base_url}', api_key={masked}, "
            f"timeout={self.timeout}, max_retries={self.max_retries})"
        )
