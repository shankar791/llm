"""
Configuration management for the SatQuery AI Vision subsystem.
Supports multi-model routing and fallbacks across OpenRouter vision models:
- Primary: google/gemma-4-26b-a4b-it:free
- Secondary: google/gemma-4-31b-it:free
- Tertiary: nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
"""
from __future__ import annotations
from dataclasses import dataclass, field
import os
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Canonical OpenRouter Model Slugs
MODEL_SLUGS: Dict[str, str] = {
    "gemma_26b": "google/gemma-4-26b-a4b-it:free",
    "gemma_31b": "google/gemma-4-31b-it:free",
    "nemotron_nano": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "minimax_m3": "minimax/minimax-m3:free",
    "qwen25_free": "qwen/qwen-2.5-vl-7b-instruct:free",
    "qwen25": "qwen/qwen-2.5-vl-7b-instruct",
    "qwen3": "qwen/qwen3-vl-8b-instruct",
    "qwen3_free": "qwen/qwen3-vl-8b-instruct:free",
    "qwen3_30b": "qwen/qwen3-vl-30b-a3b-instruct",
    "qwen3_235b": "qwen/qwen3-vl-235b-a22b-instruct",
}

# Default Vision Models
DEFAULT_VISION_PRIMARY_MODEL = MODEL_SLUGS["gemma_26b"]
DEFAULT_VISION_SECONDARY_MODEL = MODEL_SLUGS["gemma_31b"]
DEFAULT_VISION_TERTIARY_MODEL = MODEL_SLUGS["nemotron_nano"]

# Default Task Models & Fallback Lists
DEFAULT_VISION_VQA_MODEL = DEFAULT_VISION_PRIMARY_MODEL
DEFAULT_VISION_VQA_FALLBACKS = [DEFAULT_VISION_SECONDARY_MODEL, DEFAULT_VISION_TERTIARY_MODEL]

DEFAULT_VISION_CAPTION_MODEL = DEFAULT_VISION_PRIMARY_MODEL
DEFAULT_VISION_CAPTION_FALLBACKS = [DEFAULT_VISION_SECONDARY_MODEL, DEFAULT_VISION_TERTIARY_MODEL]

DEFAULT_VISION_GROUND_MODEL = DEFAULT_VISION_SECONDARY_MODEL
DEFAULT_VISION_GROUND_FALLBACKS = [DEFAULT_VISION_PRIMARY_MODEL, DEFAULT_VISION_TERTIARY_MODEL]

# User-friendly short aliases
MODEL_ALIASES: Dict[str, str] = {
    "gemma_26b": MODEL_SLUGS["gemma_26b"],
    "gemma-26b": MODEL_SLUGS["gemma_26b"],
    "gemma-4-26b": MODEL_SLUGS["gemma_26b"],
    "gemma-4-26b-a4b-it": MODEL_SLUGS["gemma_26b"],
    "gemma_31b": MODEL_SLUGS["gemma_31b"],
    "gemma-31b": MODEL_SLUGS["gemma_31b"],
    "gemma-4-31b": MODEL_SLUGS["gemma_31b"],
    "gemma-4-31b-it": MODEL_SLUGS["gemma_31b"],
    "nemotron": MODEL_SLUGS["nemotron_nano"],
    "nemotron_nano": MODEL_SLUGS["nemotron_nano"],
    "nemotron-3-nano": MODEL_SLUGS["nemotron_nano"],
    "minimax": MODEL_SLUGS["minimax_m3"],
    "minimax_m3": MODEL_SLUGS["minimax_m3"],
    "minimax-m3": MODEL_SLUGS["minimax_m3"],
    "qwen25": MODEL_SLUGS["qwen25_free"],
    "qwen2.5": MODEL_SLUGS["qwen25_free"],
    "qwen2.5-vl": MODEL_SLUGS["qwen25_free"],
    "qwen2.5-vl-7b": MODEL_SLUGS["qwen25_free"],
    "qwen3": MODEL_SLUGS["qwen3"],
    "qwen3-vl": MODEL_SLUGS["qwen3"],
    "qwen3-vl-8b": MODEL_SLUGS["qwen3"],
    "qwen3_free": MODEL_SLUGS["qwen3_free"],
}


def resolve_model_slug(raw_name: Optional[str], default: str = DEFAULT_VISION_PRIMARY_MODEL) -> str:
    """Resolve a model name or alias to its canonical OpenRouter slug."""
    if not raw_name:
        return default
    normalized = raw_name.strip().lower()
    return MODEL_ALIASES.get(normalized, raw_name.strip())


def parse_model_list(raw_list: Optional[str], default_list: List[str]) -> List[str]:
    """Parse comma-separated model slugs or aliases into a list of canonical model slugs."""
    if not raw_list:
        return list(default_list)
    items = [resolve_model_slug(item.strip()) for item in raw_list.split(",") if item.strip()]
    return items if items else list(default_list)


@dataclass(frozen=True)
class VisionConfig:
    """
    Configuration for remote multimodal vision providers.
    Secrets are automatically masked in logging and string representations.
    """
    provider: str = "openrouter"
    primary_model: str = DEFAULT_VISION_PRIMARY_MODEL
    secondary_model: str = DEFAULT_VISION_SECONDARY_MODEL
    tertiary_model: str = DEFAULT_VISION_TERTIARY_MODEL
    model: str = DEFAULT_VISION_PRIMARY_MODEL
    vqa_model: str = DEFAULT_VISION_VQA_MODEL
    vqa_fallbacks: tuple[str, ...] = tuple(DEFAULT_VISION_VQA_FALLBACKS)
    caption_model: str = DEFAULT_VISION_CAPTION_MODEL
    caption_fallbacks: tuple[str, ...] = tuple(DEFAULT_VISION_CAPTION_FALLBACKS)
    ground_model: str = DEFAULT_VISION_GROUND_MODEL
    ground_fallbacks: tuple[str, ...] = tuple(DEFAULT_VISION_GROUND_FALLBACKS)
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: Optional[str] = None
    geochat_base_url: str = "http://172.25.166.59:8000"
    geochat_api_key: Optional[str] = None
    timeout: float = 45.0
    max_retries: int = 2

    def __post_init__(self):
        # If custom model/primary_model was passed, synchronize task models if they were at default
        effective_primary = self.primary_model if self.primary_model != DEFAULT_VISION_PRIMARY_MODEL else self.model
        if effective_primary != DEFAULT_VISION_PRIMARY_MODEL:
            object.__setattr__(self, "primary_model", effective_primary)
            object.__setattr__(self, "model", effective_primary)
            if self.vqa_model == DEFAULT_VISION_VQA_MODEL:
                object.__setattr__(self, "vqa_model", effective_primary)
            if self.caption_model == DEFAULT_VISION_CAPTION_MODEL:
                object.__setattr__(self, "caption_model", effective_primary)

    @classmethod
    def from_env(cls) -> VisionConfig:
        """Load vision configuration from environment variables."""
        provider = os.environ.get("VISION_PROVIDER", "openrouter").lower()
        
        raw_primary = os.environ.get("VISION_PRIMARY_MODEL") or os.environ.get("VISION_MODEL")
        primary_model = resolve_model_slug(raw_primary, default=DEFAULT_VISION_PRIMARY_MODEL)

        raw_secondary = os.environ.get("VISION_SECONDARY_MODEL")
        secondary_model = resolve_model_slug(raw_secondary, default=DEFAULT_VISION_SECONDARY_MODEL)

        raw_tertiary = os.environ.get("VISION_TERTIARY_MODEL")
        tertiary_model = resolve_model_slug(raw_tertiary, default=DEFAULT_VISION_TERTIARY_MODEL)

        # Task-specific Primary Models
        raw_vqa = os.environ.get("VISION_VQA_MODEL")
        vqa_model = resolve_model_slug(raw_vqa, default=primary_model) if raw_vqa else primary_model

        raw_cap = os.environ.get("VISION_CAPTION_MODEL")
        caption_model = resolve_model_slug(raw_cap, default=primary_model) if raw_cap else primary_model

        raw_grd = os.environ.get("VISION_GROUND_MODEL")
        ground_model = resolve_model_slug(raw_grd, default=secondary_model) if raw_grd else secondary_model

        # Task-specific Fallbacks
        raw_vqa_fb = os.environ.get("VISION_VQA_FALLBACKS")
        vqa_fallbacks = tuple(parse_model_list(raw_vqa_fb, [secondary_model, tertiary_model]))

        raw_cap_fb = os.environ.get("VISION_CAPTION_FALLBACKS")
        caption_fallbacks = tuple(parse_model_list(raw_cap_fb, [secondary_model, tertiary_model]))

        raw_grd_fb = os.environ.get("VISION_GROUND_FALLBACKS")
        ground_fallbacks = tuple(parse_model_list(raw_grd_fb, [primary_model, tertiary_model]))

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

        geochat_base_url = os.environ.get("GEOCHAT_BASE_URL", "http://172.25.166.59:8000")
        geochat_api_key = os.environ.get("GEOCHAT_API_KEY")

        return cls(
            provider=provider,
            primary_model=primary_model,
            secondary_model=secondary_model,
            tertiary_model=tertiary_model,
            model=primary_model,
            vqa_model=vqa_model,
            vqa_fallbacks=vqa_fallbacks,
            caption_model=caption_model,
            caption_fallbacks=caption_fallbacks,
            ground_model=ground_model,
            ground_fallbacks=ground_fallbacks,
            base_url=base_url,
            api_key=api_key,
            geochat_base_url=geochat_base_url,
            geochat_api_key=geochat_api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def get_model_for_task(self, task: str) -> str:
        """Return the resolved primary model slug for a given vision task."""
        if task == "vqa":
            return self.vqa_model
        elif task == "caption":
            return self.caption_model
        elif task == "ground":
            return self.ground_model
        return self.model

    def get_candidate_models_for_task(self, task: str) -> List[str]:
        """
        Return the ordered, deduplicated candidate models (primary followed by fallbacks)
        for the given vision task.
        """
        if task == "vqa":
            candidates = [self.vqa_model] + list(self.vqa_fallbacks)
        elif task == "caption":
            candidates = [self.caption_model] + list(self.caption_fallbacks)
        elif task == "ground":
            candidates = [self.ground_model] + list(self.ground_fallbacks)
        else:
            candidates = [self.model, self.secondary_model, self.tertiary_model]

        # Deduplicate preserving order
        seen = set()
        deduped = []
        for m in candidates:
            if m and m not in seen:
                seen.add(m)
                deduped.append(m)
        return deduped

    def __repr__(self) -> str:
        masked = "***" if self.api_key else "None"
        masked_gc = "***" if self.geochat_api_key else "None"
        return (
            f"VisionConfig(provider='{self.provider}', primary_model='{self.primary_model}', "
            f"secondary_model='{self.secondary_model}', tertiary_model='{self.tertiary_model}', "
            f"vqa_model='{self.vqa_model}', ground_model='{self.ground_model}', "
            f"base_url='{self.base_url}', api_key={masked}, "
            f"geochat_base_url='{self.geochat_base_url}', geochat_api_key={masked_gc}, "
            f"timeout={self.timeout}, max_retries={self.max_retries})"
        )

