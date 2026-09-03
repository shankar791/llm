"""
Configuration management for the SatQuery AI Vision subsystem.
Supports multi-model routing and automatic sequential fallback across OpenRouter free vision models:
1. MODEL_1: google/gemma-4-26b-a4b-it:free
2. MODEL_2: google/gemma-4-31b-it:free
3. MODEL_3: nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
4. MODEL_4: minimax/minimax-m3:free
5. MODEL_5: thinkingmachines/inkling:free
"""
from __future__ import annotations
from dataclasses import dataclass, field
import os
from typing import Dict, List, Optional

try:
    from pathlib import Path
    from dotenv import load_dotenv
    _env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=True)
    else:
        load_dotenv(override=True)
except ImportError:
    pass


# Canonical OpenRouter Model Slugs
MODEL_SLUGS: Dict[str, str] = {
    "gemma_26b": "google/gemma-4-26b-a4b-it:free",
    "gemma_31b": "google/gemma-4-31b-it:free",
    "inkling": "thinkingmachines/inkling:free",
    "inkling_small": "thinkingmachines/inkling-small:free",
    "minimax_m3": "minimax/minimax-m3:free",
    "nemotron_nano": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "qwen25_free": "qwen/qwen-2.5-vl-7b-instruct:free",
    "qwen25": "qwen/qwen-2.5-vl-7b-instruct",
    "qwen3": "qwen/qwen3-vl-8b-instruct",
    "qwen3_free": "qwen/qwen3-vl-8b-instruct:free",
    "qwen3_30b": "qwen/qwen3-vl-30b-a3b-instruct",
    "qwen3_235b": "qwen/qwen3-vl-235b-a22b-instruct",
}

# Default Vision Models (in exact sequential fallback order)
MODEL_1 = MODEL_SLUGS["gemma_26b"]        # "google/gemma-4-26b-a4b-it:free"
MODEL_2 = MODEL_SLUGS["gemma_31b"]        # "google/gemma-4-31b-it:free"
MODEL_3 = MODEL_SLUGS["nemotron_nano"]     # "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
MODEL_4 = MODEL_SLUGS["minimax_m3"]       # "minimax/minimax-m3:free"
MODEL_5 = MODEL_SLUGS["inkling"]          # "thinkingmachines/inkling:free"

DEFAULT_VISION_PRIMARY_MODEL = MODEL_1
DEFAULT_VISION_SECONDARY_MODEL = MODEL_2
DEFAULT_VISION_TERTIARY_MODEL = MODEL_3
DEFAULT_VISION_QUATERNARY_MODEL = MODEL_4
DEFAULT_VISION_QUINARY_MODEL = MODEL_5

# Exact 5-Model Fallback Sequence
DEFAULT_VISION_FALLBACK_MODELS: List[str] = [
    MODEL_1,
    MODEL_2,
    MODEL_3,
    MODEL_4,
    MODEL_5,
]

# Default Task Models & Fallback Lists
DEFAULT_VISION_VQA_MODEL = DEFAULT_VISION_PRIMARY_MODEL
DEFAULT_VISION_VQA_FALLBACKS = [
    DEFAULT_VISION_SECONDARY_MODEL,
    DEFAULT_VISION_TERTIARY_MODEL,
    DEFAULT_VISION_QUATERNARY_MODEL,
    DEFAULT_VISION_QUINARY_MODEL,
]

DEFAULT_VISION_CAPTION_MODEL = DEFAULT_VISION_PRIMARY_MODEL
DEFAULT_VISION_CAPTION_FALLBACKS = [
    DEFAULT_VISION_SECONDARY_MODEL,
    DEFAULT_VISION_TERTIARY_MODEL,
    DEFAULT_VISION_QUATERNARY_MODEL,
    DEFAULT_VISION_QUINARY_MODEL,
]

DEFAULT_VISION_GROUND_MODEL = DEFAULT_VISION_SECONDARY_MODEL
DEFAULT_VISION_GROUND_FALLBACKS = [
    DEFAULT_VISION_PRIMARY_MODEL,
    DEFAULT_VISION_TERTIARY_MODEL,
    DEFAULT_VISION_QUATERNARY_MODEL,
    DEFAULT_VISION_QUINARY_MODEL,
]

# User-friendly short aliases
MODEL_ALIASES: Dict[str, str] = {
    "model_1": MODEL_1,
    "model_2": MODEL_2,
    "model_3": MODEL_3,
    "model_4": MODEL_4,
    "model_5": MODEL_5,
    "model1": MODEL_1,
    "model2": MODEL_2,
    "model3": MODEL_3,
    "model4": MODEL_4,
    "model5": MODEL_5,
    "gemma_26b": MODEL_SLUGS["gemma_26b"],
    "gemma-26b": MODEL_SLUGS["gemma_26b"],
    "gemma-4-26b": MODEL_SLUGS["gemma_26b"],
    "gemma-4-26b-a4b-it": MODEL_SLUGS["gemma_26b"],
    "gemma": MODEL_SLUGS["gemma_26b"],
    "google": MODEL_SLUGS["gemma_26b"],
    "google_gemma": MODEL_SLUGS["gemma_26b"],
    "gemma_31b": MODEL_SLUGS["gemma_31b"],
    "gemma-31b": MODEL_SLUGS["gemma_31b"],
    "gemma-4-31b": MODEL_SLUGS["gemma_31b"],
    "gemma-4-31b-it": MODEL_SLUGS["gemma_31b"],
    "inkling": MODEL_SLUGS["inkling"],
    "inkling:free": MODEL_SLUGS["inkling"],
    "thinkingmachines/inkling:free": MODEL_SLUGS["inkling"],
    "nemotron": MODEL_SLUGS["nemotron_nano"],
    "nemotron_nano": MODEL_SLUGS["nemotron_nano"],
    "nemotron-3-nano": MODEL_SLUGS["nemotron_nano"],
    "minimax": MODEL_SLUGS["minimax_m3"],
    "minimax_m3": MODEL_SLUGS["minimax_m3"],
    "minimax-m3": MODEL_SLUGS["minimax_m3"],
    "qwen25": MODEL_SLUGS["qwen25_free"],
    "qwen25_free": MODEL_SLUGS["qwen25_free"],
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
    quaternary_model: str = DEFAULT_VISION_QUATERNARY_MODEL
    quinary_model: str = DEFAULT_VISION_QUINARY_MODEL
    fallback_models: tuple[str, ...] = tuple(DEFAULT_VISION_FALLBACK_MODELS)
    model: str = DEFAULT_VISION_PRIMARY_MODEL
    vqa_model: str = DEFAULT_VISION_VQA_MODEL
    vqa_fallbacks: tuple[str, ...] = tuple(DEFAULT_VISION_VQA_FALLBACKS)
    caption_model: str = DEFAULT_VISION_CAPTION_MODEL
    caption_fallbacks: tuple[str, ...] = tuple(DEFAULT_VISION_CAPTION_FALLBACKS)
    ground_model: str = DEFAULT_VISION_GROUND_MODEL
    ground_fallbacks: tuple[str, ...] = tuple(DEFAULT_VISION_GROUND_FALLBACKS)
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: Optional[str] = None
    geochat_base_url: str = "http://100.108.110.84:8000/"
    geochat_api_key: Optional[str] = "252fa18193252fa18197"
    timeout: float = 120.0
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
        
        raw_primary = os.environ.get("VISION_MODEL") or os.environ.get("VISION_PRIMARY_MODEL") or os.environ.get("VISION_MODEL_1")
        primary_model = resolve_model_slug(raw_primary, default=DEFAULT_VISION_PRIMARY_MODEL)

        raw_secondary = os.environ.get("VISION_SECONDARY_MODEL") or os.environ.get("VISION_MODEL_2")
        secondary_model = resolve_model_slug(raw_secondary, default=DEFAULT_VISION_SECONDARY_MODEL)

        raw_tertiary = os.environ.get("VISION_TERTIARY_MODEL") or os.environ.get("VISION_MODEL_3")
        tertiary_model = resolve_model_slug(raw_tertiary, default=DEFAULT_VISION_TERTIARY_MODEL)

        raw_quaternary = os.environ.get("VISION_QUATERNARY_MODEL") or os.environ.get("VISION_MODEL_4")
        quaternary_model = resolve_model_slug(raw_quaternary, default=DEFAULT_VISION_QUATERNARY_MODEL)

        raw_quinary = os.environ.get("VISION_QUINARY_MODEL") or os.environ.get("VISION_MODEL_5")
        quinary_model = resolve_model_slug(raw_quinary, default=DEFAULT_VISION_QUINARY_MODEL)

        # Fallback sequence
        raw_fb_models = os.environ.get("VISION_FALLBACK_MODELS")
        default_fb_list = [primary_model, secondary_model, tertiary_model, quaternary_model, quinary_model]
        fallback_models = tuple(parse_model_list(raw_fb_models, default_fb_list))

        # Task-specific Primary Models
        raw_vqa = os.environ.get("VISION_VQA_MODEL")
        vqa_model = resolve_model_slug(raw_vqa, default=primary_model) if raw_vqa else primary_model

        raw_cap = os.environ.get("VISION_CAPTION_MODEL")
        caption_model = resolve_model_slug(raw_cap, default=primary_model) if raw_cap else primary_model

        raw_grd = os.environ.get("VISION_GROUND_MODEL")
        ground_model = resolve_model_slug(raw_grd, default=secondary_model) if raw_grd else secondary_model

        # Task-specific Fallbacks
        raw_vqa_fb = os.environ.get("VISION_VQA_FALLBACKS")
        default_vqa_fbs = [m for m in fallback_models if m != vqa_model]
        vqa_fallbacks = tuple(parse_model_list(raw_vqa_fb, default_vqa_fbs))

        raw_cap_fb = os.environ.get("VISION_CAPTION_FALLBACKS")
        default_cap_fbs = [m for m in fallback_models if m != caption_model]
        caption_fallbacks = tuple(parse_model_list(raw_cap_fb, default_cap_fbs))

        raw_grd_fb = os.environ.get("VISION_GROUND_FALLBACKS")
        default_grd_fbs = [primary_model, tertiary_model, quaternary_model, quinary_model]
        ground_fallbacks = tuple(parse_model_list(raw_grd_fb, default_grd_fbs))

        base_url = os.environ.get("VISION_BASE_URL", os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
        
        # Check OPENROUTER_API_KEY, fallback to VISION_API_KEY
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("VISION_API_KEY")

        timeout_str = os.environ.get("VISION_TIMEOUT", "120.0")
        try:
            timeout = float(timeout_str)
        except ValueError:
            timeout = 120.0

        retries_str = os.environ.get("VISION_MAX_RETRIES", "2")
        try:
            max_retries = int(retries_str)
        except ValueError:
            max_retries = 2

        geochat_base_url = os.environ.get("GEOCHAT_BASE_URL", "http://100.108.110.84:8000/")
        geochat_api_key = os.environ.get("GEOCHAT_API_KEY", "252fa18193252fa18197")

        return cls(
            provider=provider,
            primary_model=primary_model,
            secondary_model=secondary_model,
            tertiary_model=tertiary_model,
            quaternary_model=quaternary_model,
            quinary_model=quinary_model,
            fallback_models=fallback_models,
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
            candidates = list(self.fallback_models) if self.fallback_models else [self.model, self.secondary_model, self.tertiary_model, self.quaternary_model, self.quinary_model]

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
            f"quaternary_model='{self.quaternary_model}', quinary_model='{self.quinary_model}', "
            f"vqa_model='{self.vqa_model}', ground_model='{self.ground_model}', "
            f"base_url='{self.base_url}', api_key={masked}, "
            f"geochat_base_url='{self.geochat_base_url}', geochat_api_key={masked_gc}, "
            f"timeout={self.timeout}, max_retries={self.max_retries})"
        )

