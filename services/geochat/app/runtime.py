"""
GeoChat Runtime — Singleton model loader configured for Hugging Face Inference Endpoints.

Mounts /repository (Hugging Face custom container model mount) or falls back to HuggingFace Hub.
Uses GeoChatLlamaForCausalLM from the official GeoChat source, NOT generic AutoModelForCausalLM.
Model is loaded ONCE at startup via FastAPI lifespan and kept resident in GPU memory.
"""
from __future__ import annotations
import os
import sys
import time
import logging
from typing import Optional, Dict, Any

import torch

logger = logging.getLogger("geochat.runtime")


class GeoChatRuntime:
    """
    Singleton runtime holding the loaded GeoChat-7B model, tokenizer, and image processor.
    Loads once at service startup. All inference requests reuse this single instance.
    """

    _instance: Optional["GeoChatRuntime"] = None

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.conv_mode = "llava_v1"
        self.device: str = os.environ.get("GEOCHAT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

        # Prioritize /repository (HF custom container model mount) -> GEOCHAT_MODEL_PATH -> GEOCHAT_MODEL
        env_model_path = os.environ.get("GEOCHAT_MODEL_PATH", "/repository")
        if os.path.exists(env_model_path) and os.path.isfile(os.path.join(env_model_path, "config.json")):
            self.model_path = env_model_path
        elif os.path.exists("/repository") and os.path.isfile("/repository/config.json"):
            self.model_path = "/repository"
        else:
            self.model_path = os.environ.get("GEOCHAT_MODEL", "MBZUAI/geochat-7B")

        self.load_8bit: bool = os.environ.get("GEOCHAT_LOAD_8BIT", "true").lower() == "true"
        self.max_new_tokens: int = int(os.environ.get("MAX_NEW_TOKENS", "256"))
        self.is_loaded: bool = False
        self.load_info: Dict[str, Any] = {}
        self._parameter_count: int = 0

    @classmethod
    def get_instance(cls) -> "GeoChatRuntime":
        """Return the singleton runtime instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self) -> Dict[str, Any]:
        """
        Load the GeoChat model using the OFFICIAL GeoChat loading mechanism.
        Uses /repository mounted weights directly if available.

        Called ONCE at startup. Never per-request.
        """
        if self.is_loaded:
            return self.load_info

        start_time = time.perf_counter()
        logger.info(f"Loading GeoChat model from: {self.model_path} on device={self.device}, 8bit={self.load_8bit}")

        try:
            # ----------------------------------------------------------------
            # OFFICIAL GEOCHAT LOADING PATH
            # Uses GeoChatLlamaForCausalLM and official GeoChat loader.
            # ----------------------------------------------------------------
            geochat_repo_path = os.environ.get("GEOCHAT_REPO_PATH", "/app/GeoChat")
            if os.path.isdir(geochat_repo_path) and geochat_repo_path not in sys.path:
                sys.path.insert(0, geochat_repo_path)
                logger.info(f"Added GeoChat repo to sys.path: {geochat_repo_path}")

            try:
                from geochat.model import load_pretrained_model as geochat_load
                logger.info("Using official GeoChat model loader (geochat.model.load_pretrained_model)")

                tokenizer, model, image_processor, context_len = geochat_load(
                    model_path=self.model_path,
                    model_base=None,
                    model_name="geochat-7b",
                    load_8bit=self.load_8bit,
                    load_4bit=False,
                )
            except ImportError:
                try:
                    from llava.model import load_pretrained_model as llava_load
                    logger.info("Using LLaVA model loader (llava.model.load_pretrained_model)")

                    tokenizer, model, image_processor, context_len = llava_load(
                        model_path=self.model_path,
                        model_base=None,
                        model_name="geochat-7b",
                        load_8bit=self.load_8bit,
                        load_4bit=False,
                    )
                except ImportError:
                    raise RuntimeError(
                        "Neither geochat.model nor llava.model loaders are available. "
                        "Ensure the GeoChat repository is cloned at GEOCHAT_REPO_PATH "
                        f"(currently: {geochat_repo_path})."
                    )

            self.model = model
            self.tokenizer = tokenizer
            self.image_processor = image_processor

            # Ensure model is in eval mode
            self.model.eval()

            # Count parameters
            self._parameter_count = sum(p.numel() for p in self.model.parameters())

            # Determine actual device
            try:
                first_param = next(self.model.parameters())
                actual_device = str(first_param.device)
                actual_dtype = str(first_param.dtype)
            except StopIteration:
                actual_device = self.device
                actual_dtype = "unknown"

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            self.is_loaded = True
            self.load_info = {
                "model_name": self.model_path,
                "model_class": self.model.__class__.__name__,
                "tokenizer_class": self.tokenizer.__class__.__name__,
                "image_processor_class": self.image_processor.__class__.__name__,
                "device": actual_device,
                "dtype": actual_dtype,
                "parameter_count": self._parameter_count,
                "load_8bit": self.load_8bit,
                "load_time_ms": elapsed_ms,
            }

            logger.info(
                f"GeoChat loaded successfully: class={self.load_info['model_class']}, "
                f"params={self._parameter_count:,}, device={actual_device}, "
                f"time={elapsed_ms:.0f}ms"
            )
            return self.load_info

        except Exception as e:
            self.is_loaded = False
            self.model = None
            logger.error(f"FATAL: GeoChat model loading failed: {e}")
            raise RuntimeError(f"Failed to load GeoChat model: {e}") from e

    @property
    def parameter_count(self) -> int:
        return self._parameter_count
