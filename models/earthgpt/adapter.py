"""
EarthGPTAdapter — wraps EarthGPT optical+SAR multimodal analysis.

Capability extracted: joint optical+SAR scene understanding and classification.

Reference: EarthGPT is a unified multimodal LLM for remote-sensing interpretation
of both optical and SAR imagery.

Integration plan (Phase 1):
  1. Obtain EarthGPT model weights (see models/earthgpt/README.md)
  2. Implement _load() to instantiate the model
  3. Implement fuse() to run optical+SAR joint inference
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np


class EarthGPTAdapter:
    """Thin wrapper isolating EarthGPT model calls."""

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._model = None

    def _load(self) -> None:
        """
        Load EarthGPT weights.
        Phase 1: instantiate the multimodal model and load checkpoint.
        """
        raise NotImplementedError(
            "EarthGPTAdapter._load() requires EarthGPT weights. "
            "See models/earthgpt/README.md for setup instructions."
        )

    def fuse(self, optical: "np.ndarray", sar: "np.ndarray",
             question: Optional[str] = None) -> dict:
        """
        Run joint optical+SAR analysis.

        Args:
            optical: Optical image, RGB numpy array (H, W, 3) uint8.
            sar: SAR image, grayscale or RGB numpy array (H, W[, C]) float32.
            question: Optional question for VQA mode; if None, runs captioning.

        Returns:
            {
              "answer": str,
              "class_map": np.ndarray (H, W) int  (per-pixel land-cover class),
              "class_stats": dict[str, float],     (class_name → coverage %)
              "confidence": float,
            }
        """
        raise NotImplementedError("EarthGPTAdapter.fuse() requires Phase 1 implementation.")
