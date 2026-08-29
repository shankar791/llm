"""
ChangeFormerAdapter — wraps ChangeFormer bi-temporal change detection.

Capability extracted: pixel-level change detection between two co-registered images.

Integration plan (Phase 1):
  1. Clone: git clone https://github.com/wgcban/ChangeFormer models/changeformer/
  2. Download weights from the ChangeFormer GitHub releases
  3. Implement _load() to instantiate the Siamese ViT model
  4. Implement detect() to run inference and return a binary change mask
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np


class ChangeFormerAdapter:
    """Thin wrapper isolating ChangeFormer model calls."""

    def __init__(self, weights_path: Optional[str] = None, device: str = "cuda"):
        self.weights_path = weights_path
        self.device = device
        self._model = None

    def _load(self) -> None:
        """
        Load ChangeFormer weights.
        Phase 1: instantiate the Siamese Transformer architecture and load checkpoint.
        """
        raise NotImplementedError(
            "ChangeFormerAdapter._load() requires ChangeFormer weights. "
            "See models/changeformer/README.md for setup instructions."
        )

    def detect(self, img_t0: "np.ndarray", img_t1: "np.ndarray") -> dict:
        """
        Run bi-temporal change detection.

        Args:
            img_t0: Earlier image, RGB numpy array (H, W, 3) uint8.
            img_t1: Later image, same size and modality as img_t0.

        Returns:
            {
              "change_mask": np.ndarray (H, W) bool,
              "change_fraction": float,
              "confidence": float,
            }
        """
        raise NotImplementedError("ChangeFormerAdapter.detect() requires Phase 1 implementation.")
