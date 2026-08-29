"""
GeoChatAdapter — wraps GeoChat remote-sensing capabilities.

Capabilities extracted from GeoChat (https://github.com/mbzuai-oryx/GeoChat):
  - Remote-sensing VQA
  - Scene captioning
  - Text-guided region grounding
  - Scene classification

What we do NOT use:
  - GeoChat's Gradio demo UI
  - The conversational history loop (handled by SatQuery's SessionStore)

Integration plan (Phase 1):
  1. Clone GeoChat repo: git clone https://github.com/mbzuai-oryx/GeoChat models/geochat/
  2. Install requirements: pip install -r models/geochat/requirements.txt
  3. Download weights: huggingface-cli download MBZUAI/GeoChat-7B
  4. Implement __init__ to load the model
  5. Implement vqa(), caption(), ground() to call inference pipeline
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np


class GeoChatAdapter:
    """Thin wrapper isolating GeoChat model calls from the tool layer."""

    def __init__(
        self,
        model_path: str = "MBZUAI/GeoChat-7B",
        device: str = "cuda",
        load_on_init: bool = False,
    ):
        """
        Args:
            model_path: HuggingFace model ID or local path to GeoChat weights.
            device: 'cuda' or 'cpu'.
            load_on_init: If True, load model weights immediately; otherwise lazy-load.
        """
        self.model_path = model_path
        self.device = device
        self._model = None
        self._tokenizer = None
        if load_on_init:
            self._load()

    def _load(self) -> None:
        """
        Load GeoChat model and tokenizer from weights.

        Phase 1: import GeoChat inference pipeline and call model loader.
        This is deliberately isolated here so the rest of the codebase
        imports cleanly without GPU/model dependencies.
        """
        raise NotImplementedError(
            "GeoChatAdapter._load() requires GeoChat weights. "
            "See models/geochat/README.md for setup instructions."
        )

    def vqa(self, image: "np.ndarray", question: str) -> dict:
        """
        Run VQA inference on a satellite image.

        Args:
            image: RGB numpy array (H, W, 3) uint8.
            question: Natural-language question about the image.

        Returns:
            {"answer": str, "confidence": float}
        """
        raise NotImplementedError("GeoChatAdapter.vqa() requires Phase 1 implementation.")

    def caption(self, image: "np.ndarray") -> dict:
        """
        Generate a structured scene description.

        Args:
            image: RGB numpy array (H, W, 3) uint8.

        Returns:
            {"caption": str, "confidence": float}
        """
        raise NotImplementedError("GeoChatAdapter.caption() requires Phase 1 implementation.")

    def ground(self, image: "np.ndarray", query: str) -> dict:
        """
        Localize image regions matching a text query.

        Args:
            image: RGB numpy array (H, W, 3) uint8.
            query: Region-description query.

        Returns:
            {"boxes": list[list[int]], "labels": list[str], "scores": list[float]}
            where each box is [x0, y0, x1, y1] in pixel coordinates.
        """
        raise NotImplementedError("GeoChatAdapter.ground() requires Phase 1 implementation.")
