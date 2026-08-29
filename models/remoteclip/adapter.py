"""
RemoteCLIPAdapter — wraps RemoteCLIP vision-language foundation model.

Capability extracted: vision-language similarity/retrieval over satellite imagery.
RemoteCLIP is a CLIP-style model fine-tuned on remote-sensing data.

Use cases in SatQuery AI:
  - Image retrieval: find images most similar to a text description
  - Zero-shot classification: score how well a scene matches a category label
  - Feature extraction: dense image embeddings for downstream tasks

Integration plan (Phase 1):
  1. pip install open_clip_torch
  2. Weights available via HuggingFace: chendelong/RemoteCLIP
  3. Implement _load() to load the CLIP model
  4. Implement encode_image() and encode_text() for embedding extraction
  5. Implement similarity() for zero-shot scoring
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np


class RemoteCLIPAdapter:
    """Thin wrapper isolating RemoteCLIP model calls."""

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "chendelong/RemoteCLIP",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _load(self) -> None:
        """
        Load RemoteCLIP via open_clip.
        Phase 1: call open_clip.create_model_and_transforms() with pretrained weights.
        """
        raise NotImplementedError(
            "RemoteCLIPAdapter._load() requires open_clip. "
            "Run: pip install open_clip_torch && "
            "huggingface-cli download chendelong/RemoteCLIP"
        )

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """
        Compute image embedding.

        Returns:
            1-D float32 numpy array of dimension 768 (ViT-L-14).
        """
        raise NotImplementedError("RemoteCLIPAdapter.encode_image() requires Phase 1 implementation.")

    def encode_text(self, text: str) -> np.ndarray:
        """
        Compute text embedding.

        Returns:
            1-D float32 numpy array of dimension 768.
        """
        raise NotImplementedError("RemoteCLIPAdapter.encode_text() requires Phase 1 implementation.")

    def similarity(self, image: np.ndarray, labels: list[str]) -> dict[str, float]:
        """
        Score how well the image matches each label (zero-shot classification).

        Args:
            image: RGB numpy array.
            labels: List of text labels to score against.

        Returns:
            dict mapping each label to a cosine-similarity score in [-1, 1].
        """
        raise NotImplementedError("RemoteCLIPAdapter.similarity() requires Phase 1 implementation.")
