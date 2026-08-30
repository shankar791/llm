"""
ChangeFormerAdapter — Standalone inference adapter for ChangeFormer (Official Siamese Architecture).
Isolates preprocessing, checkpoint loading, PyTorch tensor execution, and mask extraction.
"""
from __future__ import annotations
import io
import time
import os
from typing import Any, Dict, Optional, Tuple, Union, Literal
import numpy as np
from PIL import Image
import torch
from .network import ChangeFormerModel


class ChangeFormerAdapter:
    """
    Inference adapter wrapping official ChangeFormer bi-temporal change detection model.
    Accepts two aligned input images (bytes, numpy arrays, or PIL Images),
    applies standard ImageNet normalization, executes forward pass on CPU/GPU,
    and returns a binary 2D change mask with quantitative metadata.
    """

    DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "ChangeFormer_MNCD256.safetensors")

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None,
                 mode: Literal["real", "mock"] = "real"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path or (self.DEFAULT_CHECKPOINT if os.path.exists(self.DEFAULT_CHECKPOINT) else None)
        self.mode = mode
        self._model: Optional[ChangeFormerModel] = None
        self._is_loaded = False
        self._load_info: Dict[str, Any] = {}

    def load(self, checkpoint_path: Optional[str] = None, mode: Optional[Literal["real", "mock"]] = None) -> Dict[str, Any]:
        """
        Instantiate ChangeFormer architecture and load checkpoint weights.
        In 'real' mode, requires a valid, verified checkpoint.
        In 'mock' mode, uses initialized architecture explicitly marked as mock.
        """
        start_time = time.perf_counter()
        target_mode = mode or self.mode
        path = checkpoint_path or self.checkpoint_path

        model = ChangeFormerModel()
        total_params = sum(p.numel() for p in model.parameters())

        if target_mode == "real":
            if not path or not os.path.exists(path):
                raise FileNotFoundError(
                    f"Real ChangeFormer inference requires a valid checkpoint path, but none was found at: {path!r}. "
                    f"Set mode='mock' for local mock testing or provide a valid .safetensors/.pth checkpoint."
                )

            # Load checkpoint strictly
            if path.endswith(".safetensors"):
                import safetensors.torch
                state_dict = safetensors.torch.load_file(path)
            else:
                state_dict = torch.load(path, map_location=self.device)
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]

            # Check if CD_model prefix is needed
            if not any(k.startswith("CD_model.") for k in state_dict.keys()):
                state_dict = {"CD_model." + k: v for k, v in state_dict.items()}

            load_res = model.load_state_dict(state_dict, strict=True)
            missing_keys_count = len(load_res.missing_keys)
            unexpected_keys_count = len(load_res.unexpected_keys)

            if missing_keys_count > 0 or unexpected_keys_count > 0:
                raise RuntimeError(
                    f"Checkpoint state_dict mismatch! Missing: {load_res.missing_keys}, Unexpected: {load_res.unexpected_keys}"
                )

            checkpoint_name = os.path.basename(path)
            is_mock = False

        else:
            # Mock mode: initialized model without trained weights
            checkpoint_name = "mock_initialized"
            missing_keys_count = 0
            unexpected_keys_count = 0
            is_mock = True

        model.to(self.device)
        model.eval()
        self._model = model
        self._is_loaded = True
        self.mode = target_mode
        self.checkpoint_path = path

        load_duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        self._load_info = {
            "checkpoint": checkpoint_name,
            "checkpoint_path": path,
            "mode": target_mode,
            "is_mock": is_mock,
            "total_parameters": total_params,
            "loaded_parameters": total_params if not is_mock else 0,
            "missing_keys_count": missing_keys_count,
            "unexpected_keys_count": unexpected_keys_count,
            "device": self.device,
            "load_time_ms": load_duration_ms
        }
        return self._load_info

    def _to_pil_rgb(self, img_input: Union[bytes, np.ndarray, Image.Image]) -> Image.Image:
        """Convert various input types (bytes, GeoTIFF, numpy array, PIL Image) to standard RGB PIL Image."""
        if isinstance(img_input, bytes):
            if len(img_input) == 0:
                raise ValueError("Empty image byte buffer provided.")
            # Check for GeoTIFF magic bytes (Little-endian 'II*\0' or Big-endian 'MM\0*')
            if img_input.startswith(b"II*\x00") or img_input.startswith(b"MM\x00*"):
                try:
                    from gis.raster import GeoTIFFReader
                    arr_rgb, _ = GeoTIFFReader.read_rgb(img_input)
                    return Image.fromarray(arr_rgb).convert("RGB")
                except Exception:
                    pass
            return Image.open(io.BytesIO(img_input)).convert("RGB")
        elif isinstance(img_input, Image.Image):
            return img_input.convert("RGB")
        elif isinstance(img_input, np.ndarray):
            if img_input.ndim == 2:
                return Image.fromarray(img_input).convert("RGB")
            elif img_input.ndim == 3:
                if img_input.shape[0] in {1, 3, 4} and img_input.shape[2] not in {1, 3, 4}:
                    img_input = np.transpose(img_input, (1, 2, 0))
                if img_input.shape[2] > 3:
                    img_input = img_input[:, :, :3]
                if img_input.dtype != np.uint8:
                    img_input = (np.clip(img_input, 0, 1) * 255).astype(np.uint8) if img_input.max() <= 1.0 else img_input.astype(np.uint8)
                return Image.fromarray(img_input).convert("RGB")
            else:
                raise ValueError(f"Unsupported numpy array dimensions: {img_input.shape}")
        else:
            raise TypeError(f"Unsupported input type for image: {type(img_input)}")

    def preprocess(self, img_input: Union[bytes, np.ndarray, Image.Image],
                   target_size: Tuple[int, int] = (256, 256)) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Convert input image to normalized PyTorch tensor (1, 3, H, W).
        Returns:
            tensor: (1, 3, H, W) normalized float tensor
            original_size: (orig_width, orig_height)
        """
        pil_img = self._to_pil_rgb(img_input)
        orig_size = pil_img.size  # (width, height)

        resized = pil_img.resize(target_size, Image.BILINEAR)
        arr = np.array(resized).astype(np.float32) / 255.0  # [0.0, 1.0]

        # ImageNet mean & std normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device), orig_size

    def detect(self, image_t0: Union[bytes, np.ndarray, Image.Image],
               image_t1: Union[bytes, np.ndarray, Image.Image],
               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute ChangeFormer bi-temporal change detection inference.

        Args:
            image_t0: Earlier acquisition image (bytes, numpy array, or PIL Image)
            image_t1: Later acquisition image (same format/modality)
            metadata: Optional contextual metadata dictionary

        Returns:
            Dictionary matching the standalone contract:
            {
                "status": "success",
                "change_mask": np.ndarray (H, W) uint8 (1=changed, 0=unchanged),
                "metadata": {
                    "model": "ChangeFormer",
                    "variant": "Official",
                    "checkpoint": str,
                    "mode": "real" | "mock",
                    "is_mock": bool,
                    "input_shape": [H, W, 3],
                    "output_shape": [H, W],
                    "inference_time_ms": float,
                    "changed_pixels": int,
                    "total_pixels": int,
                    "change_fraction": float
                }
            }
        """
        if image_t0 is None or image_t1 is None:
            raise ValueError("Both image_t0 and image_t1 are strictly required for change detection.")

        if not self._is_loaded or self._model is None:
            self.load()

        start_time = time.perf_counter()

        # 1. Preprocess both temporal scenes
        t0_tensor, orig_size_t0 = self.preprocess(image_t0)
        t1_tensor, orig_size_t1 = self.preprocess(image_t1)

        # 2. Execute PyTorch forward pass
        with torch.no_grad():
            logits = self._model(t0_tensor, t1_tensor)  # (1, 2, 256, 256)
            pred = torch.argmax(logits, dim=1).squeeze(0)  # (256, 256)
            mask_np = pred.cpu().numpy().astype(np.uint8)

        # 3. Resize mask back to original resolution if needed
        if orig_size_t0 != (256, 256):
            mask_pil = Image.fromarray(mask_np * 255).resize(orig_size_t0, Image.NEAREST)
            final_mask = (np.array(mask_pil) > 0).astype(np.uint8)
        else:
            final_mask = mask_np

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        total_pixels = int(final_mask.size)
        changed_pixels = int(np.count_nonzero(final_mask))
        change_fraction = round(float(changed_pixels / total_pixels), 4)

        return {
            "status": "success",
            "change_mask": final_mask,
            "metadata": {
                "model": "ChangeFormer",
                "variant": "Official",
                "checkpoint": self._load_info.get("checkpoint", "unknown"),
                "mode": self.mode,
                "is_mock": self._load_info.get("is_mock", False),
                "input_shape": [orig_size_t0[1], orig_size_t0[0], 3],
                "output_shape": list(final_mask.shape),
                "inference_time_ms": elapsed_ms,
                "changed_pixels": changed_pixels,
                "total_pixels": total_pixels,
                "change_fraction": change_fraction,
                **(metadata or {})
            }
        }


