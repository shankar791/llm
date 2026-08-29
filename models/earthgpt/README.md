# EarthGPT Model Adapter

EarthGPT is a **unified multimodal large language model** for remote-sensing image interpretation. Unlike optical-only models, EarthGPT jointly processes both **optical** (RGB/multispectral) and **SAR** (Synthetic Aperture Radar) imagery in a single forward pass, enabling more reliable land-cover analysis when both modalities are available.

**Paper:** [EarthGPT: A Universal Multimodal Large Language Model for Multisensor Image Comprehension in Remote Sensing Domain](https://arxiv.org/abs/2401.16822)

---

## Capability Used by SatQuery AI

| Capability | Tool | Description |
|---|---|---|
| **Optical + SAR Fusion** | `T5_OpticalSAR` | Joint scene understanding when both optical and SAR images are uploaded |

**Output schema:**
```python
{
    "answer":      str,                   # natural-language answer or caption
    "class_map":   np.ndarray,            # (H, W) int — per-pixel land-cover class index
    "class_stats": dict[str, float],      # class_name → coverage percentage
    "confidence":  float,                 # model confidence in [0, 1]
}
```

## Capabilities Intentionally Excluded

- **Optical-only mode** — GeoChat handles single-modality optical queries (T1/T2/T3)
- **Multispectral band analysis** — only RGB/pseudocolor input is used

---

## Phase 1 Setup Instructions

> **Note:** EarthGPT weights are available from the authors upon request. Check the paper's GitHub repository for the latest access instructions.

### 1. Obtain weights

Follow the instructions at the EarthGPT paper's official repository. Place the downloaded checkpoint at:

```
models/earthgpt/weights/earthgpt.pth
```

### 2. Install dependencies

```bash
pip install torch torchvision transformers Pillow
```

### 3. Implement `EarthGPTAdapter._load()`

Replace the stub with the appropriate model-loading code from the EarthGPT repository. Set `self.model_path` in the constructor to point to the checkpoint directory.

### 4. Wire to tool

Update `tools/optical_sar.py` to call `EarthGPTAdapter.fuse(optical_img, sar_img)`.

---

## Input Requirements

- **Optical image:** RGB, uint8, any resolution (will be resized internally)
- **SAR image:** Grayscale or 3-channel pseudo-color, float32, **must be co-registered with optical**
- Both images should cover the same geographic extent

## Resource Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 16 GB | 24 GB |
| CPU RAM | 32 GB | 64 GB |
| Disk | 20 GB | 20 GB |
