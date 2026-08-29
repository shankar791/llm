# ChangeFormer Model Adapter

ChangeFormer is a **Siamese Vision Transformer (ViT)** architecture for bi-temporal change detection in remote-sensing imagery. It produces pixel-level binary change masks showing which regions changed between two co-registered images of the same scene.

**Paper:** [ChangeFormer: A Transformer-Based Siamese Network for Change Detection](https://arxiv.org/abs/2201.01293)  
**Official repo:** https://github.com/wgcban/ChangeFormer

---

## Capability Used by SatQuery AI

| Capability | Tool | Description |
|---|---|---|
| **Bi-Temporal Change Detection** | `T4_Change` | Pixel-level binary change mask between two co-registered satellite images |

**Output schema:**
```python
{
    "change_mask":     np.ndarray,   # (H, W) bool — True = changed pixel
    "change_fraction": float,        # fraction of pixels changed, in [0, 1]
    "confidence":      float,        # model confidence score, in [0, 1]
}
```

## Capabilities Intentionally Excluded

- **Multi-class semantic change** — SatQuery uses the binary mask only; class labels are derived from VQA
- **Training/fine-tuning pipeline** — inference-only integration

---

## Phase 1 Setup Instructions

### 1. Clone ChangeFormer source

```bash
git clone https://github.com/wgcban/ChangeFormer models/changeformer/changeformer_src
```

### 2. Install dependencies

```bash
pip install torch torchvision einops timm
```

### 3. Download pretrained weights

Weights are available on the [ChangeFormer GitHub releases page](https://github.com/wgcban/ChangeFormer/releases).

Recommended checkpoint: **ChangeFormer-B4** trained on LEVIR-CD.

```bash
# Place checkpoint at:
models/changeformer/weights/ChangeFormer_LEVIR.pth
```

### 4. Implement `ChangeFormerAdapter._load()`

```python
import sys
sys.path.insert(0, "models/changeformer/changeformer_src")
from models.ChangeFormer import ChangeFormerV6

self._model = ChangeFormerV6()
ckpt = torch.load(self.weights_path, map_location=self.device)
self._model.load_state_dict(ckpt["model_state_dict"])
self._model.eval().to(self.device)
```

### 5. Wire to tool

Update `tools/change_detection.py` to call `ChangeFormerAdapter.detect()`.

---

## Resource Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 6 GB | 12 GB |
| CPU RAM | 8 GB | 16 GB |
| Disk | 500 MB | 500 MB |

Both images must be **co-registered** (same geographic extent and pixel size) before passing to the adapter.
