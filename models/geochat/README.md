# GeoChat Model Adapter

GeoChat is a grounded large vision-language model (VLM) fine-tuned specifically for **remote-sensing imagery** interpretation. It extends LLaVA with a remote-sensing instruction-tuning dataset (SkyScript) and adds region-grounding capability via referring expressions.

**Paper:** [GeoChat: Grounded Large Vision-Language Model for Remote Sensing](https://arxiv.org/abs/2311.15826)  
**Official repo:** https://github.com/mbzuai-oryx/GeoChat  
**HuggingFace model:** `MBZUAI/GeoChat-7B`

---

## Capabilities Used by SatQuery AI

| Capability | Tool | Description |
|---|---|---|
| **VQA** | `T1_VQA` | Answer natural-language questions about satellite imagery |
| **Scene Captioning** | `T2_Caption` | Generate structured descriptions of a scene |
| **Region Grounding** | `T3_Ground` | Localize image regions from text queries (returns pixel bounding boxes) |

## Capabilities Intentionally Excluded

- **Conversational history loop** — SatQuery uses its own `SessionStore` for multi-turn memory
- **Gradio demo UI** — replaced by the SatQuery React frontend
- **Scene classification head** — covered by EarthGPT with better multi-modal support

---

## Phase 1 Setup Instructions

### 1. Clone GeoChat into this directory

```bash
git clone https://github.com/mbzuai-oryx/GeoChat models/geochat/geochat_src
```

### 2. Install model dependencies

```bash
pip install -r models/geochat/geochat_src/requirements.txt
```

> **Note:** GeoChat requires `transformers>=4.36`, `torch>=2.0`, and `Pillow`. It is GPU-recommended but can run on CPU with reduced speed.

### 3. Download weights

```bash
huggingface-cli download MBZUAI/GeoChat-7B --local-dir models/geochat/weights
```

Or set `model_path` to `"MBZUAI/GeoChat-7B"` and let HuggingFace auto-download on first `_load()` call.

### 4. Implement `GeoChatAdapter._load()`

In `adapter.py`, replace the `_load()` stub with:

```python
from geochat_src.model.builder import load_pretrained_model
self._tokenizer, self._model, self._image_processor, _ = load_pretrained_model(
    model_path=self.model_path, device=self.device
)
```

### 5. Wire to tools

Update `tools/vqa.py`, `tools/captioning.py`, and `tools/grounding.py` to import and call `GeoChatAdapter`.

---

## Resource Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 14 GB | 24 GB |
| CPU RAM | 16 GB | 32 GB |
| Disk | 15 GB | 15 GB |
| CUDA | 11.8+ | 12.x |

CPU-only inference is supported but significantly slower (~10–30s per query).
