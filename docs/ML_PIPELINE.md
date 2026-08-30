# Machine Learning Pipeline & Specialist Model Adapters

This document details the external machine learning models integrated into SatQuery AI, their adapter interfaces, isolation boundaries, and reconnaissance findings.

> **CRITICAL RULE**: External model implementations must never be imported directly into the API or Master Agent layers.
> Every specialist model must be wrapped inside an isolated adapter conforming to `tools/base.py::BaseTool`.

---

## Model Adapter Directory Layout

```
ai/
  ├── llm/                        # Provider-agnostic text LLM foundation (OpenRouter / OpenAI)
  ├── vision/                     # Resilient Multimodal VisionProvider subsystem
  │     ├── base.py               # VisionProvider Protocol, GroundingBox, VisionResponse
  │     ├── openrouter_qwen.py    # OpenRouterVisionProvider with multi-model fallback & rate-limit resilience
  │     ├── config.py             # VisionConfig (Primary, Secondary, Tertiary model hierarchy & task fallbacks)
  │     └── mock.py               # MockVisionProvider
  ├── intent/                     # LLM Intent Classifier
  └── synthesis/                  # Evidence-Grounded LLM Synthesizer

models/
  ├── geochat/
  │     ├── adapter.py            # GeoChatAdapter (VQA, Captioning, Grounding)
  │     └── [isolated model code & weights]
  ├── changeformer/
  │     ├── adapter.py            # ChangeDetectionAdapter (Bi-temporal Change)
  │     └── [isolated model code & weights]
  ├── earthgpt/
  │     ├── adapter.py            # EarthGPTAdapter (Optical + SAR Fusion)
  │     └── [isolated model code & weights]
  └── remoteclip/
        ├── adapter.py            # RemoteCLIPAdapter (Zero-Shot Fallback)
        └── [isolated model code & weights]

tools/
  ├── base.py                     # BaseTool ABC defining run(**kwargs) -> dict
  ├── registry.py                 # ToolRegistry & ToolDefinition allowlist
  ├── vqa.py                      # VQATool (T1_VQA via VisionProvider / GeoChat)
  ├── captioning.py               # CaptionTool (T2_Caption via VisionProvider / GeoChat)
  ├── grounding.py                # GroundingTool (T3_Ground via VisionProvider / GeoChat)
  ├── change_detection.py         # ChangeDetectionTool (T4_Change)
  ├── optical_sar.py              # OpticalSARTool (T5_OpticalSAR)
  └── fallback.py                 # FallbackTool (RemoteCLIP)
```

---

## Multimodal Vision Model Hierarchy & Task Routing (Step 12C)

The `ai.vision.OpenRouterVisionProvider` orchestrates a resilient multi-model selection and fallback layer for visual satellite analysis:

### 1. Vision Model Roster
- **Primary Vision Model**: `google/gemma-4-26b-a4b-it:free` (`VISION_PRIMARY_MODEL`)
- **Secondary Vision Model**: `google/gemma-4-31b-it:free` (`VISION_SECONDARY_MODEL`)
- **Tertiary Vision Model**: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` (`VISION_TERTIARY_MODEL`)

### 2. Task Routing Policy
- **T1 VQA**: `VISION_VQA_MODEL` (default: Gemma 4 26B) -> `VISION_VQA_FALLBACKS` (Gemma 4 31B, Nemotron 3 Nano Omni)
- **T2 Caption**: `VISION_CAPTION_MODEL` (default: Gemma 4 26B) -> `VISION_CAPTION_FALLBACKS` (Gemma 4 31B, Nemotron 3 Nano Omni)
- **T3 Grounding**: `VISION_GROUND_MODEL` (default: Gemma 4 31B) -> `VISION_GROUND_FALLBACKS` (Gemma 4 26B, Nemotron 3 Nano Omni)

### 3. Error Classification & Fallback Behavior
- **Transient Failures (Fallback Triggered)**: Upstream 429 concurrency pool limits, 5xx server errors, socket timeouts, and provider connection drops automatically trigger fallback to the next candidate model in the chain.
- **Account-Level Rate Limits (Immediate Fail-Fast)**: If OpenRouter returns `free-models-per-day` or account quota exhaustion, the system classifies it as `ACCOUNT_RATE_LIMIT` and immediately halts to avoid wasting API calls across fallback models.
- **Non-Transient Errors (No Fallback)**: Malformed application inputs (400) or missing credentials (401/403) fail immediately without fallback loops.
- **Strict Grounding Validation**: Requires machine-readable bounding boxes with normalized coordinates $0 \le x_0 \le x_1 \le 1, 0 \le y_0 \le y_1 \le 1$. If a candidate returns unstructured natural language, it is marked `grounding_unsupported` and fails over to the next candidate model without fabricating fake boxes.
- **Observability**: `VisionResponse` and tool metadata explicitly record `selected_model`, `attempted_models`, `fallback_used`, `fallback_reason`, and `latency_ms`.


---

## Change-Detection Specialist Reconnaissance (T4_Change)

| Feature | ChangeFormer (`wgcban/ChangeFormer`) | BIT-CD (`justchenhao/BIT_CD`) | TinyCD (`AndreaCodegoni/Tiny_model_4_CD`) | GalaxEye EO-SAR (`juggtimber/galaxeye...`) |
|---|---|---|---|---|
| **Architecture** | Siamese Transformer (SegFormer MiT-B0/B1) | ResNet18 + Transformer Decoder | Lightweight Siamese CNN + Cross-Attention | EfficientNet-B0 + UNet Decoder |
| **Parameters** | ~4.1M (MiT-B0) / ~13.5M (MiT-B1) | ~3.5M (ResNet18) | ~0.32M (316k parameters) | ~5.3M |
| **Pretrained Weights** | LEVIR-CD, DSIFN-CD, S2Looking, WHU-CD | LEVIR-CD, WHU-CD, CDD | LEVIR-CD, WHU-CD, DSIFN | HuggingFace Checkpoint (`pytorch_model.bin`) |
| **Input Format** | Paired images $(T_0, T_1)$, 3-channel RGB | Paired images $(T_0, T_1)$, 3-channel RGB | Paired images $(T_0, T_1)$, 3-channel RGB | Paired 5-channel stack (RGB + CLAHE SAR + Log SAR) |
| **Modality Scope** | Homogeneous Optical Pairs | Homogeneous Optical Pairs | Homogeneous Optical Pairs | Heterogeneous Cross-Modal (Optical + SAR) |
| **Image Dimensions** | $256 \times 256$ or $512 \times 512$ | $256 \times 256$ | $256 \times 256$ | $256 \times 256$ or $512 \times 512$ |
| **Output Type** | Binary Change Logits / Mask $(H, W)$ | Binary Change Logits / Mask $(H, W)$ | Binary Change Logits / Mask $(H, W)$ | Binary Change Probability Map $(H, W)$ |
| **CPU Feasibility** | Moderate (~250-450ms on CPU) | Fast (~120-200ms on CPU) | Extremely Fast (<50ms on CPU) | Fast (~100-180ms on CPU) |
| **Source Code License** | Non-commercial / Research | Non-commercial / Research | Non-commercial / Research | MIT / Apache 2.0 (VERIFY) |
| **Weight License** | Non-commercial / Research | Non-commercial / Research | Non-commercial / Research | Open / Research |
| **Primary Suitability** | High accuracy optical change detection | Strong baseline optical CD | Optimal for edge/CPU constraint | Specialized for Optical+SAR pair CD |

---

## Specialist Model Specifications

### 1. GeoChat (`mbzuai-oryx/GeoChat`)
- **Target Tasks**: Visual Question Answering (T1_VQA), Scene Captioning (T2_Caption), Text-Guided Grounding (T3_Ground).
- **Architecture**: LLaVA-1.5 multimodal architecture with remote-sensing visual backbone and projection layer.
- **Integration Concept**:
  - `GeoChatAdapter.vqa(image_bytes, question) -> dict`
  - `GeoChatAdapter.caption(image_bytes) -> dict`
  - `GeoChatAdapter.ground(image_bytes, target_text) -> List[BoundingBox]`
- **Evaluation Requirements Prior to Full Integration**:
  - `[ ]` Verify license terms for both code and released weights (VERIFY).
  - `[ ]` Isolate minimal inference script (extracting vision encoder and projection head).
  - `[ ]` Test ONNX opset 17 export for the vision encoder backbone.
  - `[ ]` Benchmark FP16 vs. INT8 inference latency on CPU.

---

### 2. Change Detection (`T4_Change` Specialist)
- **Target Task**: Bi-Temporal Change Detection (T4_Change).
- **Recommended Candidate**: `wgcban/ChangeFormer` (or `BIT_CD` as lightweight alternative).
- **Integration Concept**:
  - `ChangeDetectionAdapter.detect(image_t0_bytes, image_t1_bytes) -> np.ndarray` (returns binary 2D change mask).
  - `ChangeDetectionTool` receives binary mask, forwards to deterministic GIS engine for polygonization and area computation, and formats `ToolResult`.
- **Evaluation Requirements**:
  - `[ ]` Verify license terms and checkpoint availability (VERIFY).
  - `[ ]` Isolate model definition, preprocessing transform, and weight loader.
  - `[ ]` Verify behavior on input dimension variations ($256 \times 256$, $512 \times 512$).
  - `[ ]` Export to ONNX opset 17 with dynamic spatial dimensions.

---

### 3. EarthGPT (`wivizhang/EarthGPT`)
- **Target Task**: Optical + SAR Cross-Modal Fusion (T5_OpticalSAR).
- **Architecture**: Dual-encoder network aligning Optical (RGB/Multispectral) and Synthetic Aperture Radar (SAR backscatter amplitude/phase) representations.
- **Integration Concept**:
  - `EarthGPTAdapter.fuse(optical_bytes, sar_bytes) -> dict` (returns fused classification and feature metrics).
- **Evaluation Requirements**:
  - `[ ]` Verify license terms for model weights and MMRS-1M dataset (VERIFY).
  - `[ ]` Verify SAR preprocessing assumptions (polarization channel configuration: VV, VH, HH).
  - `[ ]` Export dual-encoder backbone to ONNX.

---

### 4. RemoteCLIP (`RemoteCLIP/RemoteCLIP`)
- **Target Task**: Zero-Shot Image-Text Retrieval & Fallback Classification.
- **Architecture**: ViT-B-32 trained with contrastive learning on large-scale remote sensing imagery.
- **Integration Concept**:
  - `RemoteCLIPAdapter.similarity(image_bytes, text_labels) -> Dict[str, float]`
- **Evaluation Requirements**:
  - `[ ]` Verify model license terms (VERIFY).
  - `[ ]` Export vision encoder to ONNX for lightweight fallback execution.

---

## LLM Intent Classifier Foundation (`ai/intent/`)

The natural-language intent layer uses `LLMIntentClassifier` (`ai/intent/classifier.py`) atop `ai.llm.LLMProvider`.

- **Output Contract**: Emits validated `IntentSchema` constrained to task literals `{'vqa', 'caption', 'ground', 'change', 'fusion'}`.
- **Tool Selection Isolation**: The LLM is explicitly forbidden from emitting tool IDs (`T1_VQA`..`T5_OpticalSAR`).
- **Compatibility Gate Boundary**: The deterministic `ToolCompatibilityRouter` reconciles validated intent against actual uploaded raster metadata before invoking specialist adapters.
- **Fail-Safe Fallback**: Any provider timeout, authentication failure, or malformed JSON triggers an explicit switch to `RuleBasedIntentClassifier` tagged with `classifier_source="rule_fallback"`.

## LLM Synthesis & Evidence Grounding (`ai/synthesis/`)

The response synthesis layer uses `LLMSynthesizer` (`ai/synthesis/llm.py`) atop `ai.llm.LLMProvider`.

- **Structured Evidence Input**: Consumes structured `ToolResult` summaries, `EvidenceItem` geometries/labels, and authoritative GIS metrics (`area_ha`, `polygon_count`, `change_fraction`).
- **Anti-Hallucination Post-Validation**: `SynthesisValidator` catches fabricated evidence IDs, ungrounded certainty claims, or numeric discrepancies (e.g. area hallucinations).
- **Deterministic High-Availability Fallback**: If LLM execution fails or is rejected by post-validation, `DeterministicFallbackFormatter` constructs a mathematically accurate, grounded summary directly from GIS and specialist tool data (`synthesis_source="deterministic_fallback"`).

## Mock-First Testing Paradigm

Every tool in `tools/*.py` must provide a valid deterministic mock implementation that executes without loading GPU/PyTorch weights:
```python
class MockChangeDetectionTool(BaseTool):
    tool_id = "T4_Change"
    def run(self, **kwargs) -> dict:
        return {
            "tool_id": "T4_Change",
            "answer": "[MOCK] 14.25 ha new construction detected across 14 clusters",
            "confidence": 0.88,
            "evidence": [],
            "metadata": {"mock": True, "status": "success"}
        }
```
This ensures the full LangGraph orchestration graph, API layer, and Leaflet frontend can be developed, tested, and verified end-to-end prior to model checkpoint integration.

