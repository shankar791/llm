# SatQuery AI — GeoChat VQA Service (Hugging Face Inference Endpoint)

## Overview

Self-contained containerized microservice that runs the **MBZUAI/geochat-7B** multimodal remote-sensing model for Visual Question Answering (VQA). Configured specifically for deployment on **Hugging Face Inference Endpoints** using a **Custom Container**.

---

## Architecture & Hugging Face Mount Strategy

```
┌─────────────────────────────────────────────────────────────┐
│       Hugging Face Inference Endpoint (Custom Container)    │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Model Volume: /repository (Mounted by Hugging Face)   │  │
│  │ - config.json, pytorch_model shards, tokenizer files  │  │
│  └──────────────────────────┬────────────────────────────┘  │
│                             │                               │
│  ┌──────────────────────────▼────────────────────────────┐  │
│  │ FastAPI Application (uvicorn:8000, workers: 1)        │  │
│  │ - GET  /health (Readiness probe)                      │  │
│  │ - POST /vqa    (Multimodal inference)                 │  │
│  └──────────────────────────┬────────────────────────────┘  │
│                             │                               │
│  ┌──────────────────────────▼────────────────────────────┐  │
│  │ GeoChatRuntime (Singleton)                            │  │
│  │ - Loaded ONCE at startup via lifespan handler         │  │
│  │ - Uses official GeoChatLlamaForCausalLM loader        │  │
│  │ - 8-bit quantization on GPU VRAM                      │  │
│  └──────────────────────────┬────────────────────────────┘  │
│                             │                               │
│  ┌──────────────────────────▼────────────────────────────┐  │
│  │ 504×504 Multimodal Preprocessing                      │  │
│  │ PIL Image → CLIPImageProcessor → pixel_values         │  │
│  │ Tokenized Prompt → model.generate() → Decoded Answer  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Hugging Face Inference Endpoint Configuration

When creating the endpoint in the Hugging Face Console:

| Setting | Value | Description |
|---|---|---|
| **Model Repository** | `MBZUAI/geochat-7B` | HF mounts this repo into `/repository` |
| **Container Type** | `Custom Container` | Uses this Dockerfile image |
| **Container Port** | `8000` | Port exposed by Uvicorn |
| **Health Route** | `/health` | HF endpoint health probe |
| **Hardware** | `1x NVIDIA T4` (or `1x L4` / `1x A10G`) | Minimum 16GB VRAM for 8-bit |
| **Min Replicas** | `0` | Scale to zero when idle for cost savings |
| **Max Replicas** | `1` | Single worker per instance |
| **Task** | `Custom` | Managed through REST API |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEOCHAT_MODEL_PATH` | `/repository` | Path to mounted model checkpoint |
| `GEOCHAT_DEVICE` | `cuda` | Target compute device |
| `GEOCHAT_LOAD_8BIT` | `true` | Enables bitsandbytes 8-bit quantization (~7.8GB VRAM) |
| `MAX_NEW_TOKENS` | `256` | Maximum generation token length |
| `PORT` | `8000` | Microservice listening port |
| `GEOCHAT_REPO_PATH` | `/app/GeoChat` | Cloned official GeoChat repository |

---

## Verified Dependency Stack

| Component | Pinned Version | Rationale |
|---|---|---|
| **Base Image** | `nvidia/cuda:11.8.0-runtime-ubuntu22.04` | CUDA 11.8 runtime for Tesla T4 |
| **Python** | `3.10` | Verified GeoChat environment |
| **PyTorch** | `torch==2.1.2+cu118` | CUDA 11.8 wheel |
| **Transformers** | `transformers==4.31.0` | **PINNED** — required for `GeoChatLlamaForCausalLM` |
| **Tokenizers** | `tokenizers==0.13.3` | Matches Transformers 4.31.0 |
| **BitsAndBytes** | `bitsandbytes==0.41.3` | 8-bit quantization on Turing/Ampere GPUs |
| **GeoChat Commit** | `4850920e005a849bd224d0ce35aa9db031fa5155` | Verified working commit |
| **FastAPI** | `fastapi==0.104.1`, `uvicorn==0.24.0` | Production ASGI service |

---

## Docker Build & Local Verification

### 1. Build Container Image
```bash
docker build -t satquery-geochat:latest services/geochat/
```

> **Note on Model Weights**: The 13.5 GB model weights are **NEVER baked into the Docker image**. The image contains only runtime libraries and code (~4.2 GB). Weights are provided via `/repository` mount on Hugging Face or downloaded at runtime if run standalone.

### 2. Run Container with Local Model Mount
```bash
docker run --gpus all -p 8000:8000 \
  -v /path/to/local/MBZUAI/geochat-7B:/repository:ro \
  -e GEOCHAT_MODEL_PATH=/repository \
  satquery-geochat:latest
```

---

## API Specification

### 1. Health Probe (`GET /health`)

**Response (HTTP 200 OK when ready)**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_class": "GeoChatLlamaForCausalLM",
  "model_name": "/repository",
  "device": "cuda:0",
  "parameter_count": 7063027712
}
```

**Response (HTTP 503 Service Unavailable during startup)**:
```json
{
  "status": "not_ready",
  "model_loaded": false,
  "model_class": "",
  "model_name": "/repository",
  "device": "cuda",
  "parameter_count": 0
}
```

### 2. Multimodal VQA (`POST /vqa`)

**Request**: `multipart/form-data`
- `image`: Satellite imagery file (PNG / JPEG / GeoTIFF)
- `question`: Natural language prompt string

**Response (HTTP 200 OK)**:
```json
{
  "task": "vqa",
  "model": "GeoChat-7B",
  "answer": "The image shows an industrial port with cargo ships, container storage facilities, and transportation infrastructure.",
  "latency_ms": 2340.50,
  "image_size": [760, 651],
  "processed_size": [504, 504],
  "mode": "real"
}
```

---

## Error Handling

| Scenario | HTTP Status | Response Payload |
|---|---|---|
| Service initializing / model not ready | `503 Service Unavailable` | `{"detail": "GeoChat model is not loaded..."}` |
| Empty question | `400 Bad Request` | `{"detail": "Question must not be empty."}` |
| Empty / invalid image | `400 Bad Request` | `{"detail": "Invalid or empty image file."}` |
| Inference exception | `500 Internal Server Error` | `{"detail": "Inference failed: ..."}` |

---

## Image Preprocessing Guarantee

```
Uploaded Image
  ↓
PIL Image (RGB)
  ↓
GeoChat Image Processor (preprocess)
  ↓
504×504 pixel_values Tensor [1, 3, 504, 504]
  ↓
Prompt "USER: <image>\n{question}\nASSISTANT:"
  ↓
model.generate(input_ids, images=pixel_values, max_new_tokens=256)
  ↓
Decode Tokens (Slice after input_ids)
  ↓
Grounded Natural-Language Answer
```
