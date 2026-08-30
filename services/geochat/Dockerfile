# ============================================================
# GeoChat VQA Service — Hugging Face Inference Endpoint Container
# ============================================================
# Base: NVIDIA CUDA 11.8 + Python 3.10
# Model Mount: /repository (mounted dynamically by HF Inference Endpoints)
# Framework: Transformers 4.31.0 (pinned for GeoChatLlamaForCausalLM)
# ============================================================

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set default python and pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# Copy and install pinned Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Clone official GeoChat repository pinned to verified commit
RUN git clone https://github.com/mbzuai-oryx/GeoChat.git /app/GeoChat \
    && cd /app/GeoChat \
    && git checkout 4850920e005a849bd224d0ce35aa9db031fa5155 \
    && pip install --no-cache-dir -e .

# Copy service application
COPY app/ /app/app/

# Environment Configuration for Hugging Face Inference Endpoints
# HF Inference Endpoints automatically mount the model repository at /repository
ENV GEOCHAT_MODEL_PATH=/repository
ENV GEOCHAT_DEVICE=cuda
ENV GEOCHAT_LOAD_8BIT=true
ENV GEOCHAT_REPO_PATH=/app/GeoChat
ENV MAX_NEW_TOKENS=256
ENV PORT=8000
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

RUN mkdir -p /app/hf_cache /repository

EXPOSE 8000

# Container health probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start Uvicorn with single worker to guarantee single GPU model resident in VRAM
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
