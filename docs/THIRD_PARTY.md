# Third-Party Dependencies, Licensing & Attribution Register

This document serves as the formal register of all third-party repositories, open-source models, pre-trained weights, and datasets evaluated or incorporated into SatQuery AI.

> **LEGAL & ETHICAL NOTICE**:
> External components must be independently verified for licensing, attribution, model-weight terms, and appropriate use. Open-source licensing does not mean the implementation can be presented as original work.
> All items with unverified terms are marked with `VERIFY`.

---

## Evaluated Candidate Repositories & Models

### 1. wgcban/ChangeFormer (and Chen-Zhiang/ChangeFormer)
- **Repository URL**: `https://github.com/wgcban/ChangeFormer`
- **Pinned Commit / Version**: `VERIFY`
- **Source Code License**: Non-commercial / Research use only
- **Model Weights License**: Non-commercial / Research use only
- **Dataset License**: LEVIR-CD (Open research), S2Looking (Open research), Delta-SN6 (`VERIFY`)
- **Intended Purpose**: Transformer-based Siamese network for Bi-Temporal Change Detection (T4_Change).
- **Target Components**: Transformer backbone (SegFormer MiT-B0/B1), difference decoder, binary thresholding logic.
- **Isolation Boundary**: Isolated under `models/changeformer/` and accessed exclusively via adapter.
- **Attribution Requirement**: Cite Bandara & Patel, *"A Transformer-Based Siamese Network for Change Detection"*, IGARSS 2022.

---

### 2. justchenhao/BIT_CD (Bitemporal Image Transformer)
- **Repository URL**: `https://github.com/justchenhao/BIT_CD`
- **Pinned Commit / Version**: `VERIFY`
- **Source Code License**: Non-commercial / Research use only
- **Model Weights License**: Non-commercial / Research use only
- **Dataset License**: LEVIR-CD, WHU-CD, CDD (Open research)
- **Intended Purpose**: ResNet + Transformer hybrid change detection alternative for T4_Change.
- **Target Components**: Siamese ResNet18 feature extractor, spatial-temporal transformer decoder.
- **Attribution Requirement**: Cite Chen et al., *"Remote Sensing Image Change Detection with Transformers"*, IEEE TGRS 2021.

---

### 3. AndreaCodegoni/Tiny_model_4_CD (TinyCD)
- **Repository URL**: `https://github.com/AndreaCodegoni/Tiny_model_4_CD`
- **Pinned Commit / Version**: `VERIFY`
- **Source Code License**: Non-commercial / Research use only
- **Model Weights License**: Non-commercial / Research use only
- **Intended Purpose**: Ultra-lightweight change detection (<320k parameters) for low-power edge CPU/NPU execution.
- **Attribution Requirement**: Cite Codegoni et al., *"TINYM4CD: A Tiny Model for Change Detection in Remote Sensing"*, 2023.

---

### 4. juggtimber/galaxeye-eo-sar-change-detection
- **Hugging Face Model ID**: `juggtimber/galaxeye-eo-sar-change-detection`
- **GitHub URL**: `https://github.com/timbersaw-jugg/galaxeye-eo-sar-change-detection`
- **Pinned Commit / Version**: `VERIFY`
- **Source Code License**: `VERIFY`
- **Model Weights License**: `VERIFY` (Public weights on Hugging Face)
- **Intended Purpose**: Multimodal Optical + SAR cross-sensor change detection (5-channel input).
- **Target Components**: EfficientNet-B0 encoder, UNet decoder.
- **Role in SatQuery AI**: Evaluated for cross-modal optical+SAR difference analysis (T5_OpticalSAR / cross-sensor change).

---

### 5. mbzuai-oryx/GeoChat
- **Repository URL**: `https://github.com/mbzuai-oryx/GeoChat`
- **Pinned Commit / Version**: `VERIFY`
- **Source Code License**: `VERIFY`
- **Model Weights License**: `VERIFY`
- **Intended Purpose**: Vision-Language model for Single-Image VQA (T1), Captioning (T2), and Grounding (T3).
- **Attribution Requirement**: Cite Kuckreja et al., *"GeoChat: Grounded Large Vision-Language Model for Remote Sensing"*, CVPR 2024.

---

### 6. wivizhang/EarthGPT
- **Repository URL**: `https://github.com/wivizhang/EarthGPT`
- **Pinned Commit / Version**: `VERIFY`
- **Source Code License**: `VERIFY`
- **Model Weights License**: `VERIFY`
- **Intended Purpose**: Optical + SAR Cross-Modal Feature Alignment & Multimodal Fusion (T5).
- **Attribution Requirement**: Cite Zhang et al., *"EarthGPT: A Universal Multi-modal Large Language Model for Multi-sensor Remote Sensing Land-cover Comprehension"*, 2024.

---

### 7. RemoteCLIP/RemoteCLIP
- **Repository URL**: `https://github.com/RemoteCLIP/RemoteCLIP`
- **Pinned Commit / Version**: `VERIFY`
- **Source Code License**: `VERIFY`
- **Model Weights License**: `VERIFY`
- **Intended Purpose**: Zero-shot remote sensing visual-text retrieval and fallback capability.
- **Attribution Requirement**: Cite Liu et al., *"RemoteCLIP: A Vision Language Foundation Model for Remote Sensing"*, IEEE TGRS 2024.

---

## Prohibited Repositories

### `ranjithkumar1437/SatQuery-AI`
- **Policy**: **STRICTLY PROHIBITED FROM USE, REFERENCE, OR REPRODUCTION.**
- **Reason**: Identical problem statement and competition context. Must not be used as a basis for any part of this implementation.
