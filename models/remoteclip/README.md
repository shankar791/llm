# RemoteCLIP Adapter

RemoteCLIP is the first vision-language foundation model dedicated to remote sensing imagery.

## Extracted Capabilities
- **Zero-Shot Scene Classification**: Classification using text prompts without domain-specific training.
- **Cross-Modal Retrieval**: Text-to-image and image-to-text semantic search.
- **Visual-Semantic Embeddings**: Dense multimodal embeddings for satellite scene understanding.

## What is Excluded
- Downstream benchmark evaluation code.
- General dataset pre-processing scripts.

## Setup & Weights
1. Install dependencies:
   ```bash
   pip install open_clip_torch
   ```
2. Download pretrained weights:
   - Hugging Face model repository: `chendelong/RemoteCLIP`
   - Checkpoints: `RemoteCLIP-ViT-B-32`, `RemoteCLIP-ViT-L-14`, `RemoteCLIP-RN50`
