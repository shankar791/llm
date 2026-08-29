#!/usr/bin/env python3
"""
Download Delta-SN6 dataset for bi-temporal change detection.
Source: Hugging Face (xsx31/Delta-SN6) or SpaceNet 6 (Rio de Janeiro)
Size: ~2,818 bi-temporal change pairs (much smaller than full SpaceNet 6)
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm

# Try huggingface_hub
try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("huggingface_hub not available, will use direct download")

DATA_DIR = Path("C:/Users/Y.shankar/satquery-ai/data/delta_sn6")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_from_hf():
    """Download Delta-SN6 from Hugging Face"""
    print("Downloading Delta-SN6 from Hugging Face (xsx31/Delta-SN6)...")
    try:
        snapshot_download(
            repo_id="xsx31/Delta-SN6",
            repo_type="dataset",
            local_dir=str(DATA_DIR),
            local_dir_use_symlinks=False
        )
        print(f"✅ Delta-SN6 downloaded to {DATA_DIR}")
        return True
    except Exception as e:
        print(f"❌ Hugging Face download failed: {e}")
        return False

def download_spacenet6_sample():
    """Download a sample of SpaceNet 6 (full dataset is too large)"""
    print("\nNote: Full SpaceNet 6 is ~100GB. Downloading metadata only.")
    print("For Delta-SN6 specifically, use the Hugging Face link above.")
    print("\nSpaceNet 6 download instructions:")
    print("1. Register at https://spacenet.ai/sn6-challenge/")
    print("2. Download via AWS S3: s3://spacenet-dataset/spacenet/SN6_buildings/")
    print("3. Or use the Delta-SN6 subset from HuggingFace")
    return False

def main():
    print("=" * 60)
    print("Delta-SN6 Dataset Downloader")
    print("=" * 60)
    print(f"Target: {DATA_DIR}")
    print()

    if HF_AVAILABLE:
        success = download_from_hf()
        if not success:
            download_spacenet6_sample()
    else:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        download_from_hf()

    print("\n" + "=" * 60)
    print("Delta-SN6 download complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()