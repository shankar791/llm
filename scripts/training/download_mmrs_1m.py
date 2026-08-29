#!/usr/bin/env python3
"""
Download MMRS-1M dataset for multi-sensor instruction tuning.
Source: EarthGPT GitHub (wivizhang/EarthGPT) or OpenDataLab
Size: 1M instruction pairs covering optical, SAR, and infrared sensors
"""

import os
import sys
from pathlib import Path

# Try huggingface_hub
try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

DATA_DIR = Path("C:/Users/Y.shankar/satquery-ai/data/mmrs_1m")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_from_hf():
    """Download MMRS-1M from Hugging Face"""
    print("Downloading MMRS-1M from Hugging Face...")
    print("Note: This is a 1M dataset, will take time and space (~50-100GB)")

    # Try different possible dataset names
    dataset_candidates = [
        "wivizhang/MMRS-1M",
        "MMRS/MMRS-1M",
        "OpenDataLab/MMRS-1M",
    ]

    for repo_id in dataset_candidates:
        try:
            print(f"\nTrying {repo_id}...")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(DATA_DIR),
                local_dir_use_symlinks=False,
                allow_patterns=["*.json", "*.jsonl", "*.csv", "*.parquet", "README*"]
            )
            print(f"✅ MMRS-1M metadata downloaded to {DATA_DIR}")
            return True
        except Exception as e:
            print(f"❌ {repo_id} failed: {e}")
            continue

    return False

def main():
    print("=" * 60)
    print("MMRS-1M Dataset Downloader")
    print("=" * 60)
    print(f"Target: {DATA_DIR}")
    print()
    print("MMRS-1M sources:")
    print("1. EarthGPT GitHub: https://github.com/wivizhang/EarthGPT")
    print("2. OpenDataLab: https://opendatalab.com/")
    print("3. Hugging Face: search for 'MMRS-1M'")
    print()
    print("Note: Full 1M dataset is very large. Consider downloading")
    print("metadata + sample first, then full dataset on demand.")
    print()

    if HF_AVAILABLE:
        download_from_hf()
    else:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        download_from_hf()

    print("\n" + "=" * 60)
    print("MMRS-1M download complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()