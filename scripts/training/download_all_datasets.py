#!/usr/bin/env python3
"""
Master download script for all SatQuery AI training/validation datasets.
Downloads datasets that don't require restricted access.
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/Y.shankar/satquery-ai")
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "training"

def check_dataset(dataset_path: Path, description: str) -> bool:
    """Check if dataset is already downloaded"""
    if dataset_path.exists() and any(dataset_path.iterdir()):
        print(f"✅ {description}: Already exists at {dataset_path}")
        return True
    else:
        print(f"⬜ {description}: Not found at {dataset_path}")
        return False

def download_bigearthnet():
    """BigEarthNet.txt is already downloaded"""
    ben_path = DATA_DIR / "BigEarthNet_txt" / "BigEarthNet.txt.parquet"
    if ben_path.exists():
        size_mb = ben_path.stat().st_size / (1024 * 1024)
        print(f"✅ BigEarthNet.txt: Already downloaded ({size_mb:.1f} MB)")
        return True
    else:
        print("⬜ BigEarthNet.txt: Needs download")
        return False

def download_delta_sn6():
    """Download Delta-SN6 dataset"""
    script_path = SCRIPTS_DIR / "download_delta_sn6.py"
    if script_path.exists():
        print("\n📥 Downloading Delta-SN6...")
        try:
            subprocess.run([sys.executable, str(script_path)], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Delta-SN6 download failed: {e}")
            return False
    return False

def download_mmrs_1m():
    """Download MMRS-1M dataset"""
    script_path = SCRIPTS_DIR / "download_mmrs_1m.py"
    if script_path.exists():
        print("\n📥 Downloading MMRS-1M...")
        try:
            subprocess.run([sys.executable, str(script_path)], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ MMRS-1M download failed: {e}")
            return False
    return False

def download_rs_vl3m():
    """Download RS-VL3M dataset (RingMo-Agent)"""
    print("\n📥 RS-VL3M (RingMo-Agent)...")
    print("Source: https://github.com/Orion-AI-Lab/RingMo-Agent")
    print("Paper: arXiv:2507.20776")
    print("Status: Download instructions in dataset documentation")
    print("Recommendation: Start with metadata only, download full set on demand")
    return False

def main():
    print("=" * 70)
    print("SatQuery AI - Dataset Download Manager")
    print("=" * 70)
    print()

    # Check existing datasets
    print("📊 Dataset Status Check:")
    print("-" * 70)

    # Training/Validation datasets
    bigearthnet_ok = download_bigearthnet()
    delta_sn6_ok = check_dataset(DATA_DIR / "delta_sn6", "Delta-SN6")
    mmrs_1m_ok = check_dataset(DATA_DIR / "mmrs_1m", "MMRS-1M")
    rs_vl3m_ok = check_dataset(DATA_DIR / "rs_vl3m", "RS-VL3M")

    # ISRO evaluation (restricted)
    isro_ok = check_dataset(DATA_DIR / "isro_evaluation", "ISRO Evaluation Data")

    # Mendeley quick test data
    mendeley_ok = check_dataset(DATA_DIR / "mendeley_merged_sar_optical", "Mendeley Quick Test")

    print()
    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"✅ BigEarthNet.txt: {'Ready' if bigearthnet_ok else 'Missing'}")
    print(f"{'✅' if delta_sn6_ok else '⬜'} Delta-SN6: {'Ready' if delta_sn6_ok else 'Needs download'}")
    print(f"{'✅' if mmrs_1m_ok else '⬜'} MMRS-1M: {'Ready' if mmrs_1m_ok else 'Needs download'}")
    print(f"{'✅' if rs_vl3m_ok else '⬜'} RS-VL3M: {'Ready' if rs_vl3m_ok else 'Needs download'}")
    print(f"{'✅' if isro_ok else '⚠️ '} ISRO Evaluation: {'Ready' if isro_ok else 'Restricted - requires MoU'}")
    print(f"{'✅' if mendeley_ok else '⬜'} Mendeley Quick Test: {'Ready' if mendeley_ok else 'Missing'}")

    print()
    print("=" * 70)
    print("Download Options:")
    print("=" * 70)
    print("1. Download Delta-SN6")
    print("2. Download MMRS-1M")
    print("3. Download RS-VL3M metadata")
    print("4. Skip (use existing data)")
    print()

    if not bigearthnet_ok:
        print("⚠️  BigEarthNet.txt is missing! This is the primary training dataset.")
        print("   Download from: https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt")

    if not delta_sn6_ok or not mmrs_1m_ok:
        print("\n💡 To download missing datasets, run:")
        if not delta_sn6_ok:
            print("   python scripts/training/download_delta_sn6.py")
        if not mmrs_1m_ok:
            print("   python scripts/training/download_mmrs_1m.py")

    if not isro_ok:
        print("\n⚠️  ISRO evaluation data requires institutional access.")
        print("   See: data/isro_evaluation/README.md for access procedure")

    print()
    print("=" * 70)

if __name__ == "__main__":
    main()