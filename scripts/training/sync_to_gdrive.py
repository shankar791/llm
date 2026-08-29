#!/usr/bin/env python3
"""
Sync SatQuery AI datasets to Google Drive using Google Drive Desktop app.
Prerequisites: Install Google Drive Desktop (https://www.google.com/drive/download/)
"""

import os
import shutil
from pathlib import Path
import sys

# Configuration
SOURCE_DIR = Path("C:/Users/Y.shankar/satquery-ai/data")
# Update this path to match your Google Drive Desktop installation
# Default locations:
#   Windows: G:\My Drive\ or C:\Users\<user>\Google Drive\
GDRIVE_PATHS = [
    Path("G:/My Drive/SatQuery_AI_Datasets"),
    Path("C:/Users/Y.shankar/Google Drive/SatQuery_AI_Datasets"),
    Path(os.path.expanduser("~/Google Drive/SatQuery_AI_Datasets")),
]

def find_gdrive_path():
    """Find the Google Drive folder on this system"""
    for path in GDRIVE_PATHS:
        if path.parent.exists():
            print(f"✅ Found Google Drive at: {path.parent}")
            path.mkdir(parents=True, exist_ok=True)
            return path
    return None

def sync_datasets():
    """Sync datasets to Google Drive"""
    gdrive_path = find_gdrive_path()
    if not gdrive_path:
        print("❌ Google Drive folder not found!")
        print("\nPlease install Google Drive Desktop:")
        print("https://www.google.com/drive/download/")
        print("\nOr update GDRIVE_PATHS in this script with your Drive location")
        return False

    print(f"📂 Source: {SOURCE_DIR}")
    print(f"📂 Destination: {gdrive_path}")
    print()

    # Datasets to sync
    datasets = {
        "BigEarthNet_txt": "BigEarthNet.txt (464k pairs)",
        "delta_sn6": "Delta-SN6 (2,818 changes)",
        "mmrs_1m": "MMRS-1M (1M pairs)",
        "rs_vl3m": "RS-VL3M (3M pairs)",
        "mendeley_merged_sar_optical": "Mendeley quick test",
    }

    for folder, description in datasets.items():
        source = SOURCE_DIR / folder
        dest = gdrive_path / folder

        if not source.exists():
            print(f"⬜ {description}: Not found locally")
            continue

        print(f"📤 Syncing {description}...")
        try:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, dest, dirs_exist_ok=True)
            size_mb = sum(f.stat().st_size for f in dest.rglob('*') if f.is_file()) / (1024*1024)
            print(f"   ✅ Synced ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    # Copy ISRO evaluation structure
    isro_source = SOURCE_DIR / "isro_evaluation"
    isro_dest = gdrive_path / "isro_evaluation"
    if isro_source.exists():
        print(f"📤 Syncing ISRO evaluation structure...")
        try:
            isro_dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(isro_source, isro_dest, dirs_exist_ok=True)
            print(f"   ✅ Synced")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    # Copy documentation
    docs_to_copy = [
        "DATASETS.md",
        "GOOGLE_DRIVE_SETUP.md",
    ]
    print(f"\n📄 Copying documentation...")
    for doc in docs_to_copy:
        source = SOURCE_DIR.parent / doc
        if source.exists():
            shutil.copy2(source, gdrive_path / doc)
            print(f"   ✅ {doc}")

    print("\n" + "=" * 60)
    print("✅ Sync complete!")
    print(f"📂 Google Drive folder: {gdrive_path}")
    print("=" * 60)
    print("\nFiles will auto-sync to Google Drive in the background.")
    print("You can monitor sync status in Google Drive Desktop app.")

    return True

if __name__ == "__main__":
    print("=" * 60)
    print("SatQuery AI - Google Drive Sync")
    print("=" * 60)
    print()
    sync_datasets()