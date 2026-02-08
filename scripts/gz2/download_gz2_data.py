#!/usr/bin/env python3
"""
Download Galaxy Zoo 2 dataset from Kaggle
Requires: pip install kagglehub
"""

import kagglehub

print("Downloading Galaxy Zoo 2 dataset from Kaggle...")
print("This may take a while (approximately 3GB)...")

path = kagglehub.dataset_download("jaimetrickz/galaxy-zoo-2-images")

print(f"\nDataset downloaded to: {path}")
print("\nContents:")
print("  - images_gz2/images/  : Galaxy images (JPG)")
print("  - gz2_filename_mapping.csv : Image ID mapping")
