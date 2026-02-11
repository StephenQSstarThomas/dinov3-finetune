#!/usr/bin/env python3
"""
Prepare dataset splits for TNG50 Hubble morphology classification.

This script:
1. Matches processed PNG images with Hubble labels
2. Splits data by SubhaloID (not by image) to prevent data leakage
3. Stratifies by hubble_type
4. Organizes images into train/val/test directories by class

Image filename format: {SubhaloID}_i{inclination}_a{azimuth}.png

Usage:
    python scripts/tng50/05_prepare_dataset_splits.py
"""

import argparse
from pathlib import Path
import shutil
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml


def parse_image_filename(filename):
    """
    Parse image filename to extract SubhaloID and viewing config.

    Format: {SubhaloID}_i{inclination}_a{azimuth}.png
    Example: 12345_i45_a90.png -> (12345, 'i45_a90')

    Returns: (subhalo_id, config_str) or (None, None)
    """
    name = Path(filename).stem

    # Match: {id}_i{inc}_a{az}
    match = re.match(r'^(\d+)_(i\d+_a\d+)$', name)
    if match:
        return int(match.group(1)), match.group(2)

    # Fallback: subhalo_XXXXXX_config_YY
    match = re.search(r'subhalo_(\d+)_config_(\d+)', name)
    if match:
        return int(match.group(1)), f"config_{match.group(2)}"

    # Fallback: any {digits}_{rest}
    match = re.match(r'^(\d+)_(.+)$', name)
    if match:
        return int(match.group(1)), match.group(2)

    return None, None


def get_snapshot_from_dir(dir_path):
    """
    Extract snapshot number from directory name like 'snap_033'.

    Note: Directory naming may not match TNG50 official snapshot numbers.
    JWST-CEERS data uses redshift-based naming (z3, z4, etc.) but may be
    stored in directories like snap_033. The actual TNG50 snapshot mapping is:
    z3->snap25, z4->snap21, z5->snap17, z6->snap13
    """
    match = re.search(r'snap_(\d+)', Path(dir_path).name)
    return int(match.group(1)) if match else None


def collect_images(image_dir):
    """Collect all processed PNG images and their metadata."""
    image_dir = Path(image_dir)
    data = []

    snap_dirs = sorted(image_dir.glob('snap_*'))
    if len(snap_dirs) == 0:
        snap_dirs = [image_dir]

    for snap_dir in snap_dirs:
        snap_num = get_snapshot_from_dir(snap_dir)

        for img_path in snap_dir.glob('*.png'):
            subhalo_id, config_str = parse_image_filename(img_path.name)
            if subhalo_id is not None:
                data.append({
                    'image_path': str(img_path),
                    'SnapNum': snap_num if snap_num else 0,
                    'SubhaloID': subhalo_id,
                    'ConfigID': config_str,
                })

    df = pd.DataFrame(data)
    print(f"Collected {len(df)} images from {len(snap_dirs)} directories")
    if len(df) > 0:
        print(f"  Unique SubhaloIDs: {df['SubhaloID'].nunique()}")
        print(f"  Viewing configs per SubhaloID: {df.groupby('SubhaloID').size().median():.0f} (median)")

    return df


def merge_with_labels(image_df, labels_df):
    """Merge image metadata with Hubble labels."""
    print("\nMerging images with labels...")

    # Ensure matching dtypes
    image_df['SnapNum'] = image_df['SnapNum'].astype(int)
    image_df['SubhaloID'] = image_df['SubhaloID'].astype(int)
    labels_df['SnapNum'] = labels_df['SnapNum'].astype(int)
    labels_df['SubhaloID'] = labels_df['SubhaloID'].astype(int)

    label_cols = ['SnapNum', 'SubhaloID', 'hubble_type', 'hubble_label']
    available_cols = [c for c in label_cols if c in labels_df.columns]

    merged = image_df.merge(
        labels_df[available_cols],
        on=['SnapNum', 'SubhaloID'],
        how='inner'
    )

    print(f"  Images with labels: {len(merged)}")
    print(f"  Images without labels: {len(image_df) - len(merged)}")

    if len(merged) > 0:
        print(f"  Label distribution:")
        for t in ['E', 'S0', 'S', 'Irr']:
            count = (merged['hubble_type'] == t).sum()
            if count > 0:
                print(f"    {t}: {count} ({100*count/len(merged):.1f}%)")

    return merged


def split_by_subhalo(df, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split data by SubhaloID to prevent data leakage across viewing angles."""
    print("\nSplitting by SubhaloID...")

    # Get unique (SnapNum, SubhaloID) with their label
    unique_subhalos = df.groupby(['SnapNum', 'SubhaloID'])['hubble_type'].first().reset_index()
    print(f"  Unique (SnapNum, SubhaloID) pairs: {len(unique_subhalos)}")

    # Try stratification by hubble_type
    type_counts = unique_subhalos['hubble_type'].value_counts()
    min_count = type_counts.min()

    if min_count >= 3:
        stratify = unique_subhalos['hubble_type']
        print(f"  Stratifying by hubble_type")
    else:
        stratify = None
        print(f"  WARNING: Some types have < 3 samples, using random split")

    test_ratio = 1 - train_ratio - val_ratio

    train_sub, temp_sub = train_test_split(
        unique_subhalos, train_size=train_ratio,
        random_state=seed, stratify=stratify
    )

    temp_stratify = temp_sub['hubble_type'] if stratify is not None else None
    val_sub, test_sub = train_test_split(
        temp_sub, train_size=val_ratio / (val_ratio + test_ratio),
        random_state=seed, stratify=temp_stratify
    )

    print(f"  Train: {len(train_sub)} subhalos")
    print(f"  Val:   {len(val_sub)} subhalos")
    print(f"  Test:  {len(test_sub)} subhalos")

    # Assign splits
    train_keys = set(zip(train_sub['SnapNum'], train_sub['SubhaloID']))
    val_keys = set(zip(val_sub['SnapNum'], val_sub['SubhaloID']))
    test_keys = set(zip(test_sub['SnapNum'], test_sub['SubhaloID']))

    def assign_split(row):
        key = (row['SnapNum'], row['SubhaloID'])
        if key in train_keys:
            return 'train'
        elif key in val_keys:
            return 'val'
        elif key in test_keys:
            return 'test'
        return None

    df['split'] = df.apply(assign_split, axis=1)
    return df


def organize_splits(df, output_dir):
    """Copy images into split/class directory structure."""
    output_dir = Path(output_dir)

    print("\nOrganizing images into splits...")

    for split in ['train', 'val', 'test']:
        for hubble_type in ['E', 'S0', 'S', 'Irr']:
            (output_dir / split / hubble_type).mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        print(f"\n  {split.upper()}: {len(split_df)} images")

        for hubble_type in ['E', 'S0', 'S', 'Irr']:
            type_df = split_df[split_df['hubble_type'] == hubble_type]
            if len(type_df) == 0:
                continue
            print(f"    {hubble_type}: {len(type_df)}")

            for _, row in tqdm(type_df.iterrows(), total=len(type_df),
                               desc=f"    {hubble_type}", leave=False):
                src = Path(row['image_path'])
                dst = output_dir / split / hubble_type / src.name
                if src.exists():
                    shutil.copy2(src, dst)


def print_split_statistics(df):
    """Print detailed split statistics."""
    print("\n" + "="*80)
    print("Split Statistics")
    print("="*80)

    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        print(f"\n{split.upper()} ({len(split_df)} images, "
              f"{split_df.groupby(['SnapNum', 'SubhaloID']).ngroups} subhalos):")

        for hubble_type in ['E', 'S0', 'S', 'Irr']:
            count = (split_df['hubble_type'] == hubble_type).sum()
            pct = 100 * count / len(split_df) if len(split_df) > 0 else 0
            print(f"  {hubble_type}: {count:5d} ({pct:5.1f}%)")


def save_split_info(df, output_dir):
    """Save split information to files."""
    output_dir = Path(output_dir)

    df.to_csv(output_dir / 'split_info.csv', index=False)

    summary = {
        'total_images': len(df),
        'splits': {},
    }

    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        summary['splits'][split] = {
            'total': int(len(split_df)),
            'by_type': {k: int(v) for k, v in split_df['hubble_type'].value_counts().items()},
            'unique_subhalos': int(split_df.groupby(['SnapNum', 'SubhaloID']).ngroups),
        }

    with open(output_dir / 'split_summary.yaml', 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)

    print(f"\nSplit info saved to: {output_dir / 'split_info.csv'}")
    print(f"Split summary saved to: {output_dir / 'split_summary.yaml'}")


def main(args):
    image_dir = Path(args.image_dir)
    labels_file = Path(args.labels_file)
    output_dir = Path(args.output_dir)

    print("\n" + "="*80)
    print("TNG50 Dataset Split Preparation")
    print("="*80)
    print(f"\nImage directory: {image_dir}")
    print(f"Labels file: {labels_file}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratios: {args.train_ratio}/{args.val_ratio}/{1-args.train_ratio-args.val_ratio:.2f}")

    if not labels_file.exists():
        print(f"\nERROR: Labels file not found: {labels_file}")
        return

    labels_df = pd.read_csv(labels_file)
    print(f"\nLoaded {len(labels_df)} labels")

    image_df = collect_images(image_dir)
    if len(image_df) == 0:
        print("\nERROR: No images found.")
        return

    df = merge_with_labels(image_df, labels_df)
    if len(df) == 0:
        print("\nERROR: No images matched with labels.")
        return

    df = split_by_subhalo(df, args.train_ratio, args.val_ratio, args.seed)
    print_split_statistics(df)
    organize_splits(df, output_dir)
    save_split_info(df, output_dir)

    print("\n" + "="*80)
    print("Dataset Preparation Complete!")
    print("="*80)
    print(f"\nDataset ready at: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare dataset splits for TNG50 Hubble classification"
    )

    parser.add_argument("--image_dir", type=str,
                        default="./data/tng50/processed/images")
    parser.add_argument("--labels_file", type=str,
                        default="./data/tng50/processed/labels/tng50_hubble_labels.csv")
    parser.add_argument("--output_dir", type=str,
                        default="./data/tng50/splits")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
