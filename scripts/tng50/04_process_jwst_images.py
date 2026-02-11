#!/usr/bin/env python3
"""
Process TNG50 JWST-CEERS mock FITS images to RGB PNG format.

Data structure (per snapshot directory, e.g. snap_033/ for z=3 data):
  - RELEASE_TNG50_CEERS_F200W_z3_v1.1/  -> F200W single-filter FITS (1382x1382), HDU[1]=SCI
  - RELEASE_TNG50_CEERS_F356W_z3_v1.1/  -> F356W single-filter FITS (1382x1382), HDU[1]=SCI
  - NIRCam_LW_z3/                        -> 15-channel datacube (339x339), F444W=channel 10

Note: Directory naming uses snap_0XX convention but files are organized by redshift (z3, z4, etc.)
      Redshift to TNG50 snapshot mapping: z3->snap25, z4->snap21, z5->snap17, z6->snap13

Filenames: {SubhaloID}_i{inclination}_a{azimuth}.fits

This script:
1. Matches files across the three filter directories by filename
2. Extracts F200W, F356W, F444W image data
3. Resizes to common resolution
4. Applies asinh stretch and saves as RGB PNG

Usage:
    python scripts/tng50/04_process_jwst_images.py
"""

import argparse
from pathlib import Path
import re
import numpy as np
from PIL import Image
from astropy.io import fits
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# Filter directories and how to extract data from each
FILTER_DIRS = {
    'F200W': {
        'patterns': ['RELEASE_TNG50_CEERS_F200W*', 'F200W*'],
        'hdu_index': 1,       # SCI extension
        'channel': None,      # Not a datacube
    },
    'F356W': {
        'patterns': ['RELEASE_TNG50_CEERS_F356W*', 'F356W*'],
        'hdu_index': 1,
        'channel': None,
    },
    'F444W': {
        'patterns': ['NIRCam_LW*', 'nircam_lw*'],
        'hdu_index': 0,       # Primary HDU (datacube)
        'channel': 10,        # F444W is EXT11 -> index 10
    },
}


def find_filter_dir(snap_dir, filter_name):
    """Find the subdirectory for a given filter within a snapshot directory."""
    snap_dir = Path(snap_dir)
    config = FILTER_DIRS[filter_name]

    for pattern in config['patterns']:
        matches = list(snap_dir.glob(pattern))
        if matches:
            # Return the first directory match
            for m in matches:
                if m.is_dir():
                    return m

    return None


def asinh_stretch(data, scale=0.1, clip_percentile=99.5):
    """Apply asinh stretch to image data, returning values in [0, 1]."""
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if clip_percentile < 100:
        pos = data[data > 0]
        vmax = np.percentile(pos, clip_percentile) if len(pos) > 0 else 1.0
        data = np.clip(data, 0, vmax)

    stretched = np.arcsinh(data / scale)
    vmin, vmax = stretched.min(), stretched.max()
    if vmax > vmin:
        return (stretched - vmin) / (vmax - vmin)
    return np.zeros_like(stretched)


def read_filter_data(fits_path, filter_name):
    """Read image data from a FITS file for a given filter."""
    config = FILTER_DIRS[filter_name]

    with fits.open(fits_path) as hdu:
        data = hdu[config['hdu_index']].data

        if config['channel'] is not None:
            # Datacube: extract specific channel
            data = data[config['channel']]

    return data.astype(np.float32)


def process_single_image(f200w_path, f356w_path, f444w_path, output_path,
                          target_size=None, stretch_scale=0.1):
    """
    Create RGB PNG from three filter FITS files.

    R=F444W, G=F356W, B=F200W
    """
    try:
        blue = read_filter_data(f200w_path, 'F200W')
        green = read_filter_data(f356w_path, 'F356W')
        red = read_filter_data(f444w_path, 'F444W')

        # Resize to common shape (use the largest)
        shapes = [blue.shape, green.shape, red.shape]
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)

        def resize_if_needed(arr, target_h, target_w):
            if arr.shape[0] == target_h and arr.shape[1] == target_w:
                return arr
            img = Image.fromarray(arr)
            img = img.resize((target_w, target_h), Image.LANCZOS)
            return np.array(img)

        blue = resize_if_needed(blue, max_h, max_w)
        green = resize_if_needed(green, max_h, max_w)
        red = resize_if_needed(red, max_h, max_w)

        # Apply stretch to each channel independently
        red_norm = asinh_stretch(red, scale=stretch_scale)
        green_norm = asinh_stretch(green, scale=stretch_scale)
        blue_norm = asinh_stretch(blue, scale=stretch_scale)

        # Stack into RGB
        rgb = np.stack([red_norm, green_norm, blue_norm], axis=-1)
        rgb_8bit = (rgb * 255).astype(np.uint8)

        # Optionally resize to target size
        img = Image.fromarray(rgb_8bit)
        if target_size is not None:
            img = img.resize((target_size, target_size), Image.LANCZOS)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)

        return True

    except Exception as e:
        print(f"  Error: {e} ({f200w_path.name})")
        return False


def process_snapshot(snap_dir, output_dir, stretch_scale=0.1, target_size=None):
    """Process all images in a snapshot directory."""
    snap_dir = Path(snap_dir)
    output_dir = Path(output_dir)

    # Find filter directories
    f200w_dir = find_filter_dir(snap_dir, 'F200W')
    f356w_dir = find_filter_dir(snap_dir, 'F356W')
    f444w_dir = find_filter_dir(snap_dir, 'F444W')

    found = {
        'F200W': f200w_dir,
        'F356W': f356w_dir,
        'F444W': f444w_dir,
    }
    print(f"  Filter directories:")
    for name, d in found.items():
        print(f"    {name}: {d.name if d else 'NOT FOUND'}")

    if not all(found.values()):
        missing = [n for n, d in found.items() if d is None]
        print(f"  ERROR: Missing filter directories: {missing}")
        return 0, 0

    # Get list of FITS files from F200W directory (reference)
    fits_files = sorted(f200w_dir.glob('*.fits'))
    print(f"  Found {len(fits_files)} FITS files in F200W directory")

    success_count = 0
    fail_count = 0

    for f200w_path in tqdm(fits_files, desc=f"  Processing"):
        name = f200w_path.name
        f356w_path = f356w_dir / name
        f444w_path = f444w_dir / name

        if not f356w_path.exists() or not f444w_path.exists():
            fail_count += 1
            continue

        output_name = f200w_path.stem + ".png"
        output_path = output_dir / output_name

        if process_single_image(f200w_path, f356w_path, f444w_path, output_path,
                                 target_size=target_size, stretch_scale=stretch_scale):
            success_count += 1
        else:
            fail_count += 1

    return success_count, fail_count


def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("\n" + "="*80)
    print("JWST-CEERS Image Processing")
    print("="*80)
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Stretch scale: {args.scale}")
    if args.target_size:
        print(f"Target size: {args.target_size}x{args.target_size}")

    # Find snapshot directories
    snap_dirs = sorted(input_dir.glob('snap_*'))
    print(f"\nFound {len(snap_dirs)} snapshot directories: {[d.name for d in snap_dirs]}")

    total_success = 0
    total_fail = 0

    for snap_dir in snap_dirs:
        snap_name = snap_dir.name
        # Check if this directory has actual image data (not just README)
        has_data = any(snap_dir.glob('*/'))
        if not has_data:
            print(f"\n{snap_name}: No filter subdirectories, skipping")
            continue

        print(f"\n{snap_name}:")
        snap_output = output_dir / snap_name
        success, fail = process_snapshot(snap_dir, snap_output,
                                          stretch_scale=args.scale,
                                          target_size=args.target_size)
        total_success += success
        total_fail += fail
        print(f"  Result: {success} success, {fail} failed")

    print("\n" + "="*80)
    print("Processing Complete!")
    print("="*80)
    print(f"\nTotal: {total_success} images processed, {total_fail} failed")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process TNG50 JWST-CEERS FITS images to RGB PNG"
    )

    parser.add_argument("--input_dir", type=str,
                        default="./data/tng50/raw/jwst_ceers")
    parser.add_argument("--output_dir", type=str,
                        default="./data/tng50/processed/images")
    parser.add_argument("--scale", type=float, default=0.1,
                        help="Scale factor for asinh stretch")
    parser.add_argument("--target_size", type=int, default=None,
                        help="Resize output images to this size (e.g. 224)")

    args = parser.parse_args()
    main(args)
