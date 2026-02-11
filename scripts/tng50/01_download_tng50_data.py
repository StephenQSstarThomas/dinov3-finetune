#!/usr/bin/env python3
"""
Download TNG50 data for Hubble morphology classification.

This script downloads:
1. JWST-CEERS mock galaxy images (FITS datacubes)
2. Galaxy morphology kinematic data (HDF5)
3. Stellar circularities data (HDF5)
4. Star formation rates data (HDF5)

Prerequisites:
- Register at https://www.tng-project.org/users/register/
- Get your API key from your profile page

Usage:
    python scripts/tng50/01_download_tng50_data.py --api_key YOUR_API_KEY

    # Or set environment variable:
    export TNG_API_KEY=YOUR_API_KEY
    python scripts/tng50/01_download_tng50_data.py
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib


# TNG50-1 data URLs
BASE_URL = "https://www.tng-project.org/api/TNG50-1"

# Supplementary data catalog URLs
SUPPLEMENTARY_DATA = {
    "morphology_kinematic": {
        "url": "https://www.tng-project.org/api/TNG50-1/files/morphology_kinematic.hdf5",
        "filename": "morphology_kinematic.hdf5",
        "description": "Galaxy morphologies (kinematic) and bar properties",
    },
    "stellar_circs": {
        "url": "https://www.tng-project.org/api/TNG50-1/files/stellar_circs.hdf5",
        "filename": "stellar_circs.hdf5",
        "description": "Stellar circularities, angular momenta, and axis ratios",
    },
    "star_formation_rates": {
        "url": "https://www.tng-project.org/api/TNG50-1/files/star_formation_rates.hdf5",
        "filename": "star_formation_rates.hdf5",
        "description": "Star formation rates",
    },
}

# JWST-CEERS mock images: redshift to TNG50 snapshot mapping
# Files are named by redshift (e.g., skirt_images_jwst_f200w_z3.tar.gz)
# TNG50 official snapshot numbers: https://www.tng-project.org/data/docs/specifications/#sec1a
JWST_CEERS_REDSHIFTS = {
    3: {"snapshot": 25, "description": "z=3"},
    4: {"snapshot": 21, "description": "z=4"},
    5: {"snapshot": 17, "description": "z=5"},
    6: {"snapshot": 13, "description": "z=6"},
}


def get_api_key(args):
    """Get API key from args or environment variable."""
    if args.api_key:
        return args.api_key

    api_key = os.environ.get("TNG_API_KEY")
    if api_key:
        return api_key

    print("ERROR: TNG API key not provided.")
    print("Please either:")
    print("  1. Pass --api_key YOUR_API_KEY")
    print("  2. Set environment variable: export TNG_API_KEY=YOUR_API_KEY")
    print("\nRegister at: https://www.tng-project.org/users/register/")
    sys.exit(1)


def download_file(url, output_path, api_key, description="", chunk_size=8192):
    """Download a file with progress bar."""
    headers = {"api-key": api_key}

    # Make request
    response = requests.get(url, headers=headers, stream=True)

    if response.status_code == 401:
        print(f"ERROR: Authentication failed. Check your API key.")
        return False
    elif response.status_code == 404:
        print(f"ERROR: File not found: {url}")
        return False
    elif response.status_code != 200:
        print(f"ERROR: HTTP {response.status_code} for {url}")
        return False

    # Get file size
    total_size = int(response.headers.get('content-length', 0))

    # Download with progress bar
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    desc = description if description else output_path.name

    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    return True


def download_supplementary_data(output_dir, api_key, skip_existing=True):
    """Download supplementary HDF5 data files."""
    print("\n" + "="*80)
    print("Downloading Supplementary Data (HDF5)")
    print("="*80 + "\n")

    output_dir = Path(output_dir)

    for name, info in SUPPLEMENTARY_DATA.items():
        output_path = output_dir / info["filename"]

        if skip_existing and output_path.exists():
            print(f"[SKIP] {info['filename']} already exists")
            continue

        print(f"\nDownloading: {info['description']}")
        print(f"  URL: {info['url']}")
        print(f"  Output: {output_path}")

        success = download_file(
            info["url"],
            output_path,
            api_key,
            description=info["filename"]
        )

        if success:
            print(f"  [OK] Downloaded {info['filename']}")
        else:
            print(f"  [FAIL] Failed to download {info['filename']}")


def get_jwst_ceers_file_list(snapshot, api_key):
    """Get list of JWST-CEERS files for a snapshot."""
    # The JWST-CEERS mock images are typically organized by snapshot
    # This function queries the API to get the file list

    url = f"{BASE_URL}/snapshots/{snapshot}/jwst_ceers/"
    headers = {"api-key": api_key}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Warning: Could not get file list for snapshot {snapshot}")
            return None
    except Exception as e:
        print(f"Warning: Error getting file list: {e}")
        return None


def download_jwst_ceers_images(output_dir, api_key, redshifts=None, skip_existing=True):
    """Download JWST-CEERS mock galaxy images."""
    print("\n" + "="*80)
    print("Downloading JWST-CEERS Mock Galaxy Images")
    print("="*80 + "\n")

    output_dir = Path(output_dir)

    if redshifts is None:
        redshifts = list(JWST_CEERS_REDSHIFTS.keys())

    for z in redshifts:
        if z not in JWST_CEERS_REDSHIFTS:
            print(f"Warning: Unknown redshift z={z}, skipping")
            continue

        info = JWST_CEERS_REDSHIFTS[z]
        snap = info["snapshot"]
        z_dir = output_dir / f"z{z}"
        z_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nRedshift z={z} (TNG50 snapshot {snap}):")
        print(f"  Output directory: {z_dir}")

        print(f"  Note: JWST-CEERS data may need to be downloaded manually from TNG website")
        print(f"  Visit: https://www.tng-project.org/data/docs/specifications/#sec5k")
        print(f"  Files are named: skirt_images_jwst_<filter>_z{z}.tar.gz")

        # Create a placeholder file with download instructions
        readme_path = z_dir / "README.txt"
        with open(readme_path, 'w') as f:
            f.write(f"JWST-CEERS Mock Images - z={z} (TNG50 snapshot {snap})\n")
            f.write("="*60 + "\n\n")
            f.write("Download Instructions:\n")
            f.write("1. Visit: https://www.tng-project.org/data/docs/specifications/#sec5k\n")
            f.write("2. Navigate to JWST-CEERS Mock Galaxy Imaging section\n")
            f.write(f"3. Download files for z={z} (e.g., skirt_images_jwst_f200w_z{z}.tar.gz)\n")
            f.write(f"4. Extract FITS files to this directory: {z_dir}\n")
            f.write("\nExpected file format: subhalo_XXXXXX_config_YY.fits\n")
            f.write("  - XXXXXX: SubhaloID\n")
            f.write("  - YY: Viewing configuration (00-19)\n")


def verify_downloads(output_dir):
    """Verify downloaded files."""
    print("\n" + "="*80)
    print("Verifying Downloads")
    print("="*80 + "\n")

    output_dir = Path(output_dir)

    # Check supplementary data
    print("Supplementary Data:")
    for name, info in SUPPLEMENTARY_DATA.items():
        filepath = output_dir / info["filename"]
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  [OK] {info['filename']} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {info['filename']}")

    # Check JWST-CEERS directories
    print("\nJWST-CEERS Directories:")
    jwst_dir = output_dir / "jwst_ceers"
    for z, info in JWST_CEERS_REDSHIFTS.items():
        z_dir = jwst_dir / f"z{z}"
        if z_dir.exists():
            fits_files = list(z_dir.glob("*.fits"))
            print(f"  [OK] z{z} (snap {info['snapshot']}): {len(fits_files)} FITS files")
        else:
            print(f"  [MISSING] z{z}")


def main(args):
    api_key = get_api_key(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("TNG50 Data Download Script")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Skip existing: {args.skip_existing}")

    # Download supplementary data
    if not args.skip_supplementary:
        download_supplementary_data(output_dir, api_key, args.skip_existing)

    # Download JWST-CEERS images
    if not args.skip_images:
        jwst_dir = output_dir / "jwst_ceers"
        redshifts = args.redshifts if args.redshifts else None
        download_jwst_ceers_images(jwst_dir, api_key, redshifts, args.skip_existing)

    # Verify downloads
    verify_downloads(output_dir)

    print("\n" + "="*80)
    print("Download Complete!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Verify all files are downloaded correctly")
    print(f"2. If JWST-CEERS images are missing, download manually from TNG website")
    print(f"3. Run: python scripts/tng50/02_extract_physical_params.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download TNG50 data for Hubble morphology classification"
    )

    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="TNG API key (or set TNG_API_KEY environment variable)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/tng50/raw",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--redshifts",
        type=int,
        nargs="+",
        default=None,
        help="Specific redshifts to download (3, 4, 5, 6). Default: all"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip files that already exist"
    )
    parser.add_argument(
        "--skip_supplementary",
        action="store_true",
        default=False,
        help="Skip downloading supplementary HDF5 files"
    )
    parser.add_argument(
        "--skip_images",
        action="store_true",
        default=False,
        help="Skip downloading JWST-CEERS images"
    )

    args = parser.parse_args()
    main(args)
