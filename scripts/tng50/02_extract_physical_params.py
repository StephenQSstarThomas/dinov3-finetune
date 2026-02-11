#!/usr/bin/env python3
"""
Extract physical parameters from TNG50 for all galaxies with images.

This script:
1. Scans processed images to find all (SnapNum, SubhaloID) pairs
2. Fetches physical parameters from the TNG API for EVERY image SubhaloID
3. Loads supplementary HDF5 data where available (morphology, circularity, SFR)
4. Merges all sources (API as base, supplementary as enrichment)
5. Computes derived quantities (sSFR, V/sigma, etc.)

Output: CSV file with all physical parameters keyed by (SnapNum, SubhaloID)

Usage:
    python scripts/tng50/02_extract_physical_params.py
    python scripts/tng50/02_extract_physical_params.py --api_key YOUR_KEY
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# TNG50-1 cosmology
H_HUBBLE = 0.6774  # h parameter for unit conversion

# TNG50 snapshot to redshift mapping
# Directory names now match TNG50 snapshot numbers directly
SNAPSHOT_TO_REDSHIFT = {
    25: 3.0,  # z=3
    21: 4.0,  # z=4
    17: 5.0,  # z=5
    13: 6.0,  # z=6
}


# ============================================================
# Phase 1: Identify image SubhaloIDs
# ============================================================

def get_image_subhalo_ids(image_dir):
    """Scan processed images to get all unique (SnapNum, SubhaloID) pairs."""
    image_dir = Path(image_dir)
    pairs = set()

    snap_dirs = sorted(image_dir.glob('snap_*'))
    if not snap_dirs:
        snap_dirs = [image_dir]

    for snap_dir in snap_dirs:
        match = re.search(r'snap_(\d+)', snap_dir.name)
        snap_num = int(match.group(1)) if match else 0

        for img_path in snap_dir.glob('*.png'):
            m = re.match(r'^(\d+)_', img_path.name)
            if m:
                pairs.add((snap_num, int(m.group(1))))

    return sorted(pairs)


# ============================================================
# Phase 2: Fetch from TNG API
# ============================================================

# Fields to extract from API response
API_FIELDS = [
    'mass_stars', 'mass_gas', 'mass', 'mass_dm',
    'sfr', 'sfrinhalfrad', 'sfrinrad',
    'vmax', 'veldisp', 'vmaxrad',
    'halfmassrad_stars', 'halfmassrad',
    'spin_x', 'spin_y', 'spin_z',
    'len_stars', 'len_gas', 'len',
    'starmetallicity', 'starmetallicityhalfrad',
    'gasmetallicity', 'gasmetallicitysfr',
    'stellarphotometrics_u', 'stellarphotometrics_b',
    'stellarphotometrics_v', 'stellarphotometrics_k',
    'stellarphotometrics_g', 'stellarphotometrics_r',
    'stellarphotometrics_i', 'stellarphotometrics_z',
    'primary_flag', 'grnr',
]


def fetch_subhalo_from_api(snap, subhalo_id, api_key):
    """Fetch a single subhalo's properties from TNG API."""
    url = f"https://www.tng-project.org/api/TNG50-1/snapshots/{snap}/subhalos/{subhalo_id}/"
    headers = {"api-key": api_key}

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                # Rate limited - wait and retry
                time.sleep(5 * (attempt + 1))
                continue
            else:
                return None
        except requests.exceptions.RequestException:
            time.sleep(2 * (attempt + 1))
    return None


def fetch_all_from_api(pairs, api_key, cache_file, delay=0.05):
    """Fetch all subhalo properties from TNG API with caching."""
    print(f"\nFetching physical parameters from TNG API...")

    # Load cache
    cache = {}
    cache_file = Path(cache_file)
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            cache = json.load(f)

    to_fetch = [(s, i) for s, i in pairs if f"{s}_{i}" not in cache]
    print(f"  Total pairs: {len(pairs)}, Cached: {len(pairs) - len(to_fetch)}, To fetch: {len(to_fetch)}")

    if to_fetch:
        for snap, sid in tqdm(to_fetch, desc="  Querying TNG API"):
            result = fetch_subhalo_from_api(snap, sid, api_key)
            if result:
                cache[f"{snap}_{sid}"] = {k: result.get(k) for k in API_FIELDS}
            else:
                print(f"\n  WARNING: Failed to fetch snap={snap}, SubhaloID={sid}")

            time.sleep(delay)

            # Save cache every 100 fetches
            if len(cache) % 100 == 0:
                with open(cache_file, 'w') as f:
                    json.dump(cache, f)

        # Final cache save
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
        print(f"  Cache saved to: {cache_file}")

    # Convert to DataFrame
    data = []
    for snap, sid in pairs:
        key = f"{snap}_{sid}"
        if key in cache:
            row = {'SnapNum': snap, 'SubhaloID': sid}
            row.update(cache[key])
            data.append(row)

    df = pd.DataFrame(data)
    print(f"  Fetched {len(df)} subhalos from API")
    return df


# ============================================================
# Phase 3: Load supplementary HDF5 data
# ============================================================

def _parse_snap_num(key):
    """Extract snapshot number from group key like 'Snapshot_99'."""
    return int(key.replace('Snapshot_', '').replace('Snapshot', '').replace('snap_', ''))


def _find_id_field(group):
    """Find the subhalo ID field name in an HDF5 group."""
    for name in ['SubfindID', 'SubhaloID']:
        if name in group:
            return name
    return None


def load_morphology_kinematic(filepath):
    """Load kinematic morphology data from morphs_kinematic_bars.hdf5.

    Handles the actual file structure where fields like ThinDisc have shape (3, n)
    and BarStrength has shape (2, n). Uses index 0 of each multi-dim field.
    """
    print(f"\nLoading morphology kinematic data from: {filepath}")

    # Mapping from HDF5 field names to output column names
    # Fields with shape (3, n): use index 0
    fraction_fields = {
        'ThinDisc': 'ThinDiscFraction',
        'ThickDisc': 'ThickDiscFraction',
        'PseudoBulge': 'PseudoBulgeFraction',
        'Bulge': 'ClassicalBulgeFraction',
        'Halo': 'StellarHaloFraction',
    }

    data = []

    with h5py.File(filepath, 'r') as f:
        print(f"  Top-level keys: {list(f.keys())[:5]}...")

        for key in f.keys():
            if not key.startswith('Snapshot'):
                continue
            snap_num = _parse_snap_num(key)
            group = f[key]
            id_field = _find_id_field(group)
            if id_field is None:
                continue

            subhalo_ids = group[id_field][:]
            n = len(subhalo_ids)
            snap_data = {'SnapNum': np.full(n, snap_num), 'SubhaloID': subhalo_ids}

            # Extract fraction fields (shape (3, n) -> take index 0)
            for hdf5_name, col_name in fraction_fields.items():
                if hdf5_name in group:
                    arr = group[hdf5_name][:]
                    if arr.ndim == 2:
                        snap_data[col_name] = arr[0]  # First decomposition method
                    else:
                        snap_data[col_name] = arr
                else:
                    snap_data[col_name] = np.zeros(n, dtype=float)

            # Barred (boolean, shape (n,))
            if 'Barred' in group:
                snap_data['Barred'] = group['Barred'][:].astype(float)

            # BarStrength (shape (2, n) -> take index 0)
            if 'BarStrength' in group:
                arr = group['BarStrength'][:]
                snap_data['BarStrength'] = arr[0] if arr.ndim == 2 else arr

            # StellarMass from morphology catalog
            if 'StellarMass' in group:
                snap_data['StellarMass_morph'] = group['StellarMass'][:]

            data.append(pd.DataFrame(snap_data))

    df = pd.concat(data, ignore_index=True) if data else pd.DataFrame()
    print(f"  Loaded {len(df)} entries across {df['SnapNum'].nunique() if len(df) > 0 else 0} snapshots")
    return df


def load_stellar_circs(filepath):
    """Load stellar circularities data from HDF5 file."""
    print(f"\nLoading stellar circularities from: {filepath}")

    circ_fields = [
        'CircAbove07Frac', 'CircAbove07Frac_allstars',
        'CircAbove07MinusBelowNeg07Frac', 'CircAbove07MinusBelowNeg07Frac_allstars',
        'CircTwiceBelow0Frac', 'CircTwiceBelow0Frac_allstars',
        'SpecificAngMom', 'SpecificAngMom_allstars',
    ]

    data = []

    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            if not key.startswith('Snapshot'):
                continue
            snap_num = _parse_snap_num(key)
            group = f[key]
            id_field = _find_id_field(group)
            if id_field is None:
                continue

            subhalo_ids = group[id_field][:]
            n = len(subhalo_ids)
            snap_data = {'SnapNum': np.full(n, snap_num), 'SubhaloID': subhalo_ids}

            for field in circ_fields:
                if field in group:
                    arr = group[field][:]
                    if arr.ndim == 1:
                        snap_data[field] = arr

            data.append(pd.DataFrame(snap_data))

    df = pd.concat(data, ignore_index=True) if data else pd.DataFrame()
    print(f"  Loaded {len(df)} entries across {df['SnapNum'].nunique() if len(df) > 0 else 0} snapshots")
    return df


def load_star_formation_rates(filepath):
    """Load star formation rates from HDF5 file."""
    print(f"\nLoading star formation rates from: {filepath}")

    sfr_fields = [
        'SFR_MsunPerYrs_in_all_10Myrs',
        'SFR_MsunPerYrs_in_all_50Myrs',
        'SFR_MsunPerYrs_in_all_100Myrs',
        'SFR_MsunPerYrs_in_all_200Myrs',
        'SFR_MsunPerYrs_in_all_1000Myrs',
        'SFR_MsunPerYrs_in_InRad_100Myrs',
    ]

    data = []

    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            if not key.startswith('Snapshot'):
                continue
            snap_num = _parse_snap_num(key)
            group = f[key]
            id_field = _find_id_field(group)
            if id_field is None:
                continue

            subhalo_ids = group[id_field][:]
            n = len(subhalo_ids)
            snap_data = {'SnapNum': np.full(n, snap_num), 'SubhaloID': subhalo_ids}

            for field in sfr_fields:
                if field in group:
                    snap_data[field] = group[field][:]

            data.append(pd.DataFrame(snap_data))

    df = pd.concat(data, ignore_index=True) if data else pd.DataFrame()
    print(f"  Loaded {len(df)} entries across {df['SnapNum'].nunique() if len(df) > 0 else 0} snapshots")
    return df


# ============================================================
# Phase 4: Merge all data sources
# ============================================================

def merge_all_data(api_df, morph_df, circ_df, sfr_df):
    """Merge all data sources with API data as base (outer join for supplementary)."""
    print("\nMerging all data sources...")
    print(f"  API data: {len(api_df)} entries (base)")

    base_df = api_df.copy()

    supplementary = [
        (morph_df, 'morphology'),
        (circ_df, 'circularity'),
        (sfr_df, 'SFR catalog'),
    ]

    for supp_df, name in supplementary:
        if supp_df is None or len(supp_df) == 0:
            print(f"  {name}: skipped (no data)")
            continue

        # Only merge columns not already in base
        new_cols = [c for c in supp_df.columns if c not in base_df.columns]
        if not new_cols:
            print(f"  {name}: skipped (no new columns)")
            continue

        merge_cols = ['SnapNum', 'SubhaloID'] + new_cols
        before = base_df[new_cols[0]].notna().sum() if new_cols[0] in base_df.columns else 0

        base_df = base_df.merge(
            supp_df[merge_cols],
            on=['SnapNum', 'SubhaloID'],
            how='left'
        )

        matched = base_df[new_cols[0]].notna().sum()
        print(f"  {name}: {len(new_cols)} new columns, {matched}/{len(base_df)} matched")

    print(f"  Final: {len(base_df)} entries, {len(base_df.columns)} columns")
    return base_df


# ============================================================
# Phase 5: Compute derived quantities
# ============================================================

def compute_derived_quantities(df):
    """Compute derived physical quantities for classification."""
    print("\nComputing derived quantities...")

    # --- V/sigma ratio (disk vs spheroid proxy) ---
    if 'vmax' in df.columns and 'veldisp' in df.columns:
        df['v_sigma'] = df['vmax'] / df['veldisp'].clip(lower=1e-10)
        valid = df['veldisp'] > 0
        print(f"  v_sigma range: [{df.loc[valid, 'v_sigma'].min():.3f}, "
              f"{df.loc[valid, 'v_sigma'].max():.3f}] (n={valid.sum()})")

    # --- Stellar mass in Msun ---
    if 'mass_stars' in df.columns:
        df['mass_stars_msun'] = df['mass_stars'] * 1e10 / H_HUBBLE
        df['log_mass_stars'] = np.log10(df['mass_stars_msun'].clip(lower=1.0))
        print(f"  log_mass_stars range: [{df['log_mass_stars'].min():.2f}, "
              f"{df['log_mass_stars'].max():.2f}]")

    # --- Specific SFR (sSFR) from API ---
    if 'sfr' in df.columns and 'mass_stars_msun' in df.columns:
        df['sSFR'] = df['sfr'] / df['mass_stars_msun'].clip(lower=1.0)
        df['log_sSFR'] = np.log10(df['sSFR'].clip(lower=1e-20))
        valid = df['mass_stars_msun'] > 1e6
        print(f"  log_sSFR range (M*>1e6): [{df.loc[valid, 'log_sSFR'].min():.2f}, "
              f"{df.loc[valid, 'log_sSFR'].max():.2f}] (n={valid.sum()})")

    # --- Circularity-based disk fraction (from supplementary data, where available) ---
    if 'CircAbove07Frac' in df.columns:
        df['f_disk_circ'] = df['CircAbove07Frac']
        valid = df['f_disk_circ'].notna()
        print(f"  f_disk_circ available for {valid.sum()}/{len(df)} entries")

    # --- Morphology-based disk fraction (from supplementary data, where available) ---
    if 'ThinDiscFraction' in df.columns and 'ThickDiscFraction' in df.columns:
        df['f_disk_morph'] = df['ThinDiscFraction'] + df['ThickDiscFraction']
        valid = df['f_disk_morph'].notna()
        print(f"  f_disk_morph available for {valid.sum()}/{len(df)} entries")

    # --- SFR from supplementary catalog (100 Myr average) ---
    sfr_col = 'SFR_MsunPerYrs_in_all_100Myrs'
    if sfr_col in df.columns:
        df['SFR_100Myr'] = df[sfr_col]
        df['log_SFR_100Myr'] = np.log10(df['SFR_100Myr'].clip(lower=1e-15))

    # --- Gas fraction ---
    if 'mass_gas' in df.columns and 'mass_stars' in df.columns:
        total = df['mass_gas'] + df['mass_stars']
        df['f_gas'] = df['mass_gas'] / total.clip(lower=1e-20)

    # --- Spin magnitude ---
    if all(c in df.columns for c in ['spin_x', 'spin_y', 'spin_z']):
        df['spin_mag'] = np.sqrt(
            df['spin_x']**2 + df['spin_y']**2 + df['spin_z']**2
        )

    print(f"  Final columns: {list(df.columns)}")
    return df


# ============================================================
# Statistics and output
# ============================================================

def print_statistics(df):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("Physical Parameters Summary")
    print("=" * 80)

    print(f"\nTotal entries: {len(df)}")
    print(f"Unique SubhaloIDs: {df['SubhaloID'].nunique()}")
    print(f"Unique snapshots: {df['SnapNum'].nunique()}")

    # Mass distribution
    if 'log_mass_stars' in df.columns:
        print(f"\nStellar mass distribution:")
        for threshold in [6, 7, 8, 9, 10, 11]:
            count = (df['log_mass_stars'] >= threshold).sum()
            print(f"  log(M*) >= {threshold}: {count} ({100*count/len(df):.1f}%)")

    # Data source coverage
    print(f"\nSupplementary data coverage:")
    if 'f_disk_circ' in df.columns:
        print(f"  Circularity data: {df['f_disk_circ'].notna().sum()}/{len(df)}")
    if 'f_disk_morph' in df.columns:
        print(f"  Morphology data:  {df['f_disk_morph'].notna().sum()}/{len(df)}")
    if 'SFR_100Myr' in df.columns:
        print(f"  SFR catalog data: {df['SFR_100Myr'].notna().sum()}/{len(df)}")

    print(f"\nKey columns:")
    for col in ['v_sigma', 'log_sSFR', 'log_mass_stars', 'f_disk_circ']:
        if col in df.columns:
            valid = df[col].notna() & np.isfinite(df[col])
            if valid.sum() > 0:
                print(f"  {col:25s}: min={df.loc[valid, col].min():10.3f}, "
                      f"max={df.loc[valid, col].max():10.3f}, "
                      f"mean={df.loc[valid, col].mean():10.3f}, "
                      f"valid={valid.sum()}")


def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir)

    print("\n" + "=" * 80)
    print("TNG50 Physical Parameters Extraction")
    print("=" * 80)
    print(f"\nInput directory (HDF5): {input_dir}")
    print(f"Image directory: {image_dir}")
    print(f"Output directory: {output_dir}")

    # --- Phase 1: Find image SubhaloIDs ---
    print("\n--- Phase 1: Identifying image SubhaloIDs ---")
    pairs = get_image_subhalo_ids(image_dir)
    print(f"  Found {len(pairs)} unique (SnapNum, SubhaloID) pairs from images")
    if len(pairs) == 0:
        print("\nERROR: No images found.")
        return

    snaps = sorted(set(s for s, _ in pairs))
    print(f"  Snapshots: {snaps}")
    for snap in snaps:
        count = sum(1 for s, _ in pairs if s == snap)
        print(f"    Snap {snap}: {count} SubhaloIDs")

    # --- Phase 2: Fetch from TNG API ---
    print("\n--- Phase 2: Fetching from TNG API ---")
    api_key = args.api_key or os.environ.get('TNG_API_KEY', 'e692deabaea371d625d975db53dad645')
    cache_file = output_dir / "api_cache.json"

    api_df = fetch_all_from_api(pairs, api_key, cache_file, delay=args.api_delay)

    if len(api_df) == 0:
        print("\nERROR: No data fetched from API.")
        return

    # --- Phase 3: Load supplementary HDF5 data ---
    print("\n--- Phase 3: Loading supplementary HDF5 data ---")

    morph_file = input_dir / "morphs_kinematic_bars.hdf5"
    if morph_file.exists():
        morph_df = load_morphology_kinematic(morph_file)
    else:
        print(f"\n  WARNING: {morph_file.name} not found, skipping morphology")
        morph_df = pd.DataFrame()

    circ_file = input_dir / "stellar_circs.hdf5"
    if circ_file.exists():
        circ_df = load_stellar_circs(circ_file)
    else:
        print(f"\n  WARNING: {circ_file.name} not found")
        circ_df = pd.DataFrame()

    sfr_file = input_dir / "star_formation_rates.hdf5"
    if sfr_file.exists():
        sfr_df = load_star_formation_rates(sfr_file)
    else:
        print(f"\n  WARNING: {sfr_file.name} not found")
        sfr_df = pd.DataFrame()

    # --- Phase 4: Merge all data ---
    print("\n--- Phase 4: Merging all data ---")
    df = merge_all_data(
        api_df,
        morph_df if len(morph_df) > 0 else None,
        circ_df if len(circ_df) > 0 else None,
        sfr_df if len(sfr_df) > 0 else None,
    )

    # --- Phase 5: Compute derived quantities ---
    print("\n--- Phase 5: Computing derived quantities ---")
    df = compute_derived_quantities(df)

    # Statistics
    print_statistics(df)

    # Save
    output_file = output_dir / "tng50_physical_params.csv"
    df.to_csv(output_file, index=False)
    print(f"\nPhysical parameters saved to: {output_file}")
    print(f"  Shape: {df.shape}")

    print("\n" + "=" * 80)
    print("Extraction Complete!")
    print("=" * 80)
    print(f"\nNext step: python scripts/tng50/03_generate_hubble_labels.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract physical parameters from TNG50 (API + HDF5)"
    )

    parser.add_argument("--input_dir", type=str,
                        default="./data/tng50/raw",
                        help="Directory containing supplementary HDF5 files")
    parser.add_argument("--image_dir", type=str,
                        default="./data/tng50/processed/images",
                        help="Directory containing processed PNG images")
    parser.add_argument("--output_dir", type=str,
                        default="./data/tng50/processed/catalog",
                        help="Output directory for CSV file")
    parser.add_argument("--api_key", type=str, default=None,
                        help="TNG API key (or set TNG_API_KEY env var)")
    parser.add_argument("--api_delay", type=float, default=0.05,
                        help="Delay between API requests in seconds")

    args = parser.parse_args()
    main(args)
