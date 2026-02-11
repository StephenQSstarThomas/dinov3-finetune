#!/usr/bin/env python3
"""
Generate Hubble morphology labels from TNG50 physical parameters.

Adapted for z~3 high-redshift galaxies where:
- All galaxies are star-forming (no quiescent population)
- All galaxies are rotation-dominated (V/σ > 1.4)
- Classification relies primarily on morphological decomposition (D/T ratio)
  and stellar circularity (CircAbove07Frac)

Classification scheme for z~3 (based on Zana+2022, Du+2024):
- E (Elliptical/Spheroid):  D/T < 0.15 OR f_circ < 0.10 (spheroid-dominated)
- S0 (Proto-S0/Compact):    0.15 ≤ D/T < 0.35 AND low sSFR relative to mass
- S (Disk/Spiral):          D/T ≥ 0.35 OR f_circ ≥ 0.25 (disk-dominated)
- Irr (Irregular/Clumpy):   D/T < 0.35 AND high sSFR AND disordered kinematics

References:
- Zana et al. 2022 (arXiv:2206.04693) - MORDOR morphological decomposition
- Du et al. 2024 (arXiv:2403.14749) - Running circularity threshold

Usage:
    python scripts/tng50/03_generate_hubble_labels.py
    python scripts/tng50/03_generate_hubble_labels.py --config configs/tng50_hubble.yaml
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


# Default thresholds optimized for z~3 TNG50 galaxies
DEFAULT_THRESHOLDS = {
    # D/T (Disk-to-Total) thresholds from morphological decomposition
    # Based on ThinDisc + ThickDisc fractions
    'dt_spheroid': 0.15,           # D/T < this -> spheroid-dominated (E)
    'dt_disk': 0.35,               # D/T >= this -> disk-dominated (S)

    # CircAbove07Frac thresholds (stellar circularity ε > 0.7)
    # More robust than D/T for high-z galaxies
    'f_circ_spheroid': 0.10,       # f_circ < this -> spheroid-dominated
    'f_circ_disk': 0.25,           # f_circ >= this -> disk-dominated

    # sSFR thresholds relative to z~3 main sequence
    # At z~3, main sequence log(sSFR) ~ -8.5 to -9.0
    'log_sSFR_low': -9.2,          # Below main sequence
    'log_sSFR_high': -8.5,         # Above main sequence (starburst)

    # V/σ thresholds (less useful at z~3 where all are rotation-dominated)
    'v_sigma_low': 1.5,            # Lower bound for disk classification
    'v_sigma_high': 2.0,           # Strong rotation

    # Mass thresholds
    'log_mass_min': 9.0,           # Minimum log(M*/Msun) for classification
    'min_star_particles': 50000,   # Minimum star particles

    # Irregularity indicators
    'circ_asymmetry_threshold': 0.15,  # |CircAbove07 - CircBelow-07| threshold
}


def load_config(config_path):
    """Load configuration from YAML file."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        thresholds = config.get('label_thresholds', {})
        merged = DEFAULT_THRESHOLDS.copy()
        merged.update(thresholds)
        return merged
    return DEFAULT_THRESHOLDS.copy()


def classify_hubble_type_z3(row, thresholds):
    """
    Classify a z~3 galaxy into Hubble-like morphological type.

    Priority order for classification:
    1. Morphological decomposition (D/T = ThinDisc + ThickDisc)
    2. Stellar circularity (CircAbove07Frac)
    3. sSFR relative to main sequence
    4. Kinematic asymmetry for Irr identification

    Returns: 'E', 'S0', 'S', 'Irr', or 'unresolved'
    """
    # Check minimum mass / particle count
    log_mass = row.get('log_mass_stars', 0)
    n_stars = row.get('len_stars', 0)

    if log_mass < thresholds['log_mass_min']:
        return 'unresolved'
    if n_stars < thresholds['min_star_particles']:
        return 'unresolved'

    # Get morphological parameters
    dt = row.get('f_disk_morph', np.nan)  # D/T from morphological decomposition
    f_circ = row.get('f_disk_circ', np.nan)  # CircAbove07Frac
    log_ssfr = row.get('log_sSFR', -9.0)
    v_sigma = row.get('v_sigma', 1.7)

    # Kinematic asymmetry: difference between prograde and retrograde orbits
    circ_asym = row.get('CircAbove07MinusBelowNeg07Frac', np.nan)

    # Thresholds
    dt_sph = thresholds['dt_spheroid']
    dt_disk = thresholds['dt_disk']
    fc_sph = thresholds['f_circ_spheroid']
    fc_disk = thresholds['f_circ_disk']
    ssfr_low = thresholds['log_sSFR_low']
    ssfr_high = thresholds['log_sSFR_high']

    # Determine disk dominance from multiple indicators
    has_dt = not np.isnan(dt)
    has_circ = not np.isnan(f_circ)

    # Primary classification based on D/T (most reliable)
    if has_dt:
        is_spheroid_dt = dt < dt_sph
        is_disk_dt = dt >= dt_disk
        is_intermediate_dt = dt_sph <= dt < dt_disk
    else:
        is_spheroid_dt = False
        is_disk_dt = False
        is_intermediate_dt = True

    # Secondary classification based on circularity
    if has_circ:
        is_spheroid_circ = f_circ < fc_sph
        is_disk_circ = f_circ >= fc_disk
    else:
        is_spheroid_circ = False
        is_disk_circ = False

    # sSFR classification
    is_low_ssfr = log_ssfr < ssfr_low
    is_high_ssfr = log_ssfr > ssfr_high

    # Check for kinematic disorder (indicator of Irr)
    has_disorder = False
    if not np.isnan(circ_asym):
        # Low asymmetry suggests disordered/irregular kinematics
        has_disorder = abs(circ_asym) < thresholds['circ_asymmetry_threshold']

    # ========== Classification Rules ==========

    # Rule 1: Strong spheroid indicators -> E
    # Very low D/T AND very low circularity
    if is_spheroid_dt and is_spheroid_circ:
        return 'E'

    # Rule 2: Either strong spheroid indicator -> E
    if is_spheroid_dt or is_spheroid_circ:
        # But if high sSFR and disordered, might be Irr
        if is_high_ssfr and has_disorder:
            return 'Irr'
        return 'E'

    # Rule 3: Strong disk indicators -> S
    # High D/T OR high circularity
    if is_disk_dt or is_disk_circ:
        return 'S'

    # Rule 4: Intermediate morphology
    if is_intermediate_dt:
        # Low sSFR relative to main sequence -> proto-S0
        if is_low_ssfr:
            return 'S0'
        # High sSFR with disorder -> Irr
        if is_high_ssfr and has_disorder:
            return 'Irr'
        # High sSFR without disorder -> forming disk (S)
        if is_high_ssfr:
            return 'S'
        # Normal sSFR, intermediate morphology -> S0
        return 'S0'

    # Rule 5: Fallback based on circularity alone
    if has_circ:
        if f_circ >= 0.20:  # Moderate disk
            return 'S'
        elif f_circ >= 0.12:  # Weak disk
            return 'S0'
        else:
            return 'E'

    # Default: S0 for truly ambiguous cases
    return 'S0'


def generate_labels(df, thresholds):
    """Generate Hubble type labels for all galaxies."""
    print("\nGenerating Hubble type labels (z~3 optimized)...")
    print(f"Key thresholds:")
    print(f"  D/T: E < {thresholds['dt_spheroid']:.2f} < S0 < {thresholds['dt_disk']:.2f} < S")
    print(f"  f_circ: E < {thresholds['f_circ_spheroid']:.2f} < S0 < {thresholds['f_circ_disk']:.2f} < S")
    print(f"  log_sSFR: low < {thresholds['log_sSFR_low']:.1f} < normal < {thresholds['log_sSFR_high']:.1f} < high")

    df['hubble_type'] = df.apply(
        lambda row: classify_hubble_type_z3(row, thresholds), axis=1
    )

    label_map = {'E': 0, 'S0': 1, 'S': 2, 'Irr': 3, 'unresolved': -1}
    df['hubble_label'] = df['hubble_type'].map(label_map)

    return df


def print_label_statistics(df):
    """Print label distribution statistics."""
    print("\n" + "=" * 80)
    print("Label Distribution")
    print("=" * 80)

    total = len(df)
    resolved = df[df['hubble_type'] != 'unresolved']
    unresolved = df[df['hubble_type'] == 'unresolved']

    print(f"\nTotal galaxies: {total}")
    print(f"Resolved (classifiable): {len(resolved)}")
    print(f"Unresolved (below mass/particle threshold): {len(unresolved)}")

    print("\nResolved distribution:")
    for hubble_type in ['E', 'S0', 'S', 'Irr']:
        count = (resolved['hubble_type'] == hubble_type).sum()
        pct = 100 * count / len(resolved) if len(resolved) > 0 else 0
        print(f"  {hubble_type:4s}: {count:6d} ({pct:5.1f}%)")

    # Per-snapshot distribution
    if 'SnapNum' in resolved.columns:
        print("\nPer-snapshot distribution (resolved only):")
        for snap in sorted(resolved['SnapNum'].unique()):
            snap_df = resolved[resolved['SnapNum'] == snap]
            snap_total = len(snap_df)
            counts = {t: (snap_df['hubble_type'] == t).sum() for t in ['E', 'S0', 'S', 'Irr']}
            pcts = {t: f"{100*c/snap_total:.0f}%" if snap_total > 0 else "0%"
                    for t, c in counts.items()}
            print(f"  Snap {snap:3d} (n={snap_total:5d}): "
                  f"E={counts['E']:5d}({pcts['E']:>4s}) "
                  f"S0={counts['S0']:5d}({pcts['S0']:>4s}) "
                  f"S={counts['S']:5d}({pcts['S']:>4s}) "
                  f"Irr={counts['Irr']:5d}({pcts['Irr']:>4s})")

    # Physical parameter statistics by Hubble type
    print("\n" + "=" * 80)
    print("Physical Parameters by Hubble Type (resolved)")
    print("=" * 80)

    param_cols = ['f_disk_morph', 'f_disk_circ', 'log_sSFR', 'v_sigma', 'log_mass_stars']

    for hubble_type in ['E', 'S0', 'S', 'Irr']:
        type_df = resolved[resolved['hubble_type'] == hubble_type]
        if len(type_df) == 0:
            continue
        print(f"\n{hubble_type} (n={len(type_df)}):")
        for col in param_cols:
            if col in type_df.columns:
                valid = type_df[col].notna()
                if valid.sum() > 0:
                    vals = type_df.loc[valid, col]
                    print(f"  {col:20s}: {vals.mean():8.3f} +/- {vals.std():6.3f} "
                          f"[{vals.min():.3f}, {vals.max():.3f}]")


def main(args):
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("TNG50 Hubble Label Generation (z~3 Optimized)")
    print("=" * 80)
    print(f"\nInput file: {input_file}")
    print(f"Output directory: {output_dir}")

    thresholds = load_config(args.config)
    print(f"Config file: {args.config if args.config else 'Using defaults'}")

    if not input_file.exists():
        print(f"\nERROR: Input file not found: {input_file}")
        return

    print(f"\nLoading physical parameters from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} entries")

    # Generate labels
    df = generate_labels(df, thresholds)

    # Print statistics
    print_label_statistics(df)

    # Save resolved labels as the main file (used by step 05)
    resolved = df[df['hubble_type'] != 'unresolved'].copy()
    output_file = output_dir / "tng50_hubble_labels.csv"
    resolved.to_csv(output_file, index=False)
    print(f"\nResolved labels saved to: {output_file} ({len(resolved)} entries)")

    # Also save ALL labels including unresolved for reference
    all_file = output_dir / "tng50_hubble_labels_all.csv"
    df.to_csv(all_file, index=False)
    print(f"All labels (incl. unresolved) saved to: {all_file} ({len(df)} entries)")

    # Save label summary
    summary = {
        'total_galaxies': int(len(df)),
        'resolved_galaxies': int(len(resolved)),
        'unresolved_galaxies': int(len(df) - len(resolved)),
        'thresholds': {k: float(v) if isinstance(v, (int, float)) else v
                       for k, v in thresholds.items()},
        'distribution': {k: int(v) for k, v in resolved['hubble_type'].value_counts().items()},
    }

    summary_file = output_dir / "label_summary.yaml"
    with open(summary_file, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    print(f"Label summary saved to: {summary_file}")

    print("\n" + "=" * 80)
    print("Label Generation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Hubble morphology labels from TNG50 physical parameters (z~3 optimized)"
    )

    parser.add_argument("--input_file", type=str,
                        default="./data/tng50/processed/catalog/tng50_physical_params.csv")
    parser.add_argument("--output_dir", type=str,
                        default="./data/tng50/processed/labels")
    parser.add_argument("--config", type=str,
                        default="./configs/tng50_hubble.yaml")

    args = parser.parse_args()
    main(args)
