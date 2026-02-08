#!/usr/bin/env python3
"""
GZ2 5-Class Dataset Preparation (Based on Cao et al., 2024)
Paper: Galaxy morphology classification based on Convolutional vision Transformer (CVT)
Reference: A&A 683, A42 (2024)

Classes:
0. In-between smooth
1. Completely round smooth
2. Edge-on
3. Spiral
4. Cigar-shaped smooth
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
from collections import defaultdict

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path('/home/shiqiu/dinov3-tng50-finetune')
# 请确保这里指向正确的 GZ2 图片目录
IMAGES_DIR = Path('/home/shiqiu/.cache/kagglehub/datasets/jaimetrickz/galaxy-zoo-2-images/versions/1/images_gz2/images')
# 请确保使用包含去偏差概率(debiased)的 CSV，通常是 gz2_hart16.csv 或 zoo2MainSpecz.csv
CATALOG_PATH = BASE_DIR / 'gz2_hart16.csv'
MAPPING_PATH = Path('/home/shiqiu/.cache/kagglehub/datasets/jaimetrickz/galaxy-zoo-2-images/versions/1/gz2_filename_mapping.csv')

# Output directories
OUTPUT_TRAIN_DIR = BASE_DIR / 'gz2_5class_train'
OUTPUT_VAL_DIR = BASE_DIR / 'gz2_5class_val'
OUTPUT_TEST_DIR = BASE_DIR / 'gz2_5class_test'

# 论文对应的5个类别 (顺序参考论文 Table 1)
CLASSES = [
    'in_between_smooth',       # Class 0
    'completely_round_smooth', # Class 1
    'edge_on',                 # Class 2
    'spiral',                  # Class 3
    'cigar_shaped_smooth'      # Class 4
]

# Sampling Target
# 注意：论文中 Cigar-shaped 只有约 500 个样本。
# 如果设置 5000，脚本会自动获取该类别的最大可用数量。
SAMPLES_PER_CLASS = 5000 

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# =============================================================================
# Step 1: Load Data
# =============================================================================

def load_data():
    print("="*80)
    print("STEP 1: Loading Data")
    print("="*80)
    
    # 读取 CSV 时，确保列名正确。GZ2 Hart16 表通常包含详细的 voting info
    catalog_df = pd.read_csv(CATALOG_PATH)
    print(f"  Loaded catalog: {len(catalog_df):,} rows")
    
    mapping_df = pd.read_csv(MAPPING_PATH)
    print(f"  Loaded mapping: {len(mapping_df):,} rows")
    
    objid_to_asset = dict(zip(mapping_df['objid'], mapping_df['asset_id']))
    return catalog_df, objid_to_asset

# =============================================================================
# Step 2: Scientific Classification Logic (The Core)
# =============================================================================

def classify_galaxies_paper_logic(df):
    """
    Implement the specific decision tree from Cao et al. (2024) Section 3.1
    and Willett et al. (2013).
    """
    print("\n" + "="*80)
    print("STEP 2: Applying GZ2 Decision Tree (Cao et al. 2024)")
    print("="*80)

    # 1. Pre-filtering: "Well-sampled galaxies"
    # 论文指出: "the number of volunteers to classify it must be greater than 20"
    if 't01_smooth_or_features_a01_smooth_count' in df.columns:
        # 估算总票数 (Smooth + Features + Artifact)
        total_votes = (df['t01_smooth_or_features_a01_smooth_count'] + 
                       df['t01_smooth_or_features_a02_features_or_disk_count'] + 
                       df['t01_smooth_or_features_a03_star_or_artifact_count'])
        vote_mask = total_votes >= 20
        df = df[vote_mask].copy()
        print(f"Filtered for >20 votes: {len(df):,} remaining")
    
    df['label'] = 'uncertain'
    df['confidence'] = 0.0

    # 定义列名 (根据标准的 GZ2 列名)
    # Task 01: Smooth vs Features
    col_smooth = 't01_smooth_or_features_a01_smooth_debiased'
    col_features = 't01_smooth_or_features_a02_features_or_disk_debiased'
    
    # Task 02: Edge-on?
    col_edgeon_yes = 't02_edgeon_a04_yes_debiased'
    col_edgeon_no = 't02_edgeon_a05_no_debiased'
    
    # Task 04: Spiral?
    col_spiral_yes = 't04_spiral_a08_spiral_debiased'
    
    # Task 07: How round? (For smooth galaxies)
    col_round = 't07_rounded_a16_completely_round_debiased'
    col_in_between = 't07_rounded_a17_in_between_debiased'
    col_cigar = 't07_rounded_a18_cigar_shaped_debiased'

    # --- 逻辑实现 ---

    # 1. SPIRAL (Class 3)
    # Thresholds: Features >= 0.430, Not Edge-on >= 0.715, Spiral Yes >= 0.619
    mask_spiral = (
        (df[col_features] >= 0.430) &
        (df[col_edgeon_no] >= 0.715) &
        (df[col_spiral_yes] >= 0.619)
    )
    # 赋值 (优先判定复杂的形态)
    df.loc[mask_spiral, 'label'] = 'spiral'
    df.loc[mask_spiral, 'confidence'] = df.loc[mask_spiral, col_spiral_yes]

    # 2. EDGE-ON (Class 2)
    # Thresholds: Features >= 0.430, Edge-on Yes >= 0.602 (Standard GZ2 Clean)
    # 注意：需排除已被标记为 Spiral 的 (虽然逻辑上互斥，但防止重叠)
    mask_edgeon = (
        (df['label'] == 'uncertain') &
        (df[col_features] >= 0.430) &
        (df[col_edgeon_yes] >= 0.602)
    )
    df.loc[mask_edgeon, 'label'] = 'edge_on'
    df.loc[mask_edgeon, 'confidence'] = df.loc[mask_edgeon, col_edgeon_yes]

    # 3. SMOOTH SUBTYPES (Class 0, 1, 4)
    # Thresholds: Smooth >= 0.469
    # Subtypes: Paper explicitly widened threshold from 0.8 -> 0.5
    
    # Base smooth mask
    mask_smooth_base = (df['label'] == 'uncertain') & (df[col_smooth] >= 0.469)
    
    # 3a. Completely Round (Class 1)
    mask_round = mask_smooth_base & (df[col_round] >= 0.50)
    df.loc[mask_round, 'label'] = 'completely_round_smooth'
    df.loc[mask_round, 'confidence'] = df.loc[mask_round, col_round]
    
    # 3b. In-between (Class 0)
    # 注意：防止同一个星系满足多个 >= 0.5 (理论上归一化后很少见，但为了严谨排除已标记的)
    mask_in_between = mask_smooth_base & (df['label'] == 'uncertain') & (df[col_in_between] >= 0.50)
    df.loc[mask_in_between, 'label'] = 'in_between_smooth'
    df.loc[mask_in_between, 'confidence'] = df.loc[mask_in_between, col_in_between]
    
    # 3c. Cigar-shaped (Class 4)
    mask_cigar = mask_smooth_base & (df['label'] == 'uncertain') & (df[col_cigar] >= 0.50)
    df.loc[mask_cigar, 'label'] = 'cigar_shaped_smooth'
    df.loc[mask_cigar, 'confidence'] = df.loc[mask_cigar, col_cigar]

    # Filter out uncertain
    classified_df = df[df['label'] != 'uncertain'].copy()
    
    print("\nResulting Class Distribution:")
    print(classified_df['label'].value_counts())
    
    return classified_df

# =============================================================================
# Step 3: Balanced Sampling (Robust)
# =============================================================================

def select_balanced_samples(classified_df, target_per_class):
    print("\n" + "="*80)
    print("STEP 3: Balanced Sampling")
    print("="*80)

    selected_dfs = []
    
    for cls in CLASSES:
        class_df = classified_df[classified_df['label'] == cls].copy()
        available = len(class_df)
        
        # 论文中提到 cigar-shaped 样本很少，所以我们取 min(available, target)
        n_select = min(available, target_per_class)
        
        if n_select < target_per_class:
            print(f"WARNING: Class '{cls}' only has {available} samples. Taking all.")
        
        # 按置信度排序取最好的样本
        selected = class_df.sort_values('confidence', ascending=False).head(n_select)
        selected_dfs.append(selected)
        
        print(f"  {cls:25s}: Selected {len(selected)} (from {available})")

    balanced_df = pd.concat(selected_dfs, ignore_index=True)
    return balanced_df

# =============================================================================
# Step 4: Split (Standard)
# =============================================================================

def split_dataset(df):
    print("\n" + "="*80)
    print("STEP 4: Splitting Data")
    print("="*80)
    
    split_list = []
    
    for cls in CLASSES:
        sub_df = df[df['label'] == cls].copy()
        # Shuffle
        sub_df = sub_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        
        n = len(sub_df)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        
        sub_df.loc[:n_train, 'split'] = 'train'
        sub_df.loc[n_train:n_train+n_val, 'split'] = 'val'
        sub_df.loc[n_train+n_val:, 'split'] = 'test'
        
        split_list.append(sub_df)
        
    final_df = pd.concat(split_list, ignore_index=True)
    return final_df

# =============================================================================
# Step 5 & 6: Execution Utilities
# =============================================================================

def copy_images(df, objid_to_asset):
    print("\n" + "="*80)
    print("STEP 5: Copying Images")
    print("="*80)
    
    # Mappings
    split_path_map = {
        'train': OUTPUT_TRAIN_DIR,
        'val': OUTPUT_VAL_DIR,
        'test': OUTPUT_TEST_DIR
    }
    
    # Make dirs
    for split_path in split_path_map.values():
        for cls in CLASSES:
            (split_path / cls).mkdir(parents=True, exist_ok=True)
            
    success_count = 0
    missing_count = 0
    
    # 使用 tqdm 显示进度
    for _, row in tqdm(df.iterrows(), total=len(df)):
        objid = row['dr7objid'] # 注意：Hart16 通常叫 objid 或 dr7objid，请根据 CSV 调整
        label = row['label']
        split = row['split']
        
        # 获取文件名
        asset_id = objid_to_asset.get(objid)
        if not asset_id:
            # 尝试把 objid 转 int 或 str 再次查找
            asset_id = objid_to_asset.get(int(objid))
            
        if not asset_id:
            missing_count += 1
            continue
            
        src = IMAGES_DIR / f"{asset_id}.jpg"
        dst = split_path_map[split] / label / f"{objid}.jpg"
        
        if src.exists():
            shutil.copy2(src, dst)
            success_count += 1
        else:
            missing_count += 1
            
    print(f"\nDone. Copied: {success_count}, Missing: {missing_count}")

# =============================================================================
# Main
# =============================================================================

def main():
    # 1. Load
    catalog, mapper = load_data()
    
    # 2. Classify (Paper Logic)
    classified = classify_galaxies_paper_logic(catalog)
    
    # 3. Balance
    balanced = select_balanced_samples(classified, SAMPLES_PER_CLASS)
    
    # 4. Split
    final_df = split_dataset(balanced)
    
    # 5. Copy
    copy_images(final_df, mapper)
    
    # 6. Metadata
    final_df.to_csv(BASE_DIR / 'gz2_5class_metadata.csv', index=False)
    print("\nProcess Complete.")

if __name__ == '__main__':
    main()