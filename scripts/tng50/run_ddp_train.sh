#!/bin/bash
# DDP Training Script for DINOv3 ViT-L on TNG50 Dataset
# Uses 2 GPUs with data parallel training

# Number of GPUs
NGPUS=2

# Training parameters
MODEL_NAME="facebook/dinov3-vitl16-pretrain-sat493m"
BATCH_SIZE=32  # Per GPU batch size
EPOCHS=30
LR=1e-4
OUTPUT_DIR="./outputs/tng50_dinov3_vitl"

# Run with torchrun for DDP
torchrun --nproc_per_node=$NGPUS \
    scripts/tng50/train_tng50_dinov3.py \
    --model_name $MODEL_NAME \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LR \
    --output_dir $OUTPUT_DIR \
    --gradient_checkpointing \
    --use_amp \
    "$@"
