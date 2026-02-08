#!/bin/bash
# Example training commands for GZ2 DINOv3 fine-tuning

# Set base directory
BASE_DIR="/home/shiqiu/dinov3-tng50-finetune"

# =============================================================================
# Example 1: Standard LoRA Training (RECOMMENDED)
# =============================================================================
echo "Example 1: Standard LoRA Training"
echo "This is the recommended starting point"
echo ""
echo "python scripts/train_gz2_dinov3.py \\"
echo "    --data_dir ${BASE_DIR} \\"
echo "    --output_dir ${BASE_DIR}/outputs/gz2_lora_r16 \\"
echo "    --model_name facebook/dinov2-large \\"
echo "    --use_lora \\"
echo "    --lora_r 16 \\"
echo "    --lora_alpha 32 \\"
echo "    --epochs 30 \\"
echo "    --batch_size 32 \\"
echo "    --learning_rate 5e-4 \\"
echo "    --weight_decay 0.01 \\"
echo "    --warmup_ratio 0.1 \\"
echo "    --num_workers 8 \\"
echo "    --save_freq 5"
echo ""
echo "Expected: ~85-90% accuracy, ~2-3 hours on A100"
echo "Memory: ~12-16 GB"
echo ""

# Uncomment to run:
# python scripts/train_gz2_dinov3.py \
#     --data_dir ${BASE_DIR} \
#     --output_dir ${BASE_DIR}/outputs/gz2_lora_r16 \
#     --model_name facebook/dinov2-large \
#     --use_lora \
#     --lora_r 16 \
#     --lora_alpha 32 \
#     --epochs 30 \
#     --batch_size 32 \
#     --learning_rate 5e-4 \
#     --weight_decay 0.01 \
#     --warmup_ratio 0.1 \
#     --num_workers 8 \
#     --save_freq 5

# =============================================================================
# Example 2: Higher Capacity LoRA (More Parameters)
# =============================================================================
echo "========================================="
echo "Example 2: Higher Capacity LoRA"
echo "Use if Example 1 underfits"
echo ""
echo "python scripts/train_gz2_dinov3.py \\"
echo "    --data_dir ${BASE_DIR} \\"
echo "    --output_dir ${BASE_DIR}/outputs/gz2_lora_r32 \\"
echo "    --use_lora \\"
echo "    --lora_r 32 \\"
echo "    --lora_alpha 64 \\"
echo "    --epochs 30 \\"
echo "    --batch_size 32 \\"
echo "    --learning_rate 5e-4"
echo ""
echo "Expected: ~87-91% accuracy"
echo ""

# =============================================================================
# Example 3: Full Fine-tuning with Partial Freezing
# =============================================================================
echo "========================================="
echo "Example 3: Full Fine-tuning"
echo "Use if LoRA is not sufficient"
echo ""
echo "python scripts/train_gz2_dinov3.py \\"
echo "    --data_dir ${BASE_DIR} \\"
echo "    --output_dir ${BASE_DIR}/outputs/gz2_full_finetune \\"
echo "    --no-use_lora \\"
echo "    --freeze_first_n_layers 12 \\"
echo "    --epochs 50 \\"
echo "    --batch_size 16 \\"
echo "    --learning_rate 1e-5 \\"
echo "    --weight_decay 0.05"
echo ""
echo "Expected: ~88-92% accuracy, ~6-8 hours on A100"
echo "Memory: ~20-24 GB"
echo ""

# =============================================================================
# Example 4: Small Batch Size (Limited GPU Memory)
# =============================================================================
echo "========================================="
echo "Example 4: Small Batch for Limited Memory"
echo "Use if you get OOM errors"
echo ""
echo "python scripts/train_gz2_dinov3.py \\"
echo "    --data_dir ${BASE_DIR} \\"
echo "    --output_dir ${BASE_DIR}/outputs/gz2_lora_bs8 \\"
echo "    --use_lora \\"
echo "    --lora_r 16 \\"
echo "    --epochs 30 \\"
echo "    --batch_size 8 \\"
echo "    --learning_rate 5e-4"
echo ""

# =============================================================================
# Evaluation Example
# =============================================================================
echo "========================================="
echo "Evaluation on Test Set"
echo ""
echo "python scripts/evaluate_gz2_dinov3.py \\"
echo "    --checkpoint ${BASE_DIR}/outputs/gz2_lora_r16/best_model_acc.pth \\"
echo "    --data_dir ${BASE_DIR} \\"
echo "    --split test \\"
echo "    --output_dir ${BASE_DIR}/evaluation_results/gz2_test \\"
echo "    --batch_size 64"
echo ""

# =============================================================================
# Inference Examples
# =============================================================================
echo "========================================="
echo "Single Image Inference"
echo ""
echo "python scripts/inference_gz2_dinov3.py \\"
echo "    --checkpoint ${BASE_DIR}/outputs/gz2_lora_r16/best_model_acc.pth \\"
echo "    --image path/to/galaxy.jpg \\"
echo "    --output prediction.json"
echo ""

echo "========================================="
echo "Batch Inference"
echo ""
echo "python scripts/inference_gz2_dinov3.py \\"
echo "    --checkpoint ${BASE_DIR}/outputs/gz2_lora_r16/best_model_acc.pth \\"
echo "    --image_dir path/to/galaxy_images/ \\"
echo "    --output predictions.csv \\"
echo "    --batch_size 32"
echo ""

# =============================================================================
# Utility: Check GPU
# =============================================================================
echo "========================================="
echo "Check GPU Status"
nvidia-smi

echo ""
echo "========================================="
echo "To run an example, uncomment the corresponding section in this script"
echo "or copy-paste the command into your terminal"
echo "========================================="
