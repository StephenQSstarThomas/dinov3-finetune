# GZ2 DINOv3 Training Guide

Complete training pipeline for Galaxy Zoo 2 5-class morphology classification using DINOv3 ViT-L/16.

## Dataset Structure

```
dinov3-tng50-finetune/
├── gz2_5class_train/
│   ├── cigar_shaped_smooth/
│   ├── completely_round_smooth/
│   ├── edge_on/
│   ├── in_between_smooth/
│   └── spiral/
├── gz2_5class_val/
│   └── (same structure)
├── gz2_5class_test/
│   └── (same structure)
└── gz2_5class_metadata.csv
```

**Dataset Stats:**
- Train: 19,995 images (~4,000 per class)
- Val: 2,500 images (~500 per class)
- Test: 2,500 images (~500 per class)

## Model: DINOv3 ViT-L/16

**Model ID:** `facebook/dinov2-large` (ViT-L/14, note: use dinov2-large for ViT-L)

**Architecture:**
- Parameters: ~300M
- Hidden size: 1024
- Layers: 24 transformer blocks
- Pre-trained on: ImageNet-1K and other vision tasks

## Training Strategies

### Strategy 1: LoRA Fine-tuning (Recommended)

**Pros:**
- ✅ Parameter efficient (~1-2% trainable parameters)
- ✅ Fast training, lower memory
- ✅ Good regularization, less overfitting
- ✅ Easy to experiment with different configs

**Recommended hyperparameters:**
```bash
python scripts/train_gz2_dinov3.py \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1
```

**LoRA Configuration:**
- **Rank (r):** 16-32 (higher for complex features)
- **Alpha:** Usually 2×rank
- **Target modules:** query, key, value, dense (all attention layers)
- **Trainable parameters:** ~2-5M (< 2% of total)

### Strategy 2: Full Fine-tuning

**Pros:**
- ✅ Maximum model capacity
- ✅ Potentially higher accuracy

**Cons:**
- ❌ More memory required
- ❌ Higher risk of overfitting
- ❌ Longer training time

**Recommended hyperparameters:**
```bash
python scripts/train_gz2_dinov3.py \
    --no-use_lora \
    --freeze_first_n_layers 12 \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --weight_decay 0.05 \
    --warmup_ratio 0.1
```

**Layer-wise Learning Rates:**
- First 12 layers: lr × 0.1 (or frozen)
- Last 12 layers: lr × 0.5
- Classification head: lr × 1.0

## Training Commands

### 1. Basic LoRA Training

```bash
python scripts/train_gz2_dinov3.py \
    --data_dir /home/shiqiu/dinov3-tng50-finetune \
    --output_dir outputs/gz2_dinov3_lora \
    --use_lora \
    --lora_r 16 \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 5e-4
```

### 2. Higher Capacity LoRA (More Parameters)

```bash
python scripts/train_gz2_dinov3.py \
    --data_dir /home/shiqiu/dinov3-tng50-finetune \
    --output_dir outputs/gz2_dinov3_lora_r32 \
    --use_lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 5e-4
```

### 3. Full Fine-tuning with Partial Freezing

```bash
python scripts/train_gz2_dinov3.py \
    --data_dir /home/shiqiu/dinov3-tng50-finetune \
    --output_dir outputs/gz2_dinov3_full_finetune \
    --no-use_lora \
    --freeze_first_n_layers 12 \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --weight_decay 0.05
```

### 4. Small Batch Size for Limited GPU Memory

```bash
python scripts/train_gz2_dinov3.py \
    --data_dir /home/shiqiu/dinov3-tng50-finetune \
    --output_dir outputs/gz2_dinov3_lora_bs16 \
    --use_lora \
    --lora_r 16 \
    --epochs 30 \
    --batch_size 16 \
    --learning_rate 5e-4
```

## Evaluation

### Evaluate on Test Set

```bash
python scripts/evaluate_gz2_dinov3.py \
    --checkpoint outputs/gz2_dinov3_lora/best_model_acc.pth \
    --data_dir /home/shiqiu/dinov3-tng50-finetune \
    --split test \
    --output_dir evaluation_results/gz2_test \
    --batch_size 64
```

**Outputs:**
- `test_results.json`: Overall and per-class metrics
- `test_confusion_matrix.png`: Normalized confusion matrix
- `test_confusion_matrix_raw.png`: Raw counts confusion matrix
- `test_per_class_metrics.png`: Precision/Recall/F1 bar chart
- `test_predictions.csv`: Per-image predictions with probabilities

## Inference

### Single Image Prediction

```bash
python scripts/inference_gz2_dinov3.py \
    --checkpoint outputs/gz2_dinov3_lora/best_model_acc.pth \
    --image path/to/galaxy_image.jpg \
    --output prediction_result.json
```

### Batch Prediction

```bash
python scripts/inference_gz2_dinov3.py \
    --checkpoint outputs/gz2_dinov3_lora/best_model_acc.pth \
    --image_dir path/to/galaxy_images/ \
    --output batch_predictions.csv \
    --batch_size 32
```

## Expected Results

Based on similar galaxy morphology tasks with DINOv3:

### LoRA Fine-tuning (r=16, 30 epochs)
- **Expected Accuracy:** 85-90%
- **Training Time:** ~2-3 hours on A100
- **Memory Usage:** ~12-16 GB
- **Trainable Parameters:** ~2-3M

### Full Fine-tuning (50 epochs)
- **Expected Accuracy:** 88-92%
- **Training Time:** ~6-8 hours on A100
- **Memory Usage:** ~20-24 GB
- **Trainable Parameters:** ~300M

### Per-Class Performance (typical):
- **Spiral:** 90-95% (most distinctive)
- **Edge-on:** 85-90% (clear features)
- **Completely round smooth:** 85-90%
- **In-between smooth:** 80-85% (more ambiguous)
- **Cigar-shaped smooth:** 75-80% (fewer samples, harder to distinguish)

## Hyperparameter Tuning Tips

### Learning Rate
- **LoRA:** Start with 5e-4, try 1e-3 if training is slow
- **Full fine-tuning:** Start with 1e-5, use layer-wise LR

### Batch Size
- **Larger (32-64):** Better for LoRA, more stable gradients
- **Smaller (8-16):** Necessary for full fine-tuning on limited memory

### LoRA Rank
- **r=8:** Minimal parameters, fast, may underfit
- **r=16:** Good balance (recommended)
- **r=32:** More capacity, slower but potentially better

### Regularization
- **Dropout:** 0.1 default, increase to 0.2 if overfitting
- **Weight decay:** 0.01 for LoRA, 0.05 for full fine-tuning
- **Data augmentation:** Already includes rotations, flips, color jitter

### Warmup Ratio
- **0.1:** Standard, helps stabilize early training
- **0.2:** If experiencing early training instability

## Monitoring Training

The training script saves:
- `config.json`: All hyperparameters
- `training_history.csv`: Loss and metrics per epoch
- `best_model_acc.pth`: Best model by validation accuracy
- `best_model_f1.pth`: Best model by validation F1 score
- `final_model.pth`: Model after last epoch
- `checkpoint_epoch_N.pth`: Periodic checkpoints (every 5 epochs)

**Monitor metrics:**
```bash
# View training history
cat outputs/gz2_dinov3_lora/training_history.csv

# Or plot with Python/pandas
python -c "import pandas as pd; import matplotlib.pyplot as plt; df = pd.read_csv('outputs/gz2_dinov3_lora/training_history.csv'); df[['train_acc', 'val_acc']].plot(); plt.savefig('training_curve.png')"
```

## Common Issues

### Out of Memory (OOM)
- ✅ Reduce batch size: `--batch_size 8`
- ✅ Use LoRA instead of full fine-tuning
- ✅ Reduce image size: `--image_size 224` (already default)
- ✅ Enable gradient checkpointing (requires code modification)

### Overfitting (val_acc < train_acc)
- ✅ Use LoRA with lower rank
- ✅ Increase dropout: `--dropout 0.2`
- ✅ Increase weight decay: `--weight_decay 0.05`
- ✅ Add more data augmentation

### Underfitting (both low)
- ✅ Increase LoRA rank: `--lora_r 32`
- ✅ Train longer: `--epochs 50`
- ✅ Increase learning rate: `--learning_rate 1e-3`
- ✅ Try full fine-tuning

### Slow Training
- ✅ Increase batch size if memory allows
- ✅ Use more workers: `--num_workers 16`
- ✅ Ensure using GPU (check `nvidia-smi`)
- ✅ Use LoRA instead of full fine-tuning

## Next Steps

1. **Start with LoRA training** (recommended first experiment)
2. **Evaluate on test set** to get baseline performance
3. **Try different LoRA ranks** (8, 16, 32) to find best trade-off
4. **If accuracy insufficient**, try full fine-tuning
5. **Error analysis**: Check confusion matrix to identify problem classes
6. **Class-specific improvements**: Add more data or augmentation for difficult classes

## References

- **DINOv3 Paper:** "DINOv2: Learning Robust Visual Features without Supervision"
- **LoRA Paper:** "LoRA: Low-Rank Adaptation of Large Language Models"
- **GZ2 Paper:** Willett et al. 2013, MNRAS
- **Classification Scheme:** Cao et al. 2024, A&A 683, A42
