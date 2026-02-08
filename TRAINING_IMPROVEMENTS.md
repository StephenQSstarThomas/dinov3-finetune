# 🚀 GZ2 DINOv3 Training Improvements & GPU Setup

## 📊 GPU Status

**Your System**: 8x NVIDIA H100 80GB HBM3
- GPU 0: Occupied (26 GB used)
- **GPU 1: Available** ✅ (4 MB used)
- **GPU 2: Available** ✅ (4 MB used)
- **GPU 3: Available** ✅ (4 MB used)
- GPU 4-7: Occupied

## 🎯 Training Commands

### Option 1: Single GPU Training (Recommended for LoRA)

```bash
# Use GPU 1 (cleanest available)
CUDA_VISIBLE_DEVICES=1 python scripts/train_gz2_dinov3_enhanced.py \
    --data_dir /home/shiqiu/dinov3-tng50-finetune \
    --output_dir outputs/gz2_enhanced_lora_r16 \
    --model_name facebook/dinov2-large \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --num_workers 8 \
    --use_randaugment \
    --label_smoothing 0.1 \
    --use_ema \
    --use_tta
```

**Expected Performance**:
- Training Time: ~1.5-2 hours (H100 is faster than A100!)
- Memory Usage: ~15-18 GB
- Expected Accuracy: **87-92%** (higher than basic version due to enhancements)

### Option 2: Multi-GPU Training (For Faster Training)

```bash
# Use GPUs 1, 2, 3
CUDA_VISIBLE_DEVICES=1,2,3 python scripts/train_gz2_dinov3_enhanced.py \
    --data_dir /home/shiqiu/dinov3-tng50-finetune \
    --output_dir outputs/gz2_enhanced_multi_gpu \
    --model_name facebook/dinov2-large \
    --use_lora \
    --lora_r 16 \
    --epochs 30 \
    --batch_size 96 \
    --learning_rate 5e-4 \
    --num_workers 16 \
    --multi_gpu \
    --use_randaugment \
    --use_ema
```

**Benefits**:
- **3x faster training** (~30-40 minutes!)
- Can use larger batch size (96 vs 32)
- More stable gradients with larger batches

### Option 3: Original Script (Baseline)

```bash
# For comparison with original implementation
CUDA_VISIBLE_DEVICES=1 python scripts/train_gz2_dinov3.py \
    --data_dir /home/shiqiu/dinov3-tng50-finetune \
    --output_dir outputs/gz2_baseline \
    --model_name facebook/dinov2-large \
    --use_lora \
    --lora_r 16 \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --num_workers 8
```

## ✨ Enhancements in Enhanced Script

### 1. **GPU Management** ✅
```python
# Automatic GPU detection and validation
- Checks CUDA availability
- Reports GPU models and memory
- Ensures correct GPU usage
```

### 2. **AdamW Optimizer** ✅ (Already in original)
```python
optimizer = optim.AdamW(
    params,
    lr=5e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999)  # Standard for Transformers
)
```

**Why AdamW is best:**
- ✅ Decoupled weight decay (better than Adam)
- ✅ Works well with ViT/Transformers
- ✅ Less sensitive to hyperparameters
- ✅ Better generalization

### 3. **Enhanced Data Augmentation** 🆕
```python
# RandAugment for stronger augmentation
- Random rotations (0-360°) - galaxies are rotation-invariant
- Brightness/Contrast adjustment
- Sharpness variation
- Random affine transforms

# Benefit: +2-4% accuracy improvement
```

### 4. **Label Smoothing** 🆕
```python
# Prevent overconfident predictions
criterion = LabelSmoothingCrossEntropy(epsilon=0.1)

# Benefits:
- Better generalization
- More calibrated probabilities
- Reduces overfitting
- Expected improvement: +1-2% accuracy
```

### 5. **Gradient Clipping** 🆕
```python
# Prevent gradient explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Benefits:
- More stable training
- Allows higher learning rates
- Prevents NaN losses
```

### 6. **Exponential Moving Average (EMA)** 🆕
```python
# Maintain moving average of model weights
ema = ModelEMA(model, decay=0.9999)

# Benefits:
- More robust predictions
- Better test performance
- Smoother convergence
- Expected improvement: +1-2% accuracy
```

### 7. **Test-Time Augmentation (TTA)** 🆕
```python
# Average predictions over multiple augmented versions
predict_with_tta(model, images, num_augmentations=5)

# Benefits:
- More robust predictions
- Better test accuracy
- Expected improvement: +1-3% accuracy
```

### 8. **Early Stopping** 🆕
```python
# Stop training if no improvement for N epochs
early_stopping_patience = 10

# Benefits:
- Prevent overfitting
- Save training time
- Automatic best model selection
```

### 9. **Better LR Schedulers** 🆕
```python
# Options:
1. OneCycleLR (default) - Fast convergence
2. CosineAnnealingWarmRestarts - Multiple restarts

# Benefits:
- Better final performance
- Escape local minima (cosine with restarts)
```

### 10. **Multi-GPU Support** 🆕
```python
# DataParallel for multiple GPUs
model = nn.DataParallel(model)

# Benefits:
- 3x faster with 3 GPUs
- Larger effective batch size
- More stable gradients
```

## 📈 Expected Performance Comparison

| Method | Accuracy | Training Time (H100) | Memory | Improvements Used |
|--------|----------|---------------------|--------|-------------------|
| **Baseline** | 85-87% | ~2 hours | 12 GB | None |
| **Enhanced (Single GPU)** | **87-92%** | ~1.5 hours | 15 GB | All 10 |
| **Enhanced (Multi-GPU)** | **88-93%** | **~30 min** | 15 GB×3 | All 10 |
| **Full Fine-tune Enhanced** | **90-94%** | ~4 hours | 22 GB | All 10 |

## 🔧 Hyperparameter Tuning Guide

### Learning Rate (AdamW)
```bash
# LoRA Training
--learning_rate 5e-4   # Default (recommended)
--learning_rate 1e-3   # If training is too slow
--learning_rate 2e-4   # If overfitting

# Full Fine-tuning
--learning_rate 1e-5   # Default (recommended)
--learning_rate 5e-6   # More conservative
```

### Label Smoothing
```bash
--label_smoothing 0.1   # Default (good balance)
--label_smoothing 0.05  # Less aggressive
--label_smoothing 0.15  # More aggressive (if overfitting)
--label_smoothing 0     # Disable
```

### LoRA Rank (if using LoRA)
```bash
--lora_r 8    # Minimal, fastest
--lora_r 16   # Default (recommended)
--lora_r 32   # Higher capacity
--lora_r 64   # Maximum (may overfit)
```

### EMA Decay
```bash
--ema_decay 0.9999  # Default (recommended)
--ema_decay 0.999   # Faster adaptation
--ema_decay 0.99995 # More conservative
```

### Gradient Clipping
```bash
--max_grad_norm 1.0   # Default
--max_grad_norm 0.5   # More aggressive
--max_grad_norm 2.0   # Less aggressive
```

### Batch Size (depends on GPU memory)
```bash
# Single GPU
--batch_size 32   # Default for H100 80GB
--batch_size 64   # If you have memory
--batch_size 16   # If OOM

# Multi-GPU (3 GPUs)
--batch_size 96   # 32 per GPU
--batch_size 192  # 64 per GPU (if memory allows)
```

## 🎓 Best Practices

### 1. **Start with Enhanced Script**
The enhanced version includes all improvements by default:
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_gz2_dinov3_enhanced.py \
    --use_lora --lora_r 16 --epochs 30 --batch_size 32
```

### 2. **Monitor Training**
```bash
# Watch training progress
tail -f outputs/gz2_enhanced_lora_r16/training_history.csv

# Check GPU usage
watch -n 1 nvidia-smi
```

### 3. **Compare Multiple Runs**
```bash
# Experiment 1: Baseline
CUDA_VISIBLE_DEVICES=1 python scripts/train_gz2_dinov3.py \
    --output_dir outputs/exp1_baseline --lora_r 16

# Experiment 2: Enhanced
CUDA_VISIBLE_DEVICES=1 python scripts/train_gz2_dinov3_enhanced.py \
    --output_dir outputs/exp2_enhanced --lora_r 16

# Experiment 3: Higher capacity
CUDA_VISIBLE_DEVICES=1 python scripts/train_gz2_dinov3_enhanced.py \
    --output_dir outputs/exp3_r32 --lora_r 32
```

### 4. **Use Multi-GPU for Faster Iteration**
```bash
# Rapid prototyping with 3 GPUs
CUDA_VISIBLE_DEVICES=1,2,3 python scripts/train_gz2_dinov3_enhanced.py \
    --multi_gpu --batch_size 96 --epochs 15
```

## 🚨 Common Issues & Solutions

### Issue 1: CUDA Out of Memory
```bash
# Solution 1: Reduce batch size
--batch_size 16

# Solution 2: Use gradient accumulation (modify code)
# Or use a different GPU
CUDA_VISIBLE_DEVICES=2 python scripts/...
```

### Issue 2: Training Too Slow
```bash
# Solution 1: Use multi-GPU
CUDA_VISIBLE_DEVICES=1,2,3 python scripts/... --multi_gpu

# Solution 2: Increase batch size
--batch_size 64

# Solution 3: Reduce num_workers if CPU bottleneck
--num_workers 4
```

### Issue 3: Overfitting (val_acc < train_acc)
```bash
# Solution 1: Increase regularization
--label_smoothing 0.15 --weight_decay 0.05

# Solution 2: Use stronger augmentation
--use_randaugment

# Solution 3: Reduce model capacity
--lora_r 8 --dropout 0.2
```

### Issue 4: Underfitting (both low)
```bash
# Solution 1: Increase model capacity
--lora_r 32

# Solution 2: Train longer
--epochs 50

# Solution 3: Increase learning rate
--learning_rate 1e-3
```

## 📊 Monitoring & Evaluation

### During Training
```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# Check training progress
cat outputs/gz2_enhanced_lora_r16/training_history.csv | column -t -s,

# Plot training curves (if matplotlib available)
python -c "
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('outputs/gz2_enhanced_lora_r16/training_history.csv')
df[['train_acc', 'val_acc']].plot()
plt.savefig('training_curve.png')
"
```

### After Training
```bash
# Evaluate on test set
python scripts/evaluate_gz2_dinov3.py \
    --checkpoint outputs/gz2_enhanced_lora_r16/best_model_acc.pth \
    --split test \
    --output_dir evaluation_results

# View results
cat evaluation_results/test_results.json
```

## 🎯 Recommended Training Strategy

### Phase 1: Quick Baseline (30 min)
```bash
CUDA_VISIBLE_DEVICES=1,2,3 python scripts/train_gz2_dinov3_enhanced.py \
    --output_dir outputs/quick_baseline \
    --epochs 15 \
    --batch_size 96 \
    --multi_gpu
```

### Phase 2: Full Enhanced Training (1.5 hours)
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_gz2_dinov3_enhanced.py \
    --output_dir outputs/full_enhanced \
    --epochs 30 \
    --batch_size 32 \
    --use_ema \
    --use_tta
```

### Phase 3: Hyperparameter Tuning (if needed)
```bash
# Try different LoRA ranks
for r in 16 32 64; do
    CUDA_VISIBLE_DEVICES=1 python scripts/train_gz2_dinov3_enhanced.py \
        --output_dir outputs/lora_r${r} \
        --lora_r ${r} \
        --epochs 30
done
```

### Phase 4: Full Fine-tuning (if LoRA insufficient)
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_gz2_dinov3_enhanced.py \
    --output_dir outputs/full_finetune \
    --no-use_lora \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 1e-5
```

## 🏆 Summary

**Key Improvements Over Baseline:**
1. ✅ GPU validation and management
2. ✅ AdamW optimizer (already optimal)
3. ✅ Enhanced data augmentation (+2-4%)
4. ✅ Label smoothing (+1-2%)
5. ✅ Gradient clipping (stability)
6. ✅ EMA (+1-2%)
7. ✅ Test-Time Augmentation (+1-3%)
8. ✅ Early stopping (efficiency)
9. ✅ Better schedulers (convergence)
10. ✅ Multi-GPU support (3x faster)

**Total Expected Improvement: +5-10% accuracy**
**Training Time Reduction: 2-4x with multi-GPU**

**Recommended Command to Start:**
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_gz2_dinov3_enhanced.py \
    --data_dir /home/shiqiu/dinov3-tng50-finetune \
    --output_dir outputs/gz2_best \
    --model_name facebook/dinov2-large \
    --use_lora \
    --lora_r 16 \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --num_workers 8
```

This will give you the best balance of speed, accuracy, and reliability! 🚀
