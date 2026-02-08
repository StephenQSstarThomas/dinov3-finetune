#!/usr/bin/env python3
"""
Enhanced DINOv3 ViT-L/16 Training for GZ2 with Advanced Techniques

Enhancements:
1. ✅ GPU selection and validation
2. ✅ Enhanced data augmentation (RandAugment, stronger transforms)
3. ✅ Label smoothing
4. ✅ Gradient clipping
5. ✅ EMA (Exponential Moving Average)
6. ✅ Test-Time Augmentation (TTA)
7. ✅ Mixed precision with gradient scaling
8. ✅ Better learning rate schedules (Cosine with restarts)
9. ✅ Early stopping
10. ✅ Multi-GPU support (DataParallel)

Usage:
    # Use specific GPU (e.g., GPU 1)
    CUDA_VISIBLE_DEVICES=1 python scripts/train_gz2_dinov3_enhanced.py --use_lora --epochs 30

    # Multi-GPU training
    CUDA_VISIBLE_DEVICES=1,2,3 python scripts/train_gz2_dinov3_enhanced.py --use_lora --epochs 30 --multi_gpu
"""

import os
import sys
import argparse
import json
import copy
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from peft import LoraConfig, get_peft_model, TaskType

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logger(output_dir: Path, log_level: int = logging.INFO):
    """Setup comprehensive logging to both file and console"""

    # Create logs directory
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_{timestamp}.log'

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove existing handlers
    logger.handlers.clear()

    # File handler with detailed format
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler with simpler format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


def log_system_info():
    """Log system and GPU information"""
    logger = logging.getLogger()

    logger.info("=" * 80)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 80)

    # Python and PyTorch versions
    logger.info(f"Python version: {sys.version.split()[0]}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA version: {torch.version.cuda}")

    # GPU information
    if torch.cuda.is_available():
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        logger.info(f"Current GPU: {torch.cuda.current_device()}")

    logger.info("=" * 80)


def log_gpu_memory():
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logging.debug(f"GPU {i} Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")


# =============================================================================
# Enhanced Data Augmentation
# =============================================================================

class RandAugment:
    """Simplified RandAugment for astronomical images"""

    def __init__(self, n=2, m=9):
        self.n = n  # Number of augmentations to apply
        self.m = m  # Magnitude of augmentations

    def __call__(self, img):
        ops = [
            lambda img: transforms.functional.rotate(img, np.random.uniform(-180, 180)),
            lambda img: transforms.functional.adjust_brightness(img, 1 + (self.m/30) * np.random.uniform(-1, 1)),
            lambda img: transforms.functional.adjust_contrast(img, 1 + (self.m/30) * np.random.uniform(-1, 1)),
            lambda img: transforms.functional.adjust_sharpness(img, 1 + (self.m/30) * np.random.uniform(-1, 1)),
        ]

        for _ in range(self.n):
            op = np.random.choice(ops)
            img = op(img)

        return img


def get_enhanced_transforms(image_size: int = 224, split: str = 'train', use_randaugment: bool = True):
    """Enhanced data transforms"""

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == 'train':
        transforms_list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
        ]

        if use_randaugment:
            transforms_list.append(RandAugment(n=2, m=9))
        else:
            transforms_list.extend([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ])

        transforms_list.extend([
            transforms.ToTensor(),
            normalize,
        ])

        return transforms.Compose(transforms_list)
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


# =============================================================================
# Model with EMA
# =============================================================================

class ModelEMA:
    """Exponential Moving Average of model parameters"""

    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# =============================================================================
# Import models from original script
# =============================================================================

from train_gz2_dinov3 import DINOv3WithLoRA, DINOv3Classifier, GZ2Dataset


# =============================================================================
# Label Smoothing Loss
# =============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing loss"""

    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, target):
        n_classes = logits.size(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # One-hot encoding with smoothing
        targets = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / n_classes

        loss = (-targets * log_probs).sum(dim=-1).mean()
        return loss


# =============================================================================
# Test-Time Augmentation
# =============================================================================

@torch.no_grad()
def predict_with_tta(
    model: nn.Module,
    images: torch.Tensor,
    num_augmentations: int = 5,
) -> torch.Tensor:
    """Test-Time Augmentation for more robust predictions"""

    batch_size = images.size(0)
    all_logits = []

    # Original prediction
    logits = model(images)
    all_logits.append(logits)

    # Augmented predictions
    for _ in range(num_augmentations - 1):
        # Random horizontal flip
        if np.random.rand() > 0.5:
            aug_images = torch.flip(images, dims=[3])
        else:
            aug_images = images

        # Random vertical flip
        if np.random.rand() > 0.5:
            aug_images = torch.flip(aug_images, dims=[2])

        aug_logits = model(aug_images)
        all_logits.append(aug_logits)

    # Average predictions
    avg_logits = torch.stack(all_logits).mean(dim=0)
    return avg_logits


# =============================================================================
# Enhanced Training Functions
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    ema: Optional[ModelEMA] = None,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """Enhanced training with gradient clipping and EMA"""

    logger = logging.getLogger()
    model.train()

    epoch_start_time = time.time()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    logger.info(f"Starting training epoch {epoch}")
    log_gpu_memory()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [TRAIN]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        # Backward with gradient scaling
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        # Update EMA
        if ema is not None:
            ema.update()

        if scheduler is not None:
            scheduler.step()

        # Statistics
        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'lr': current_lr
        })

        # Log every N batches
        if batch_idx % 50 == 0:
            logger.debug(f"Epoch {epoch} Batch {batch_idx}/{len(dataloader)}: "
                        f"Loss={loss.item():.4f}, LR={current_lr:.6f}, GradNorm={grad_norm:.4f}")

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    epoch_time = time.time() - epoch_start_time
    logger.info(f"Epoch {epoch} training completed in {epoch_time:.2f}s: "
               f"Loss={running_loss / len(dataloader):.4f}, Acc={accuracy:.4f}, F1={f1:.4f}")

    return {
        'loss': running_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'time': epoch_time,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    split: str = 'VAL',
    use_tta: bool = False,
) -> Dict:
    """Enhanced validation with optional TTA"""

    logger = logging.getLogger()
    model.eval()

    val_start_time = time.time()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    logger.info(f"Starting {split} evaluation for epoch {epoch} (TTA={'ON' if use_tta else 'OFF'})")

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [{split:4s}]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        with autocast():
            if use_tta:
                logits = predict_with_tta(model, images, num_augmentations=5)
            else:
                logits = model(images)

            loss = criterion(logits, labels)

        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    val_time = time.time() - val_start_time
    logger.info(f"Epoch {epoch} {split} completed in {val_time:.2f}s: "
               f"Loss={running_loss / len(dataloader):.4f}, Acc={accuracy:.4f}, F1={f1:.4f}")

    return {
        'loss': running_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'time': val_time,
    }


# =============================================================================
# Main Enhanced Training
# =============================================================================

def main(args):
    # Create output directory first
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(output_dir, log_level=logging.DEBUG if args.debug else logging.INFO)

    try:
        # Log training start
        logger.info("=" * 80)
        logger.info("ENHANCED GZ2 DINOV3 TRAINING")
        logger.info("=" * 80)
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Output directory: {output_dir}")

        # Validate GPU setup
        if not torch.cuda.is_available():
            logger.error("CUDA is not available! Please check your GPU setup.")
            raise RuntimeError("CUDA is not available! Please check your GPU setup.")

        # Log system information
        log_system_info()

        # Log training configuration
        logger.info("=" * 80)
        logger.info("TRAINING CONFIGURATION")
        logger.info("=" * 80)
        for key, value in sorted(vars(args).items()):
            logger.info(f"  {key:30s}: {value}")
        logger.info("=" * 80)

        # Set random seeds
        logger.info(f"Setting random seed: {args.seed}")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True  # Enable for faster training

        device = torch.device('cuda')

        # Save config
        config_file = output_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info(f"Configuration saved to {config_file}")

        # Enhanced datasets
        logger.info("=" * 80)
        logger.info("LOADING DATASETS")
        logger.info("=" * 80)
        logger.info(f"Data directory: {args.data_dir}")
        logger.info(f"Image size: {args.image_size}")
        logger.info(f"Enhanced augmentation: {args.use_randaugment}")

        train_dataset = GZ2Dataset(
            args.data_dir,
            split='train',
            transform=get_enhanced_transforms(args.image_size, 'train', args.use_randaugment)
        )
        val_dataset = GZ2Dataset(
            args.data_dir,
            split='val',
            transform=get_enhanced_transforms(args.image_size, 'val')
        )
        test_dataset = GZ2Dataset(
            args.data_dir,
            split='test',
            transform=get_enhanced_transforms(args.image_size, 'test')
        )

        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Val dataset:   {len(val_dataset)} samples")
        logger.info(f"Test dataset:  {len(test_dataset)} samples")
        logger.info("=" * 80)

        # Dataloaders
        logger.info("Creating data loaders...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
        )
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Num workers: {args.num_workers}")
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")

        # Create model
        logger.info("=" * 80)
        logger.info("MODEL SETUP")
        logger.info("=" * 80)
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Number of classes: 5")
        logger.info(f"Use LoRA: {args.use_lora}")

        if args.use_lora:
            logger.info(f"LoRA rank: {args.lora_r}")
            logger.info(f"LoRA alpha: {args.lora_alpha}")
            logger.info(f"LoRA dropout: {args.lora_dropout}")
            model = DINOv3WithLoRA(
                model_name=args.model_name,
                num_classes=5,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                freeze_backbone=True,
                dropout=args.dropout,
            )
        else:
            logger.info(f"Dropout: {args.dropout}")
            logger.info(f"Freeze first N layers: {args.freeze_first_n_layers}")
            model = DINOv3Classifier(
                model_name=args.model_name,
                num_classes=5,
                dropout=args.dropout,
                freeze_first_n_layers=args.freeze_first_n_layers,
            )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

        # Multi-GPU support
        if args.multi_gpu and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)

        model = model.to(device)
        logger.info(f"Model moved to device: {device}")
        log_gpu_memory()
        logger.info("=" * 80)

        # EMA
        ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None
        if ema:
            logger.info(f"Using EMA with decay: {args.ema_decay}")

        # Loss function with label smoothing
        if args.label_smoothing > 0:
            criterion = LabelSmoothingCrossEntropy(epsilon=args.label_smoothing)
            logger.info(f"Using label smoothing: {args.label_smoothing}")
        else:
            criterion = nn.CrossEntropyLoss()
            logger.info("Using standard CrossEntropyLoss")

        # Optimizer
        logger.info("=" * 80)
        logger.info("OPTIMIZER & SCHEDULER")
        logger.info("=" * 80)
        if args.use_lora:
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                betas=(0.9, 0.999),
            )
            logger.info("Using AdamW optimizer for LoRA parameters")
        else:
            # Layer-wise learning rates
            backbone_params = list(model.backbone.parameters())
            n_layers = len(model.backbone.encoder.layer)

            param_groups = []
            for i in range(n_layers // 2):
                param_groups.append({
                    'params': model.backbone.encoder.layer[i].parameters(),
                    'lr': args.learning_rate * 0.1
                })
            for i in range(n_layers // 2, n_layers):
                param_groups.append({
                    'params': model.backbone.encoder.layer[i].parameters(),
                    'lr': args.learning_rate * 0.5
                })
            param_groups.append({
                'params': model.classifier.parameters(),
                'lr': args.learning_rate
            })

            optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
            logger.info("Using AdamW optimizer with layer-wise learning rates")

        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Weight decay: {args.weight_decay}")

        # Learning rate scheduler
        total_steps = len(train_loader) * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)

        if args.scheduler == 'onecycle':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.learning_rate,
                total_steps=total_steps,
                pct_start=warmup_steps / total_steps,
                anneal_strategy='cos',
            )
            logger.info(f"Using OneCycleLR scheduler (warmup: {warmup_steps} steps)")
        elif args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=len(train_loader) * 5,  # Restart every 5 epochs
                T_mult=2,
                eta_min=args.learning_rate * 0.01,
            )
            logger.info("Using CosineAnnealingWarmRestarts scheduler")
        else:
            scheduler = None
            logger.info("No scheduler used")

        logger.info("=" * 80)

        scaler = GradScaler()

        # Training history
        history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_time': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_time': [],
        }

        best_val_acc = 0.0
        best_val_f1 = 0.0
        patience_counter = 0

        # Training loop
        logger.info("=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)
        logger.info(f"Total epochs: {args.epochs}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Gradient clipping: {args.max_grad_norm}")
        logger.info(f"Label smoothing: {args.label_smoothing}")
        logger.info(f"EMA: {args.use_ema}")
        logger.info(f"RandAugment: {args.use_randaugment}")
        logger.info(f"Early stopping patience: {args.early_stopping_patience}")
        logger.info("=" * 80)

        training_start_time = time.time()

        for epoch in range(args.epochs):
            # Train
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, scaler, device,
                epoch + 1, ema=ema, max_grad_norm=args.max_grad_norm
            )

            # Validate with original model
            val_metrics = validate(model, val_loader, criterion, device, epoch + 1, 'VAL')

            # Validate with EMA model (if enabled)
            if ema is not None:
                ema.apply_shadow()
                val_ema_metrics = validate(model, val_loader, criterion, device, epoch + 1, 'EMA')
                ema.restore()

                # Use EMA metrics if better
                if val_ema_metrics['accuracy'] > val_metrics['accuracy']:
                    logger.info(f"Using EMA model (Acc: {val_ema_metrics['accuracy']:.4f} > {val_metrics['accuracy']:.4f})")
                    val_metrics = val_ema_metrics

            # Log metrics summary
            logger.info("=" * 80)
            logger.info(f"EPOCH {epoch+1}/{args.epochs} SUMMARY")
            logger.info("=" * 80)
            logger.info(f"Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f}, "
                       f"F1={train_metrics['f1']:.4f}, Time={train_metrics['time']:.2f}s")
            logger.info(f"Val:   Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.4f}, "
                       f"F1={val_metrics['f1']:.4f}, Time={val_metrics['time']:.2f}s")

            # Save history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['train_f1'].append(train_metrics['f1'])
            history['train_time'].append(train_metrics['time'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            history['val_time'].append(val_metrics['time'])

            # Save best models
            improved = False
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                save_model = model.module if args.multi_gpu else model
                checkpoint_path = output_dir / 'best_model_acc.pth'
                torch.save(save_model.state_dict(), checkpoint_path)
                logger.info(f"✓ New best accuracy: {best_val_acc:.4f} (saved to {checkpoint_path})")
                improved = True

            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                save_model = model.module if args.multi_gpu else model
                checkpoint_path = output_dir / 'best_model_f1.pth'
                torch.save(save_model.state_dict(), checkpoint_path)
                logger.info(f"✓ New best F1: {best_val_f1:.4f} (saved to {checkpoint_path})")
                improved = True

            # Early stopping
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{args.early_stopping_patience}")
                if patience_counter >= args.early_stopping_patience:
                    logger.warning(f"Early stopping triggered after {epoch+1} epochs")
                    break

            logger.info("=" * 80)
            log_gpu_memory()

        # Calculate total training time
        training_time = time.time() - training_start_time

        # Save training history
        history_file = output_dir / 'training_history.csv'
        pd.DataFrame(history).to_csv(history_file, index=False)
        logger.info(f"Training history saved to {history_file}")

        # Final evaluation with TTA
        logger.info("=" * 80)
        logger.info("FINAL EVALUATION")
        logger.info("=" * 80)

        # Load best model
        best_model_path = output_dir / 'best_model_acc.pth'
        logger.info(f"Loading best model from {best_model_path}")
        best_model_state = torch.load(best_model_path)
        if args.multi_gpu:
            model.module.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)

        test_metrics = validate(
            model, test_loader, criterion, device, args.epochs, 'TEST', use_tta=args.use_tta
        )

        logger.info("=" * 80)
        logger.info(f"TEST RESULTS (TTA={'ON' if args.use_tta else 'OFF'})")
        logger.info("=" * 80)
        logger.info(f"Accuracy:  {test_metrics['accuracy']:.4f}")
        logger.info(f"Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Recall:    {test_metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {test_metrics['f1']:.4f}")
        logger.info("=" * 80)

        # Save results
        test_results = {
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
            'use_tta': args.use_tta,
            'best_val_accuracy': best_val_acc,
            'best_val_f1': best_val_f1,
            'total_training_time': training_time,
        }

        results_file = output_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"Test results saved to {results_file}")

        # Training summary
        logger.info("=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total training time: {training_time/3600:.2f} hours")
        logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
        logger.info(f"Best validation F1: {best_val_f1:.4f}")
        logger.info(f"Final test accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Final test F1: {test_metrics['f1']:.4f}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("=" * 80)
        logger.error("TRAINING FAILED WITH ERROR")
        logger.error("=" * 80)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        logger.error("=" * 80)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced GZ2 DINOv3 Training')

    # Data
    parser.add_argument('--data_dir', type=str, default='/home/shiqiu/dinov3-tng50-finetune')
    parser.add_argument('--output_dir', type=str, default='outputs/gz2_dinov3_enhanced')

    # Model
    parser.add_argument('--model_name', type=str, default='facebook/dinov2-large')
    parser.add_argument('--use_lora', action='store_true', default=True)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--freeze_first_n_layers', type=int, default=12)

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--image_size', type=int, default=224)

    # Enhancements
    parser.add_argument('--use_randaugment', action='store_true', default=True,
                        help='Use RandAugment for data augmentation')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor (0=disabled)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use Exponential Moving Average')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='EMA decay rate')
    parser.add_argument('--use_tta', action='store_true', default=True,
                        help='Use Test-Time Augmentation for final evaluation')
    parser.add_argument('--scheduler', type=str, default='onecycle',
                        choices=['onecycle', 'cosine', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience (epochs)')

    # System
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Use multiple GPUs with DataParallel')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    args = parser.parse_args()

    main(args)
