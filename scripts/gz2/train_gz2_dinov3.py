#!/usr/bin/env python3
"""
DINOv3 ViT-L/16 Fine-tuning for Galaxy Zoo 2 5-Class Classification

Model: facebook/dinov3-vitl16-pretrain-sat493m
Classes: 0=in_between_smooth, 1=completely_round_smooth, 2=edge_on, 3=spiral, 4=cigar_shaped_smooth

Training Strategies:
1. LoRA fine-tuning (parameter-efficient, recommended)
2. Full fine-tuning with layer-wise learning rates

Usage:
    # LoRA training (recommended)
    python scripts/train_gz2_dinov3.py --use_lora --lora_r 16 --epochs 30

    # Full fine-tuning
    python scripts/train_gz2_dinov3.py --no-use_lora --epochs 50
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Logging setup
import logging
from torch.utils.tensorboard import SummaryWriter


# =============================================================================
# Model Definitions
# =============================================================================

class DINOv3WithLoRA(nn.Module):
    """DINOv3 ViT-L/16 with LoRA for efficient fine-tuning"""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        num_classes: int = 5,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        freeze_backbone: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        print(f"Loading DINOv3 model: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.hidden_size = self.backbone.config.hidden_size

        # Configure LoRA
        print(f"Applying LoRA (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value", "dense"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.backbone = get_peft_model(self.backbone, lora_config)

        # Freeze non-LoRA parameters if specified
        if freeze_backbone:
            print("Freezing backbone parameters (only LoRA adapters will train)")
            for name, param in self.backbone.named_parameters():
                if "lora" not in name.lower():
                    param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_classes)
        )

        self._print_trainable_parameters()

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)

        # Use CLS token
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0]

        logits = self.classifier(features)
        return logits

    def _print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


class DINOv3Classifier(nn.Module):
    """Standard DINOv3 ViT-L/16 with classification head (full fine-tuning)"""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        num_classes: int = 5,
        dropout: float = 0.1,
        freeze_first_n_layers: int = 0,
    ):
        super().__init__()

        print(f"Loading DINOv3 model: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.hidden_size = self.backbone.config.hidden_size

        # Optionally freeze first N transformer layers
        if freeze_first_n_layers > 0:
            print(f"Freezing first {freeze_first_n_layers} transformer layers")
            for i in range(freeze_first_n_layers):
                for param in self.backbone.encoder.layer[i].parameters():
                    param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_classes)
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)

        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0]

        logits = self.classifier(features)
        return logits


# =============================================================================
# Dataset
# =============================================================================

class GZ2Dataset(Dataset):
    """Galaxy Zoo 2 5-class dataset"""

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform = None,
        use_confidence: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.use_confidence = use_confidence

        # Class names (alphabetically ordered by folder name)
        self.classes = [
            'cigar_shaped_smooth',
            'completely_round_smooth',
            'edge_on',
            'in_between_smooth',
            'spiral'
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Load image paths and labels
        self.samples = []
        split_dir = self.data_dir / f'gz2_5class_{split}'

        for class_name in self.classes:
            class_dir = split_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))

        print(f"{split.upper()}: {len(self.samples)} images")

        # Print class distribution
        labels = [s[1] for s in self.samples]
        for cls_name, cls_idx in self.class_to_idx.items():
            count = labels.count(cls_idx)
            print(f"  {cls_name:25s}: {count:5d}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(image_size: int = 224, split: str = 'train'):
    """Get data transforms for training/validation"""

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),  # Galaxies are rotationally invariant
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


# =============================================================================
# Training Functions
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
    logger: Optional[logging.Logger] = None,
    writer: Optional[SummaryWriter] = None,
    global_step: int = 0,
) -> Tuple[Dict[str, float], int]:
    """Train for one epoch"""
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    batch_losses = []
    batch_lrs = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [TRAIN]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        if scheduler is not None:
            scheduler.step()

        # Statistics
        running_loss += loss.item()
        batch_losses.append(loss.item())
        batch_lrs.append(current_lr)

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'lr': current_lr
        })

        # Log to tensorboard every 10 batches
        if writer is not None and batch_idx % 10 == 0:
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            writer.add_scalar('Train/LearningRate', current_lr, global_step)

        global_step += 1

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    # Log epoch summary
    if logger:
        logger.info(f"Epoch {epoch} - Train Loss: {running_loss / len(dataloader):.4f}, "
                   f"Acc: {accuracy:.4f}, F1: {f1:.4f}, LR: {batch_lrs[-1]:.6f}")

    return {
        'loss': running_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'batch_losses': batch_losses,
        'batch_lrs': batch_lrs,
        'learning_rate': batch_lrs[-1] if batch_lrs else 0,
    }, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    split: str = 'VAL',
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Validate the model"""
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [{split:4s}]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    # Log validation results
    if logger:
        logger.info(f"Epoch {epoch} - {split} Loss: {running_loss / len(dataloader):.4f}, "
                   f"Acc: {accuracy:.4f}, F1: {f1:.4f}")

    return {
        'loss': running_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict,
    args: argparse.Namespace,
    filename: Path,
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'args': vars(args),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")


# =============================================================================
# Main Training Loop
# =============================================================================

def main(args):
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    # Text logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # TensorBoard logger
    tensorboard_dir = output_dir / 'tensorboard'
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)

    logger.info(f"\n{'='*80}")
    logger.info(f"Training Configuration")
    logger.info(f"{'='*80}")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA: {torch.version.cuda}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Strategy: {'LoRA' if args.use_lora else 'Full Fine-tuning'}")
    logger.info(f"Image Size: {args.image_size}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Log Directory: {log_dir}")
    logger.info(f"TensorBoard Directory: {tensorboard_dir}")
    logger.info(f"{'='*80}\n")

    print(f"\n{'='*80}")
    print(f"Training Configuration")
    print(f"{'='*80}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    print(f"Model: {args.model_name}")
    print(f"Strategy: {'LoRA' if args.use_lora else 'Full Fine-tuning'}")
    print(f"Image Size: {args.image_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"{'='*80}\n")

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Create datasets
    print("Loading datasets...")
    train_dataset = GZ2Dataset(
        args.data_dir,
        split='train',
        transform=get_transforms(args.image_size, 'train')
    )
    val_dataset = GZ2Dataset(
        args.data_dir,
        split='val',
        transform=get_transforms(args.image_size, 'val')
    )
    test_dataset = GZ2Dataset(
        args.data_dir,
        split='test',
        transform=get_transforms(args.image_size, 'test')
    )

    # Create dataloaders
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

    # Create model
    print("\nCreating model...")
    if args.use_lora:
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
        model = DINOv3Classifier(
            model_name=args.model_name,
            num_classes=5,
            dropout=args.dropout,
            freeze_first_n_layers=args.freeze_first_n_layers,
        )

    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if args.use_lora:
        # Only optimize LoRA parameters and classifier
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        # Layer-wise learning rates for full fine-tuning
        backbone_params = list(model.backbone.parameters())
        n_layers = len(model.backbone.encoder.layer)

        param_groups = []
        # First half of layers: smaller lr
        for i in range(n_layers // 2):
            param_groups.append({
                'params': model.backbone.encoder.layer[i].parameters(),
                'lr': args.learning_rate * 0.1
            })
        # Second half: medium lr
        for i in range(n_layers // 2, n_layers):
            param_groups.append({
                'params': model.backbone.encoder.layer[i].parameters(),
                'lr': args.learning_rate * 0.5
            })
        # Classifier: full lr
        param_groups.append({
            'params': model.classifier.parameters(),
            'lr': args.learning_rate
        })

        optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy='cos',
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # Training history
    history = {
        'epoch': [],
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_lr': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
    }

    best_val_acc = 0.0
    best_val_f1 = 0.0
    global_step = 0

    # Training loop
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Training")
    logger.info(f"{'='*80}\n")
    print(f"\n{'='*80}")
    print(f"Starting Training")
    print(f"{'='*80}\n")

    for epoch in range(args.epochs):
        # Train
        train_metrics, global_step = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device,
            epoch + 1, logger, writer, global_step
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch + 1, 'VAL', logger)

        # Print metrics
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f}, F1={train_metrics['f1']:.4f}, LR={train_metrics['learning_rate']:.6f}")
        print(f"  Val:   Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}")

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch + 1)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch + 1)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch + 1)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch + 1)
        writer.add_scalar('F1/train', train_metrics['f1'], epoch + 1)
        writer.add_scalar('F1/val', val_metrics['f1'], epoch + 1)
        writer.add_scalar('LearningRate', train_metrics['learning_rate'], epoch + 1)

        # Save history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['train_lr'].append(train_metrics['learning_rate'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])

        # Save best models
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics, args,
                output_dir / 'best_model_acc.pth'
            )

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics, args,
                output_dir / 'best_model_f1.pth'
            )

        # Save periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics, args,
                output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            )

    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1, val_metrics, args,
        output_dir / 'final_model.pth'
    )

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / 'training_history.csv', index=False)
    logger.info(f"Training history saved to: {output_dir / 'training_history.csv'}")

    # Close TensorBoard writer
    writer.close()

    # Final evaluation on test set
    print(f"\n{'='*80}")
    print("Final Evaluation on Test Set")
    print(f"{'='*80}\n")

    # Load best model
    checkpoint = torch.load(output_dir / 'best_model_acc.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = validate(model, test_loader, criterion, device, args.epochs, 'TEST')

    print(f"\nTest Results:")
    print(f"  Loss:      {test_metrics['loss']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")

    print(f"\nConfusion Matrix:")
    print(test_metrics['confusion_matrix'])

    # Detailed classification report
    print(f"\nPer-class Metrics:")
    print(classification_report(
        test_metrics['labels'],
        test_metrics['predictions'],
        target_names=train_dataset.classes,
        digits=4
    ))

    # Save test results
    test_results = {
        'test_loss': test_metrics['loss'],
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
    }

    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info("Training completed successfully!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Training log saved to: {log_dir / 'training.log'}")
    logger.info(f"TensorBoard logs saved to: {tensorboard_dir}")
    logger.info(f"Training history CSV saved to: {output_dir / 'training_history.csv'}")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Best validation F1: {best_val_f1:.4f}")
    logger.info(f"\nTo view TensorBoard logs, run:")
    logger.info(f"  tensorboard --logdir {tensorboard_dir}")
    logger.info(f"{'='*80}\n")

    print(f"\n{'='*80}")
    print("Training completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"Training log saved to: {log_dir / 'training.log'}")
    print(f"TensorBoard logs saved to: {tensorboard_dir}")
    print(f"Training history CSV saved to: {output_dir / 'training_history.csv'}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"\nTo view TensorBoard logs, run:")
    print(f"  tensorboard --logdir {tensorboard_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fine-tune DINOv3 ViT-L/16 for GZ2 5-class classification'
    )

    # Data
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/shiqiu/dinov3-tng50-finetune',
        help='Base directory containing gz2_5class_train/val/test folders'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/shiqiu/dinov3-tng50-finetune/outputs/gz2_dinov3_lora',
        help='Output directory for checkpoints and logs'
    )

    # Model
    parser.add_argument(
        '--model_name',
        type=str,
        default='facebook/dinov2-large',
        help='HuggingFace model name (use facebook/dinov2-large for ViT-L/16)'
    )
    parser.add_argument(
        '--use_lora',
        action='store_true',
        default=True,
        help='Use LoRA for parameter-efficient fine-tuning'
    )
    parser.add_argument(
        '--lora_r',
        type=int,
        default=16,
        help='LoRA rank'
    )
    parser.add_argument(
        '--lora_alpha',
        type=int,
        default=32,
        help='LoRA alpha'
    )
    parser.add_argument(
        '--lora_dropout',
        type=float,
        default=0.1,
        help='LoRA dropout'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Classifier dropout rate'
    )
    parser.add_argument(
        '--freeze_first_n_layers',
        type=int,
        default=12,
        help='Freeze first N transformer layers (only for full fine-tuning)'
    )

    # Training
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size per GPU'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5e-4,
        help='Peak learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay'
    )
    parser.add_argument(
        '--warmup_ratio',
        type=float,
        default=0.1,
        help='Warmup ratio (fraction of total steps)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=224,
        help='Input image size'
    )

    # System
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--save_freq',
        type=int,
        default=5,
        help='Save checkpoint every N epochs'
    )

    args = parser.parse_args()

    main(args)
