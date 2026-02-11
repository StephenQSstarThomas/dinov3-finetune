#!/usr/bin/env python3
"""
DINOv3 Fine-tuning for TNG50 Hubble Morphology Classification

Model: facebook/dinov3-vit7b16-pretrain-sat493m (or other DINOv3 variants)
Classes: E (Elliptical), S0 (Lenticular), S (Spiral), Irr (Irregular)

Training Strategy: LoRA fine-tuning (parameter-efficient) for 7B model
                   Full fine-tuning option for smaller models

Anti-overfitting features:
  - Layer freezing (default: 18/24 layers frozen)
  - Discriminative layer-wise LR decay
  - Mixup + CutMix regularization
  - Class-weighted loss (critical for 4% Irr class)
  - RandAugment + RandomErasing augmentation
  - EMA model selection
  - Gradient clipping
  - Early stopping on val F1

DINOv3 7B Architecture:
- Parameters: 6,716M
- Embedding dimension: 4096
- Attention heads: 32
- FFN type: SwiGLU
- Position encoding: RoPE

Usage:
    python scripts/tng50/train_tng50_dinov3.py
    python scripts/tng50/train_tng50_dinov3.py --model_name facebook/dinov3-vit7b16-pretrain-sat493m
    python scripts/tng50/train_tng50_dinov3.py --model_name facebook/dinov3-vitl16-pretrain-sat493m
"""

import os
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from PIL import Image
from torchvision import transforms
from transformers import AutoModel

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm
import yaml
import warnings
warnings.filterwarnings('ignore')

import logging
from torch.utils.tensorboard import SummaryWriter


# =============================================================================
# DINOv3 Model Definitions
# =============================================================================

# Model configurations
DINOV3_CONFIGS = {
    # 7B models
    "facebook/dinov3-vit7b16-pretrain-sat493m": {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "is_7b": True,
        "description": "DINOv3 ViT-7B/16 pretrained on SAT-493M satellite imagery"
    },
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "is_7b": True,
        "description": "DINOv3 ViT-7B/16 pretrained on LVD-1689M web images"
    },
    # Large models
    "facebook/dinov3-vitl16-pretrain-sat493m": {
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "is_7b": False,
        "description": "DINOv3 ViT-L/16 distilled from 7B on SAT-493M"
    },
    "facebook/dinov3-vitl16-pretrain-lvd1689m": {
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "is_7b": False,
        "description": "DINOv3 ViT-L/16 pretrained on LVD-1689M"
    },
    # Base/Small models
    "facebook/dinov3-vitb16-pretrain-lvd1689m": {
        "hidden_size": 768,
        "num_attention_heads": 12,
        "is_7b": False,
        "description": "DINOv3 ViT-B/16 pretrained on LVD-1689M"
    },
    "facebook/dinov3-vits16-pretrain-lvd1689m": {
        "hidden_size": 384,
        "num_attention_heads": 6,
        "is_7b": False,
        "description": "DINOv3 ViT-S/16 pretrained on LVD-1689M"
    },
}


class DINOv3Classifier(nn.Module):
    """DINOv3 with classification head for galaxy morphology"""

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16-pretrain-sat493m",
        num_classes: int = 4,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True,
        freeze_backbone: bool = False,
        freeze_first_n_layers: int = 0,
    ):
        super().__init__()

        self.model_name = model_name
        config = DINOV3_CONFIGS.get(model_name, {})

        print(f"Loading DINOv3 model: {model_name}")
        print(f"  Description: {config.get('description', 'Unknown')}")

        # Load model
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Get hidden size from config or model
        if hasattr(self.backbone.config, 'hidden_size'):
            self.hidden_size = self.backbone.config.hidden_size
        else:
            self.hidden_size = config.get("hidden_size", 1024)

        print(f"  Hidden size: {self.hidden_size}")

        # Enable gradient checkpointing for memory efficiency
        if use_gradient_checkpointing and hasattr(self.backbone, 'gradient_checkpointing_enable'):
            print("  Enabling gradient checkpointing")
            self.backbone.gradient_checkpointing_enable()

        # Freeze backbone if requested
        if freeze_backbone:
            print("  Freezing entire backbone")
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_first_n_layers > 0:
            print(f"  Freezing first {freeze_first_n_layers} layers")
            self._freeze_layers(freeze_first_n_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_classes)
        )

        self._print_trainable_parameters()

    def _freeze_layers(self, n_layers: int):
        """Freeze first n transformer layers"""
        if hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
            for i in range(min(n_layers, len(self.backbone.encoder.layer))):
                for param in self.backbone.encoder.layer[i].parameters():
                    param.requires_grad = False

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)

        # Get pooled output (CLS token)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            # Use CLS token from last hidden state
            features = outputs.last_hidden_state[:, 0]

        # Align classifier device with features (critical for multi-GPU)
        classifier_device = next(self.classifier.parameters()).device
        if classifier_device != features.device:
            self.classifier = self.classifier.to(features.device)

        logits = self.classifier(features)
        return logits

    def _print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


class DINOv3WithLoRA(nn.Module):
    """DINOv3 with LoRA for efficient fine-tuning of large models"""

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16-pretrain-sat493m",
        num_classes: int = 4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()

        from peft import LoraConfig, get_peft_model, TaskType

        self.model_name = model_name
        config = DINOV3_CONFIGS.get(model_name, {})

        print(f"Loading DINOv3 model with LoRA: {model_name}")
        print(f"  Description: {config.get('description', 'Unknown')}")

        # Load model
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Get hidden size
        if hasattr(self.backbone.config, 'hidden_size'):
            self.hidden_size = self.backbone.config.hidden_size
        else:
            self.hidden_size = config.get("hidden_size", 1024)

        print(f"  Hidden size: {self.hidden_size}")

        # Apply LoRA
        print(f"  Applying LoRA (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})")

        # DINOv3 module names: q_proj, k_proj, v_proj, o_proj (attention), up_proj, down_proj (mlp)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Attention layers only

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        try:
            self.backbone = get_peft_model(self.backbone, lora_config)
            print("  LoRA applied successfully")
        except Exception as e:
            print(f"  Warning: Could not apply LoRA ({e}), using frozen backbone instead")
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Enable gradient checkpointing
        if use_gradient_checkpointing:
            if hasattr(self.backbone, 'gradient_checkpointing_enable'):
                self.backbone.gradient_checkpointing_enable()
                print("  Gradient checkpointing enabled")

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_classes)
        )

        self._print_trainable_parameters()

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)

        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0]

        # Align classifier device with features (critical for multi-GPU)
        classifier_device = next(self.classifier.parameters()).device
        if classifier_device != features.device:
            self.classifier = self.classifier.to(features.device)

        logits = self.classifier(features)
        return logits

    def _print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


# Backward compatibility aliases (evaluate script imports these names)
DINOv2Classifier = DINOv3Classifier
DINOv2WithLoRA = DINOv3WithLoRA


# =============================================================================
# Dataset
# =============================================================================

class TNG50HubbleDataset(Dataset):
    """TNG50 Hubble morphology dataset"""

    CLASSES = ['E', 'S0', 'S', 'Irr']

    def __init__(self, data_dir: str, split: str = 'train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(self.CLASSES)}
        self.idx_to_class = {i: c for i, c in enumerate(self.CLASSES)}

        self.samples = []
        split_dir = self.data_dir / split

        for class_name in self.CLASSES:
            class_dir = split_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.png'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))

        print(f"{split.upper()}: {len(self.samples)} images")
        labels = [s[1] for s in self.samples]
        for cls_name, cls_idx in self.class_to_idx.items():
            count = labels.count(cls_idx)
            print(f"  {cls_name:4s}: {count:5d}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def classes(self):
        return self.CLASSES

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights, normalized so mean weight = 1"""
        labels = [s[1] for s in self.samples]
        counts = np.bincount(labels, minlength=len(self.CLASSES)).astype(np.float64)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(self.CLASSES)
        return torch.FloatTensor(weights)


def get_transforms(
    image_size: int = 224,
    split: str = 'train',
    use_randaugment: bool = False,
    use_random_erasing: bool = False,
):
    """Get data transforms for DINOv3.

    Args:
        image_size: Target image size.
        split: Dataset split ('train', 'val', 'test').
        use_randaugment: Enable RandAugment + stronger color jitter (default False for backward compat).
        use_random_erasing: Enable RandomErasing after normalization (default False for backward compat).
    """
    # DINOv3 uses ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == 'train':
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
        ]

        if use_randaugment:
            transform_list.extend([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3
                ),
                transforms.RandAugment(num_ops=2, magnitude=9),
            ])
        else:
            transform_list.append(
                transforms.ColorJitter(brightness=0.1, contrast=0.1)
            )

        transform_list.extend([
            transforms.ToTensor(),
            normalize,
        ])

        if use_random_erasing:
            transform_list.append(
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))
            )

        return transforms.Compose(transform_list)
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


# =============================================================================
# Loss, EMA, and Regularization
# =============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing and optional class weights"""

    def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = torch.log_softmax(pred, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, target, weight=self.weight)
        return self.smoothing * loss / n_classes + (1 - self.smoothing) * nll


class MixupCutmixLoss(nn.Module):
    """Unified loss handling both hard labels and soft labels from Mixup/CutMix.

    Dispatches on target dimensionality:
      - target.dim() == 1  →  standard CE with label smoothing + class weights
      - target.dim() == 2  →  soft-label CE (from mixup/cutmix) with class weights
    """

    def __init__(self, weight: Optional[torch.Tensor] = None, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer('weight', weight)

    def forward(self, pred, target):
        log_preds = torch.log_softmax(pred, dim=-1)

        if target.dim() == 1:
            # Hard labels
            nll = F.nll_loss(log_preds, target, weight=self.weight)
            if self.smoothing > 0:
                smooth_loss = -log_preds.mean(dim=-1).mean()
                return (1 - self.smoothing) * nll + self.smoothing * smooth_loss
            return nll
        else:
            # Soft labels [B, C] from mixup/cutmix
            if self.smoothing > 0:
                n_classes = pred.size(-1)
                target = (1 - self.smoothing) * target + self.smoothing / n_classes

            if self.weight is not None:
                loss = -(target * log_preds * self.weight.unsqueeze(0)).sum(dim=-1).mean()
            else:
                loss = -(target * log_preds).sum(dim=-1).mean()
            return loss


class ModelEMA:
    """Exponential Moving Average of model weights.

    Tracks only requires_grad parameters. Provides apply_shadow / restore
    pattern so that EMA weights can be temporarily loaded for validation
    and then reverted for continued training.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module):
        """Copy EMA weights into model, saving originals for later restore."""
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original (online) weights from backup."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class EarlyStopping:
    """Stop training when a monitored metric has stopped improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


# =============================================================================
# Mixup / CutMix
# =============================================================================

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
    num_classes: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Mixup: linear interpolation of random image pairs.

    Returns mixed images and soft label tensor [B, num_classes].
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]

    y_onehot = torch.zeros(batch_size, num_classes, device=x.device)
    y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot[index]
    return mixed_x, mixed_y


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    num_classes: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply CutMix: paste a random rectangular patch from one image onto another.

    Returns cutmixed images and soft label tensor [B, num_classes].
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape
    cut_ratio = np.sqrt(1 - lam)
    cut_w = max(1, int(W * cut_ratio))
    cut_h = max(1, int(H * cut_ratio))

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Adjust lambda based on actual pasted area
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

    y_onehot = torch.zeros(batch_size, num_classes, device=x.device)
    y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot[index]
    return mixed_x, mixed_y


def apply_mixup_cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
    num_classes: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly apply either Mixup or CutMix with 50/50 probability."""
    if random.random() > 0.5:
        return mixup_data(x, y, alpha, num_classes)
    else:
        return cutmix_data(x, y, alpha, num_classes)


# =============================================================================
# Optimizer with Discriminative LR
# =============================================================================

def build_optimizer_param_groups(
    model: nn.Module,
    base_lr: float,
    lr_decay: float = 0.85,
    classifier_lr_mult: float = 5.0,
    weight_decay: float = 0.05,
) -> List[Dict]:
    """Build param groups with exponential layer-wise LR decay.

    layer_i_lr = base_lr * decay^(num_layers - 1 - i)
    classifier_lr = base_lr * classifier_lr_mult
    """
    base_model = model.module if hasattr(model, 'module') else model

    # Try to find backbone transformer layers
    backbone = base_model.backbone
    # PEFT models may wrap the backbone
    if hasattr(backbone, 'base_model'):
        backbone_inner = backbone.base_model
    else:
        backbone_inner = backbone

    layers = None
    if hasattr(backbone_inner, 'encoder') and hasattr(backbone_inner.encoder, 'layer'):
        layers = backbone_inner.encoder.layer

    if layers is None:
        # Fallback: single param group for all trainable params
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            return [{'params': [torch.zeros(1, requires_grad=True)], 'lr': base_lr, 'weight_decay': weight_decay}]
        return [{'params': trainable, 'lr': base_lr, 'weight_decay': weight_decay}]

    num_layers = len(layers)
    param_groups = []
    assigned_ids = set()

    # Backbone layers with exponential lr decay (only unfrozen params)
    for i in range(num_layers):
        params = [p for p in layers[i].parameters() if p.requires_grad]
        if params:
            layer_lr = base_lr * (lr_decay ** (num_layers - 1 - i))
            param_groups.append({
                'params': params,
                'lr': layer_lr,
                'weight_decay': weight_decay,
            })
            for p in params:
                assigned_ids.add(id(p))

    # Classifier head
    classifier_params = [p for p in base_model.classifier.parameters() if p.requires_grad]
    if classifier_params:
        param_groups.append({
            'params': classifier_params,
            'lr': base_lr * classifier_lr_mult,
            'weight_decay': weight_decay,
        })
        for p in classifier_params:
            assigned_ids.add(id(p))

    # Remaining trainable params (embeddings, layernorms, etc.)
    remaining = [p for p in base_model.parameters() if p.requires_grad and id(p) not in assigned_ids]
    if remaining:
        # Assign smallest backbone lr to remaining params
        remaining_lr = base_lr * (lr_decay ** (num_layers - 1))
        param_groups.append({
            'params': remaining,
            'lr': remaining_lr,
            'weight_decay': weight_decay,
        })

    return param_groups


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    ema: Optional[ModelEMA] = None,
    logger: Optional[logging.Logger] = None,
    writer: Optional[SummaryWriter] = None,
    global_step: int = 0,
    use_amp: bool = True,
    max_grad_norm: float = 0.0,
    use_mixup_cutmix: bool = False,
    mixup_alpha: float = 0.4,
    num_classes: int = 4,
) -> Tuple[Dict[str, float], int]:
    """Train for one epoch"""
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    batch_lrs = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [TRAIN]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        # Save original labels for metrics (before mixup/cutmix)
        original_labels = labels.clone()

        # Apply Mixup / CutMix
        if use_mixup_cutmix:
            images, labels = apply_mixup_cutmix(images, labels, alpha=mixup_alpha, num_classes=num_classes)

        optimizer.zero_grad()

        if use_amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            base_model = model.module if hasattr(model, 'module') else model
            ema.update(base_model)

        running_loss += loss.item()
        batch_lrs.append(current_lr)

        # Use original hard labels for accuracy computation
        preds = torch.argmax(logits.float(), dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(original_labels.cpu().numpy())

        pbar.set_postfix({'loss': running_loss / (batch_idx + 1), 'lr': f'{current_lr:.2e}'})

        if writer is not None and batch_idx % 10 == 0:
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            writer.add_scalar('Train/LearningRate', current_lr, global_step)

        global_step += 1

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    if logger:
        logger.info(f"Epoch {epoch} - Train Loss: {running_loss / len(dataloader):.4f}, "
                   f"Acc: {accuracy:.4f}, F1: {f1:.4f}")

    return {
        'loss': running_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'learning_rate': batch_lrs[-1] if batch_lrs else 0,
    }, global_step


@torch.no_grad()
def predict_with_tta(
    model: nn.Module,
    images: torch.Tensor,
    num_augments: int = 4,
) -> torch.Tensor:
    """Test-Time Augmentation: average logits over geometric augmentations.

    Augmentations: original + random flips + 90-degree rotations.
    """
    all_logits = [model(images)]

    for _ in range(num_augments - 1):
        aug = images.clone()
        if random.random() > 0.5:
            aug = torch.flip(aug, dims=[3])  # horizontal flip
        if random.random() > 0.5:
            aug = torch.flip(aug, dims=[2])  # vertical flip
        k = random.randint(0, 3)
        if k > 0:
            aug = torch.rot90(aug, k, dims=[2, 3])
        all_logits.append(model(aug))

    return torch.stack(all_logits, dim=0).mean(dim=0)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    split: str = 'VAL',
    logger: Optional[logging.Logger] = None,
    use_amp: bool = True,
    use_tta: bool = False,
    num_tta: int = 4,
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

        if use_amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                if use_tta:
                    logits = predict_with_tta(model, images, num_augments=num_tta)
                else:
                    logits = model(images)
                loss = criterion(logits, labels)
        else:
            if use_tta:
                logits = predict_with_tta(model, images, num_augments=num_tta)
            else:
                logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item()
        preds = torch.argmax(logits.float(), dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

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


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, args, filename, ema=None):
    """Save model checkpoint.

    Saves unwrapped model state (no 'module.' prefix from DDP).
    Optionally includes EMA shadow weights for training resumption.
    """
    model_to_save = model.module if hasattr(model, 'module') else model
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'args': vars(args) if hasattr(args, '__dict__') else args,
    }
    if ema is not None:
        checkpoint['ema_shadow'] = ema.shadow
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")


# =============================================================================
# Main Training Loop
# =============================================================================

def load_config(config_path):
    """Load configuration from YAML file"""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main(args):
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_distributed = world_size > 1
    is_main_process = rank == 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Determine device
    if torch.cuda.is_available():
        device = torch.device('cuda', local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
        if is_main_process:
            print("WARNING: CUDA not available, using CPU")

    output_dir = Path(args.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / 'logs'
    if is_main_process:
        log_dir.mkdir(exist_ok=True)

    # Only setup logging on main process
    logger = None
    writer = None
    if is_main_process:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)

        tensorboard_dir = output_dir / 'tensorboard'
        tensorboard_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(tensorboard_dir)

        logger.info(f"\n{'='*80}")
        logger.info("TNG50 Hubble Morphology Classification - DINOv3 Training")
        logger.info(f"{'='*80}")
        logger.info(f"Device: {device}")
        logger.info(f"Distributed: {is_distributed}, World Size: {world_size}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(local_rank)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Strategy: {'LoRA' if args.use_lora else 'Full/Frozen Fine-tuning'}")
        logger.info("Classes: E, S0, S, Irr (4 classes)")
        logger.info(f"Freeze first {args.freeze_first_n_layers} layers")
        logger.info(f"LR: {args.learning_rate}, LR decay: {args.lr_decay}, Classifier LR mult: {args.classifier_lr_mult}")
        logger.info(f"Dropout: {args.dropout}, Weight decay: {args.weight_decay}")
        logger.info(f"EMA: {args.use_ema}, Mixup/CutMix: {args.use_mixup_cutmix}, RandAugment: {args.use_randaugment}")
        logger.info(f"Class weights: {args.use_class_weights}, Grad clip: {args.max_grad_norm}")
        logger.info(f"Early stopping patience: {args.early_stopping_patience}")
        logger.info(f"{'='*80}\n")

        with open(output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2)

    # Load datasets
    if is_main_process:
        print("Loading datasets...")
    train_dataset = TNG50HubbleDataset(
        args.data_dir, split='train',
        transform=get_transforms(
            args.image_size, 'train',
            use_randaugment=args.use_randaugment,
            use_random_erasing=args.use_random_erasing,
        )
    )
    val_dataset = TNG50HubbleDataset(
        args.data_dir, split='val',
        transform=get_transforms(args.image_size, 'val')
    )
    test_dataset = TNG50HubbleDataset(
        args.data_dir, split='test',
        transform=get_transforms(args.image_size, 'test')
    )

    # Compute class weights
    class_weights = None
    if args.use_class_weights:
        class_weights = train_dataset.get_class_weights().to(device)
        if is_main_process:
            cw_str = ", ".join(f"{c}={w:.3f}" for c, w in zip(TNG50HubbleDataset.CLASSES, class_weights))
            print(f"Class weights: {cw_str}")

    # Use DistributedSampler for DDP
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Create model
    if is_main_process:
        print("\nCreating model...")

    if args.use_lora:
        model = DINOv3WithLoRA(
            model_name=args.model_name,
            num_classes=4,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            dropout=args.dropout,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    else:
        model = DINOv3Classifier(
            model_name=args.model_name,
            num_classes=4,
            dropout=args.dropout,
            use_gradient_checkpointing=args.gradient_checkpointing,
            freeze_backbone=args.freeze_backbone,
            freeze_first_n_layers=args.freeze_first_n_layers,
        )

    # Move model to device
    model = model.to(device)

    # Wrap model with DDP for distributed training
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Loss function
    if args.use_mixup_cutmix or args.use_class_weights:
        criterion = MixupCutmixLoss(weight=class_weights, smoothing=args.label_smoothing)
    elif args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    if is_main_process:
        print(f"Loss: {criterion.__class__.__name__}")

    # Optimizer with discriminative layer-wise lr
    param_groups = build_optimizer_param_groups(
        model,
        base_lr=args.learning_rate,
        lr_decay=args.lr_decay,
        classifier_lr_mult=args.classifier_lr_mult,
        weight_decay=args.weight_decay,
    )

    optimizer = optim.AdamW(param_groups)

    if is_main_process:
        print(f"\nOptimizer param groups ({len(param_groups)}):")
        for i, group in enumerate(param_groups):
            n_params = sum(p.numel() for p in group['params'])
            print(f"  Group {i}: lr={group['lr']:.2e}, wd={group.get('weight_decay', 0):.4f}, params={n_params:,}")

    # Scheduler (OneCycleLR with per-group max_lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    max_lrs = [group['lr'] for group in param_groups]

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps if total_steps > 0 else 0.1,
        anneal_strategy='cos',
    )

    # Mixed precision scaler
    scaler = GradScaler()
    use_amp = args.use_amp and torch.cuda.is_available()

    # EMA (create from unwrapped model to avoid 'module.' prefix)
    base_model = model.module if hasattr(model, 'module') else model
    ema = ModelEMA(base_model, decay=args.ema_decay) if args.use_ema else None
    if ema and is_main_process:
        n_tracked = len(ema.shadow)
        print(f"EMA tracking {n_tracked} parameter tensors (decay={args.ema_decay})")

    # Early stopping
    early_stopper = EarlyStopping(patience=args.early_stopping_patience, mode='max') \
        if args.early_stopping_patience > 0 else None

    # Training history
    history = {
        'epoch': [],
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
    }

    best_val_acc = 0.0
    best_val_f1 = 0.0
    global_step = 0

    if is_main_process and logger:
        logger.info(f"\n{'='*80}")
        logger.info("Starting Training")
        logger.info(f"{'='*80}\n")

    for epoch in range(args.epochs):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_metrics, global_step = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device,
            epoch + 1, ema, logger, writer, global_step, use_amp,
            max_grad_norm=args.max_grad_norm,
            use_mixup_cutmix=args.use_mixup_cutmix,
            mixup_alpha=args.mixup_alpha,
            num_classes=4,
        )

        # Validate with EMA weights if available
        if ema is not None:
            base_model = model.module if hasattr(model, 'module') else model
            ema.apply_shadow(base_model)

        val_metrics = validate(model, val_loader, criterion, device, epoch + 1, 'VAL', logger, use_amp)

        if is_main_process:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f}, F1={train_metrics['f1']:.4f}")
            print(f"  Val:   Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}")

            # TensorBoard logging
            if writer is not None:
                writer.add_scalar('Loss/train', train_metrics['loss'], epoch + 1)
                writer.add_scalar('Loss/val', val_metrics['loss'], epoch + 1)
                writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch + 1)
                writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch + 1)
                writer.add_scalar('F1/train', train_metrics['f1'], epoch + 1)
                writer.add_scalar('F1/val', val_metrics['f1'], epoch + 1)

            # Update history
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['train_f1'].append(train_metrics['f1'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])

            # Save best models (EMA weights applied if ema is active)
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, args,
                              output_dir / 'best_model_acc.pth', ema=ema)

            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, args,
                              output_dir / 'best_model_f1.pth', ema=ema)

        # Restore online weights for continued training
        if ema is not None:
            base_model = model.module if hasattr(model, 'module') else model
            ema.restore(base_model)

        # Periodic checkpoint (online weights + ema shadow)
        if is_main_process and (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, args,
                          output_dir / f'checkpoint_epoch_{epoch+1}.pth', ema=ema)

        # Early stopping
        if early_stopper is not None:
            if early_stopper(val_metrics['f1']):
                if is_main_process:
                    msg = (f"Early stopping triggered at epoch {epoch+1} "
                           f"(patience={args.early_stopping_patience}, best_f1={early_stopper.best_score:.4f})")
                    print(f"\n{msg}")
                    if logger:
                        logger.info(msg)
                break

    # Only main process saves final results
    if is_main_process:
        # Save final model
        save_checkpoint(model, optimizer, scheduler, args.epochs - 1, val_metrics, args,
                       output_dir / 'final_model.pth', ema=ema)

        # Save training history
        history_df = pd.DataFrame(history)
        history_df.to_csv(output_dir / 'training_history.csv', index=False)

        if writer is not None:
            writer.close()

        # Final evaluation on test set
        print(f"\n{'='*80}")
        print("Final Evaluation on Test Set")
        print(f"{'='*80}\n")

        checkpoint = torch.load(output_dir / 'best_model_acc.pth', weights_only=False)
        model_to_load = model.module if is_distributed else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

        test_metrics = validate(
            model, test_loader, criterion, device, args.epochs, 'TEST',
            use_amp=use_amp, use_tta=args.use_tta, num_tta=args.num_tta,
        )

        print(f"\nTest Results:")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  F1 Score:  {test_metrics['f1']:.4f}")

        print("\nConfusion Matrix:")
        print(test_metrics['confusion_matrix'])

        print("\nPer-class Metrics:")
        print(classification_report(
            test_metrics['labels'],
            test_metrics['predictions'],
            target_names=train_dataset.classes,
            digits=4
        ))

        # Save test results
        test_results = {
            'test_accuracy': float(test_metrics['accuracy']),
            'test_precision': float(test_metrics['precision']),
            'test_recall': float(test_metrics['recall']),
            'test_f1': float(test_metrics['f1']),
            'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
        }

        with open(output_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)

        if logger:
            logger.info(f"\n{'='*80}")
            logger.info("Training completed!")
            logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
            logger.info(f"Best validation F1: {best_val_f1:.4f}")
            logger.info(f"Results saved to: {output_dir}")
            logger.info(f"{'='*80}\n")

    # Cleanup distributed training
    cleanup_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fine-tune DINOv3 for TNG50 Hubble morphology classification'
    )

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data/tng50/splits')
    parser.add_argument('--output_dir', type=str, default='./outputs/tng50_dinov3')
    parser.add_argument('--config', type=str, default='./configs/tng50_hubble.yaml')

    # Model arguments - Default to ViT-L (300M) for DDP training
    parser.add_argument('--model_name', type=str,
                        default='facebook/dinov3-vitl16-pretrain-sat493m',
                        help='DINOv3 model name from HuggingFace (default: ViT-L 300M)')
    parser.add_argument('--use_lora', action='store_true', default=False,
                        help='Use LoRA for parameter-efficient fine-tuning (not needed for ViT-L)')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Classifier dropout (default: 0.3, was 0.1)')
    parser.add_argument('--freeze_backbone', action='store_true', default=False)
    parser.add_argument('--freeze_first_n_layers', type=int, default=18,
                        help='Freeze first N backbone layers (default: 18 of 24)')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True)
    parser.add_argument('--no_gradient_checkpointing', dest='gradient_checkpointing', action='store_false')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU (can use larger value for ViT-L)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Base learning rate (default: 5e-5, was 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay (default: 0.05, was 0.01)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # EMA
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use EMA (default: True)')
    parser.add_argument('--no_ema', dest='use_ema', action='store_false')
    parser.add_argument('--ema_decay', type=float, default=0.999)

    # Mixed precision
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false')

    # Discriminative LR
    parser.add_argument('--lr_decay', type=float, default=0.85,
                        help='Layer-wise LR exponential decay factor (default: 0.85)')
    parser.add_argument('--classifier_lr_mult', type=float, default=5.0,
                        help='Classifier head LR multiplier (default: 5.0)')

    # Regularization: class weights
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='Use inverse-frequency class weights in loss (default: True)')
    parser.add_argument('--no_class_weights', dest='use_class_weights', action='store_false')

    # Regularization: Mixup + CutMix
    parser.add_argument('--use_mixup_cutmix', action='store_true', default=True,
                        help='Apply Mixup/CutMix regularization (default: True)')
    parser.add_argument('--no_mixup_cutmix', dest='use_mixup_cutmix', action='store_false')
    parser.add_argument('--mixup_alpha', type=float, default=0.4,
                        help='Mixup/CutMix Beta distribution alpha (default: 0.4)')

    # Regularization: augmentation
    parser.add_argument('--use_randaugment', action='store_true', default=True,
                        help='Use RandAugment + stronger color jitter (default: True)')
    parser.add_argument('--no_randaugment', dest='use_randaugment', action='store_false')
    parser.add_argument('--use_random_erasing', action='store_true', default=True,
                        help='Use RandomErasing augmentation (default: True)')
    parser.add_argument('--no_random_erasing', dest='use_random_erasing', action='store_false')

    # Gradient clipping
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping (0 = disabled, default: 1.0)')

    # Early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience on val F1 (0 = disabled, default: 10)')

    # Test-Time Augmentation
    parser.add_argument('--use_tta', action='store_true', default=False,
                        help='Use TTA for final test evaluation')
    parser.add_argument('--num_tta', type=int, default=4,
                        help='Number of TTA augmentations (default: 4)')

    # System arguments
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_freq', type=int, default=5)

    args = parser.parse_args()
    main(args)
