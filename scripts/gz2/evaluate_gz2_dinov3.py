#!/usr/bin/env python3
"""
Evaluation script for trained DINOv3 GZ2 models

Usage:
    python scripts/evaluate_gz2_dinov3.py --checkpoint outputs/gz2_dinov3_lora/best_model_acc.pth
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import from train script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_gz2_dinov3 import DINOv3WithLoRA, DINOv3Classifier, GZ2Dataset, get_transforms


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: Path,
    normalize: bool = True,
):
    """Plot and save confusion matrix"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_per_class_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list,
    save_path: Path,
):
    """Plot per-class precision, recall, F1"""
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1 Score', alpha=0.8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics plot saved to {save_path}")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list,
) -> dict:
    """Comprehensive model evaluation"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    print("Running evaluation...")
    for images, labels in tqdm(dataloader):
        images = images.to(device)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = \
        precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)

    results = {
        'overall': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        },
        'per_class': {},
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probs.tolist(),
    }

    for i, class_name in enumerate(class_names):
        results['per_class'][class_name] = {
            'precision': float(per_class_precision[i]),
            'recall': float(per_class_recall[i]),
            'f1': float(per_class_f1[i]),
            'support': int(per_class_support[i]),
        }

    return results


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_args = argparse.Namespace(**checkpoint['args'])

    # Load dataset
    print("\nLoading dataset...")
    dataset = GZ2Dataset(
        args.data_dir,
        split=args.split,
        transform=get_transforms(checkpoint_args.image_size, 'test')
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model
    print("\nCreating model...")
    if checkpoint_args.use_lora:
        model = DINOv3WithLoRA(
            model_name=checkpoint_args.model_name,
            num_classes=5,
            lora_r=checkpoint_args.lora_r,
            lora_alpha=checkpoint_args.lora_alpha,
            lora_dropout=checkpoint_args.lora_dropout,
            freeze_backbone=True,
            dropout=checkpoint_args.dropout,
        )
    else:
        model = DINOv3Classifier(
            model_name=checkpoint_args.model_name,
            num_classes=5,
            dropout=checkpoint_args.dropout,
            freeze_first_n_layers=checkpoint_args.freeze_first_n_layers,
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Evaluate
    print("\n" + "="*80)
    print(f"Evaluating on {args.split.upper()} set")
    print("="*80 + "\n")

    results = evaluate_model(model, dataloader, device, dataset.classes)

    # Print results
    print("\nOverall Metrics:")
    for metric, value in results['overall'].items():
        print(f"  {metric.capitalize():12s}: {value:.4f}")

    print("\nPer-Class Metrics:")
    for class_name, metrics in results['per_class'].items():
        print(f"\n{class_name}:")
        for metric, value in metrics.items():
            if metric == 'support':
                print(f"  {metric:10s}: {value}")
            else:
                print(f"  {metric:10s}: {value:.4f}")

    print("\nConfusion Matrix:")
    cm = np.array(results['confusion_matrix'])
    print(cm)

    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(
        results['labels'],
        results['predictions'],
        target_names=dataset.classes,
        digits=4
    ))

    # Save results
    results_file = output_dir / f'{args.split}_results.json'
    with open(results_file, 'w') as f:
        # Remove large arrays before saving
        save_results = {
            'overall': results['overall'],
            'per_class': results['per_class'],
            'confusion_matrix': results['confusion_matrix'],
        }
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Plot confusion matrix
    cm_path = output_dir / f'{args.split}_confusion_matrix.png'
    plot_confusion_matrix(cm, dataset.classes, cm_path, normalize=True)

    cm_raw_path = output_dir / f'{args.split}_confusion_matrix_raw.png'
    plot_confusion_matrix(cm, dataset.classes, cm_raw_path, normalize=False)

    # Plot per-class metrics
    metrics_path = output_dir / f'{args.split}_per_class_metrics.png'
    plot_per_class_metrics(
        np.array(results['labels']),
        np.array(results['predictions']),
        dataset.classes,
        metrics_path
    )

    # Save predictions
    predictions_df = pd.DataFrame({
        'true_label': [dataset.classes[i] for i in results['labels']],
        'predicted_label': [dataset.classes[i] for i in results['predictions']],
        'true_label_idx': results['labels'],
        'predicted_label_idx': results['predictions'],
    })

    # Add probability columns
    for i, class_name in enumerate(dataset.classes):
        predictions_df[f'prob_{class_name}'] = [p[i] for p in results['probabilities']]

    predictions_file = output_dir / f'{args.split}_predictions.csv'
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Predictions saved to {predictions_file}")

    print("\n" + "="*80)
    print("Evaluation completed!")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained GZ2 DINOv3 model')

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/shiqiu/dinov3-tng50-finetune',
        help='Base data directory'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./evaluation_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of data loading workers'
    )

    args = parser.parse_args()
    main(args)
