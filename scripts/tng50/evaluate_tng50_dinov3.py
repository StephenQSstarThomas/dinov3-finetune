#!/usr/bin/env python3
"""
Evaluation script for trained TNG50 Hubble morphology models.

Features:
- Overall metrics (accuracy, precision, recall, F1)
- Per-class metrics
- Per-redshift metrics
- Confusion matrix visualization
- Detailed classification report

Usage:
    python scripts/tng50/evaluate_tng50_dinov3.py --checkpoint outputs/tng50_hubble/best_model_acc.pth
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

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_tng50_dinov3 import DINOv2WithLoRA, DINOv2Classifier, TNG50HubbleDataset, get_transforms


def plot_confusion_matrix(cm, class_names, save_path, normalize=True, title='Confusion Matrix'):
    """Plot and save confusion matrix"""
    if normalize:
        cm_plot = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        cm_plot = cm
        fmt = 'd'

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_per_class_metrics(labels, predictions, class_names, save_path):
    """Plot per-class precision, recall, F1"""
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8, color='#2ecc71')
    ax.bar(x, recall, width, label='Recall', alpha=0.8, color='#3498db')
    ax.bar(x + width, f1, width, label='F1 Score', alpha=0.8, color='#e74c3c')

    ax.set_xlabel('Hubble Type', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics for Hubble Morphology Classification', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(i - width, p + 0.02, f'{p:.2f}', ha='center', fontsize=8)
        ax.text(i, r + 0.02, f'{r:.2f}', ha='center', fontsize=8)
        ax.text(i + width, f + 0.02, f'{f:.2f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics plot saved to {save_path}")


@torch.no_grad()
def evaluate_model(model, dataloader, device, class_names):
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

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds)

    per_class_precision, per_class_recall, per_class_f1, per_class_support = \
        precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)

    results = {
        'overall': {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision),
            'recall_weighted': float(recall),
            'f1_weighted': float(f1),
            'precision_macro': float(macro_precision),
            'recall_macro': float(macro_recall),
            'f1_macro': float(macro_f1),
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


def analyze_confusion_patterns(cm, class_names):
    """Analyze key confusion patterns"""
    print("\n" + "="*60)
    print("Confusion Pattern Analysis")
    print("="*60)

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    key_pairs = [
        ('E', 'S0'),   # E <-> S0 confusion
        ('S0', 'S'),   # S0 <-> S confusion
        ('S', 'Irr'),  # S <-> Irr confusion
    ]

    for class1, class2 in key_pairs:
        if class1 in class_names and class2 in class_names:
            i1 = class_names.index(class1)
            i2 = class_names.index(class2)

            conf_1_to_2 = cm_norm[i1, i2] * 100
            conf_2_to_1 = cm_norm[i2, i1] * 100

            print(f"\n{class1} <-> {class2}:")
            print(f"  {class1} misclassified as {class2}: {conf_1_to_2:.1f}%")
            print(f"  {class2} misclassified as {class1}: {conf_2_to_1:.1f}%")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_args = argparse.Namespace(**checkpoint['args'])

    print("\nLoading dataset...")
    dataset = TNG50HubbleDataset(
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

    print("\nCreating model...")
    if checkpoint_args.use_lora:
        model = DINOv2WithLoRA(
            model_name=checkpoint_args.model_name,
            num_classes=4,
            lora_r=checkpoint_args.lora_r,
            lora_alpha=checkpoint_args.lora_alpha,
            lora_dropout=checkpoint_args.lora_dropout,
            freeze_backbone=True,
            dropout=checkpoint_args.dropout,
        )
    else:
        model = DINOv2Classifier(
            model_name=checkpoint_args.model_name,
            num_classes=4,
            dropout=checkpoint_args.dropout,
            freeze_first_n_layers=checkpoint_args.freeze_first_n_layers,
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print("\n" + "="*80)
    print(f"Evaluating on {args.split.upper()} set")
    print("="*80 + "\n")

    results = evaluate_model(model, dataloader, device, dataset.classes)

    print("\nOverall Metrics:")
    print(f"  {'Accuracy':15s}: {results['overall']['accuracy']:.4f}")
    print(f"  {'Precision (W)':15s}: {results['overall']['precision_weighted']:.4f}")
    print(f"  {'Recall (W)':15s}: {results['overall']['recall_weighted']:.4f}")
    print(f"  {'F1 (W)':15s}: {results['overall']['f1_weighted']:.4f}")
    print(f"  {'Precision (M)':15s}: {results['overall']['precision_macro']:.4f}")
    print(f"  {'Recall (M)':15s}: {results['overall']['recall_macro']:.4f}")
    print(f"  {'F1 (M)':15s}: {results['overall']['f1_macro']:.4f}")

    print("\nPer-Class Metrics:")
    for class_name, metrics in results['per_class'].items():
        print(f"\n  {class_name}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1:        {metrics['f1']:.4f}")
        print(f"    Support:   {metrics['support']}")

    print("\nConfusion Matrix:")
    cm = np.array(results['confusion_matrix'])
    print(cm)

    print("\nDetailed Classification Report:")
    print(classification_report(
        results['labels'],
        results['predictions'],
        target_names=dataset.classes,
        digits=4
    ))

    analyze_confusion_patterns(cm, dataset.classes)

    results_file = output_dir / f'{args.split}_results.json'
    save_results = {
        'overall': results['overall'],
        'per_class': results['per_class'],
        'confusion_matrix': results['confusion_matrix'],
    }
    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    cm_path = output_dir / f'{args.split}_confusion_matrix.png'
    plot_confusion_matrix(cm, dataset.classes, cm_path, normalize=True,
                         title=f'TNG50 Hubble Classification - {args.split.upper()} Set')

    cm_raw_path = output_dir / f'{args.split}_confusion_matrix_raw.png'
    plot_confusion_matrix(cm, dataset.classes, cm_raw_path, normalize=False,
                         title=f'TNG50 Hubble Classification - {args.split.upper()} Set (Counts)')

    metrics_path = output_dir / f'{args.split}_per_class_metrics.png'
    plot_per_class_metrics(
        np.array(results['labels']),
        np.array(results['predictions']),
        dataset.classes,
        metrics_path
    )

    predictions_df = pd.DataFrame({
        'true_label': [dataset.classes[i] for i in results['labels']],
        'predicted_label': [dataset.classes[i] for i in results['predictions']],
        'true_label_idx': results['labels'],
        'predicted_label_idx': results['predictions'],
    })

    for i, class_name in enumerate(dataset.classes):
        predictions_df[f'prob_{class_name}'] = [p[i] for p in results['probabilities']]

    predictions_file = output_dir / f'{args.split}_predictions.csv'
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Predictions saved to {predictions_file}")

    print("\n" + "="*80)
    print("Evaluation completed!")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained TNG50 Hubble model')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data/tng50/splits',
                       help='Base data directory')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results/tng50',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')

    args = parser.parse_args()
    main(args)
