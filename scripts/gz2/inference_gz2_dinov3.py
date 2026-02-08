#!/usr/bin/env python3
"""
Inference script for GZ2 DINOv3 model

Usage:
    # Single image
    python scripts/inference_gz2_dinov3.py --checkpoint outputs/best_model_acc.pth --image path/to/galaxy.jpg

    # Batch inference
    python scripts/inference_gz2_dinov3.py --checkpoint outputs/best_model_acc.pth --image_dir path/to/images/ --output predictions.csv
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import pandas as pd

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import from train script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_gz2_dinov3 import DINOv3WithLoRA, DINOv3Classifier


CLASS_NAMES = [
    'cigar_shaped_smooth',
    'completely_round_smooth',
    'edge_on',
    'in_between_smooth',
    'spiral'
]


def get_inference_transform(image_size: int = 224):
    """Get transform for inference"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    """Load trained model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = argparse.Namespace(**checkpoint['args'])

    # Create model
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
            freeze_first_n_layers=getattr(args, 'freeze_first_n_layers', 0),
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, args


@torch.no_grad()
def predict_single_image(
    model: nn.Module,
    image_path: str,
    transform,
    device: torch.device,
) -> dict:
    """Predict class for a single image"""
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Transform
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    logits = model(image_tensor)
    probs = torch.softmax(logits, dim=1)[0]
    pred_class = logits.argmax(dim=1).item()

    result = {
        'predicted_class': CLASS_NAMES[pred_class],
        'predicted_class_idx': pred_class,
        'confidence': float(probs[pred_class]),
        'probabilities': {
            class_name: float(probs[i])
            for i, class_name in enumerate(CLASS_NAMES)
        }
    }

    return result


@torch.no_grad()
def predict_batch(
    model: nn.Module,
    image_paths: List[str],
    transform,
    device: torch.device,
    batch_size: int = 32,
) -> List[dict]:
    """Predict classes for multiple images"""
    results = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []

        # Load and transform images
        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image)
                batch_images.append(image_tensor)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

        if not batch_images:
            continue

        # Stack into batch
        batch_tensor = torch.stack(batch_images).to(device)

        # Predict
        logits = model(batch_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_classes = logits.argmax(dim=1)

        # Collect results
        for j, img_path in enumerate(batch_paths):
            result = {
                'image_path': img_path,
                'image_name': Path(img_path).name,
                'predicted_class': CLASS_NAMES[pred_classes[j]],
                'predicted_class_idx': int(pred_classes[j]),
                'confidence': float(probs[j, pred_classes[j]]),
            }

            # Add all probabilities
            for k, class_name in enumerate(CLASS_NAMES):
                result[f'prob_{class_name}'] = float(probs[j, k])

            results.append(result)

    return results


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load model
    model, model_args = load_model(args.checkpoint, device)
    transform = get_inference_transform(model_args.image_size)

    # Single image inference
    if args.image:
        print(f"\nProcessing single image: {args.image}")
        result = predict_single_image(model, args.image, transform, device)

        print("\n" + "="*60)
        print("Prediction Results")
        print("="*60)
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence:      {result['confidence']:.4f}")
        print("\nClass Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name:25s}: {prob:.4f}")
        print("="*60 + "\n")

        # Save result if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")

    # Batch inference
    elif args.image_dir:
        print(f"\nProcessing images in directory: {args.image_dir}")

        # Find all image files
        image_dir = Path(args.image_dir)
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(list(image_dir.glob(ext)))
            image_paths.extend(list(image_dir.glob(ext.upper())))

        image_paths = [str(p) for p in sorted(image_paths)]
        print(f"Found {len(image_paths)} images")

        if not image_paths:
            print("No images found!")
            return

        # Predict
        results = predict_batch(model, image_paths, transform, device, args.batch_size)

        # Print summary
        print("\n" + "="*60)
        print("Batch Prediction Summary")
        print("="*60)
        print(f"Total images processed: {len(results)}")

        # Class distribution
        pred_classes = [r['predicted_class'] for r in results]
        print("\nPredicted class distribution:")
        for class_name in CLASS_NAMES:
            count = pred_classes.count(class_name)
            pct = 100 * count / len(results)
            print(f"  {class_name:25s}: {count:5d} ({pct:5.1f}%)")

        # Average confidence per class
        print("\nAverage confidence per predicted class:")
        for class_name in CLASS_NAMES:
            class_results = [r for r in results if r['predicted_class'] == class_name]
            if class_results:
                avg_conf = np.mean([r['confidence'] for r in class_results])
                print(f"  {class_name:25s}: {avg_conf:.4f}")

        print("="*60 + "\n")

        # Save results
        if args.output:
            # Save as CSV
            df = pd.DataFrame(results)
            df.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")
        else:
            print("Warning: --output not specified, results not saved")

    else:
        print("Error: Must specify either --image or --image_dir")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference with trained GZ2 DINOv3 model')

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--image',
        type=str,
        help='Path to single image'
    )
    group.add_argument(
        '--image_dir',
        type=str,
        help='Directory containing images for batch inference'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (JSON for single image, CSV for batch)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for batch inference'
    )

    args = parser.parse_args()
    main(args)
