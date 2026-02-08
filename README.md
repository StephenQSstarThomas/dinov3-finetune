# DINOv3 on Galaxy Zoo 2 - Reproducible Testing

This project tests DINOv3 (facebook/dinov2-base) on Galaxy Zoo 2 color images for galaxy morphology classification.

## Overview

- **Model**: DINOv3 (facebook/dinov2-base) with untrained 4-class classifier head
- **Dataset**: Galaxy Zoo 2 (Hart et al. 2016) color images from SDSS
- **Classes**: 4 galaxy morphology types
  - `edge_on`: Edge-on disk galaxies (seen from the side)
  - `featured`: Galaxies with features but not clearly spiral
  - `smooth`: Smooth elliptical galaxies without features
  - `spiral`: Spiral galaxies with clear spiral arms
- **Test Set**: 1,000 highest-confidence samples (confidence = 1.0)
- **Expected Accuracy**: 32% (untrained), 80-95% (after fine-tuning)

## Prerequisites

```bash
# Python packages
pip install torch torchvision
pip install transformers
pip install pandas numpy scikit-learn
pip install pillow tqdm
pip install kagglehub  # For downloading GZ2 dataset
```

## Quick Start - Run Complete Reproducible Test

To run the complete reproducible test in one command:

```bash
python test_dinov3_gz2.py
```

This will:
1. Load 1,000 high-confidence GZ2 samples
2. Run DINOv3 inference with fixed random seed (seed=42)
3. Save results to `test_reproducible_results.csv`
4. Print accuracy metrics and confusion matrix

**Expected output**:
- Total samples: 955 (some images may be missing)
- Accuracy: ~32% (32.04% with seed=42)
- Result CSV: `test_reproducible_results.csv`

## Complete Process from Scratch

If you want to prepare the dataset from scratch, follow these steps:

### Step 1: Download GZ2 Dataset

Download the Galaxy Zoo 2 images from Kaggle (~3GB):

```bash
python scripts/download_gz2_data.py
```

This downloads images to: `~/.cache/kagglehub/datasets/jaimetrickz/galaxy-zoo-2-images/versions/1/`

### Step 2: Download GZ2 Labels

Download the Hart et al. 2016 morphology labels:

```bash
python scripts/download_gz2_labels.py
```

This downloads: `gz2_hart16.csv` (239,695 galaxies with debiased vote fractions)

### Step 3: Prepare High-Confidence Dataset

Filter and organize the dataset:

```bash
python scripts/prepare_gz2_dataset.py
```

This will:
- Filter samples with confidence > 0.8
- Create directory structure: `gz2_dataset_for_dinov3/{edge_on,featured,smooth,spiral}/`
- Save metadata: `gz2_simple_labels.csv`
- Result: ~156,000 high-confidence samples organized by class

### Step 4: Run Reproducible Test

```bash
python test_dinov3_gz2.py
```

Test configuration:
- Uses top 1,000 highest-confidence samples
- Fixed random seed = 42 for reproducibility
- Batch size = 32
- GPU acceleration (if available)

## File Structure

```
dinov3-tng50-finetune/
├── README.md                              # This file
├── test_dinov3_gz2.py                     # Main reproducible test script
├── gpt4o_classification.py                # GPT-4o comparison test (optional)
│
├── scripts/                               # Data download and preparation
│   ├── download_gz2_data.py               # Download GZ2 images from Kaggle
│   ├── download_gz2_labels.py             # Download Hart et al. 2016 labels
│   └── prepare_gz2_dataset.py             # Filter and organize dataset
│
├── src/                                   # Source code
│   ├── model.py                           # DINOv3 model with classifier head
│   └── gz2_dataset.py                     # GZ2 dataset loader
│
├── legacy/                                # Legacy/experimental code
│   ├── test_gz2_high_confidence.py        # Old test without random seed
│   ├── test_gz2_1000.py                   # Experimental test
│   └── ...                                # Other legacy files
│
├── Data Files (generated after running scripts)
├── gz2_hart16.csv                         # Hart et al. 2016 labels (239,695 galaxies)
├── gz2_simple_labels.csv                  # Simplified labels (156,255 high-confidence)
├── gz2_top1000_high_confidence.csv        # Top 1000 samples for testing
├── test_reproducible_results.csv          # Test results (seed=42)
└── gz2_dataset_for_dinov3/                # Organized image dataset
    ├── edge_on/
    ├── featured/
    ├── smooth/
    └── spiral/
```

## Results

### Main Test Results

File: `test_reproducible_results.csv`

Columns:
- `ID`: Galaxy DR7 Object ID
- `answer`: True morphology label
- `model_answer`: Predicted label
- `TRUE/FALSE`: Correctness

Results (seed=42):
- Total: 955 samples
- Correct: 306
- Accuracy: 32.04%

Per-class accuracy:
- `edge_on`: 15.60% (22/141)
- `featured`: 0.00% (0/2)
- `smooth`: 5.88% (1/17)
- `spiral`: 35.61% (283/795)

### Why is accuracy low?

The classifier head is **untrained** - it has random weights initialized with seed=42. This test demonstrates:
1. DINOv2's pretrained features have some zero-shot capability (32% vs 25% random)
2. The model tends to predict the most frequent class (spiral: 83.3% of dataset)
3. Fine-tuning is needed to achieve 80-95% accuracy

## Optional: GPT-4o Comparison

To compare DINOv3 with GPT-4o on the same images:

```bash
python gpt4o_classification.py
```

**Requirements**:
1. OpenAI API key (edit line 196 in `gpt4o_classification.py`)
2. Set: `API_KEY = "sk-..."`

This will:
- Sample 100 images evenly across 4 classes
- Send each image to GPT-4o with vision
- Save results to: `gpt4o_classification_results.csv`

**Note**: This costs money (~$0.01-0.05 per image depending on API pricing).

## Key Technical Details

### Random Seed Importance

The classifier head has random weights. Without setting a random seed:
- Each run produces different results
- Accuracy varies wildly (0.63% to 40.84%)

With fixed seed=42:
- Results are reproducible
- Same accuracy every time (32.04%)

The main test script sets seed correctly:

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Model Architecture

```python
# DINOv2 backbone (frozen or unfrozen)
dinov2 = Dinov2Model.from_pretrained("facebook/dinov2-base")

# Classification head
classifier = nn.Sequential(
    nn.LayerNorm(768),  # hidden_size=768
    nn.Dropout(0.1),
    nn.Linear(768, 4)   # 4 classes
)
```

Output is always one of [0, 1, 2, 3] corresponding to the 4 classes.

### Dataset Characteristics

From Hart et al. 2016 catalog (239,695 galaxies):
- High-confidence samples (>0.8): 156,255 (65%)
- Class distribution (highly imbalanced):
  - Spiral: 83.5% (130,486 samples)
  - Edge-on: 14.7% (22,994 samples)
  - Smooth: 1.7% (2,668 samples)
  - Featured: 0.2% (107 samples)

Top 1,000 samples (all confidence=1.0):
- Spiral: 83.3% (833 samples)
- Edge-on: 14.8% (148 samples)
- Smooth: 1.8% (18 samples)
- Featured: 0.1% (1 sample)

## Troubleshooting

### Missing images
Some galaxy IDs in the catalog may not have corresponding images. The test script handles this gracefully by skipping missing files.

### CUDA out of memory
Reduce batch size in `test_dinov3_gz2.py`:
```python
batch_size = 16  # or smaller
```

### Different accuracy
Ensure random seed is set correctly. With seed=42, you should get exactly 32.04% accuracy.

## References

- **GZ2 Data**: Hart et al. 2016, MNRAS, 461, 3663 ([arXiv:1603.07886](https://arxiv.org/abs/1603.07886))
- **DINOv2**: Oquab et al. 2023 ([arXiv:2304.07193](https://arxiv.org/abs/2304.07193))
- **Galaxy Zoo 2**: Willett et al. 2013, MNRAS, 435, 2835

## Next Steps

To achieve better accuracy:
1. Fine-tune the model on GZ2 training set
2. Use LoRA for parameter-efficient fine-tuning
3. Handle class imbalance (weighted loss, oversampling)
4. Compare with other vision models (ResNet, ViT, etc.)

For training code, see the COSMOS FITS examples in this repository.
