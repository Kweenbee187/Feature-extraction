# Multilabel Image Classification with ResNet18

A clean PyTorch implementation of multilabel image classification using a pretrained **ResNet18** backbone (via `timm`), with two-stage fine-tuning, masked BCE loss for missing labels, class-imbalance weighting, and early stopping.

Built as an assignment submission for a multilabel classification task on a custom image dataset.

---

## Features

- **ResNet18 backbone** pretrained on ImageNet (via `timm`)
- **Two-stage fine-tuning**: head-only first, then unfreeze `layer4`
- **Masked BCE loss** — gracefully handles `NA` / missing labels per sample
- **Class-imbalance weighting** via `pos_weight` in `BCEWithLogitsLoss`
- **Rich data augmentation**: random crop, flip, colour jitter, perspective, random erasing
- **Early stopping** with automatic best-model checkpointing
- **Loss curve plots**: iteration-level and epoch-level (train vs val)
- Works on **CPU, CUDA, or Google Colab**

---

## Project Structure

```
Feature-extraction/
├── train.py          # Full training pipeline
├── inference.py      # Single-image prediction script
├── requirements.txt  # Python dependencies
├── .gitignore
└── Notebook/          # ← created automatically during training
    ├── Notebook to create model
    └── loss_curve.png
```

---

## Dataset Format

### `data/labels.txt`

Whitespace-separated file. The script **auto-detects** whether a header row is present.

**With header:**
```
image_name  Attr1  Attr2  Attr3  Attr4
img_001.jpg    1      0      1     NA
img_002.jpg    0      1      0      1
```

**Without header (columns assumed in order):**
```
img_001.jpg  1  0  1  NA
img_002.jpg  0  1  0   1
```

- Values must be `0`, `1`, or `NA` (missing label — excluded from loss).
- Image filenames in the first column must match files inside `data/images/`.

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Kweenbee187/Feature-extraction.git
cd Feature-extraction

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your dataset
#    → images go in  data/images/
#    → labels go in  data/labels.txt
```

---

## Training

```bash
python train.py
```

**Key hyperparameters** (edit at the top of `train.py`):

| Variable | Default | Description |
|---|---|---|
| `IMAGES_DIR` | `data/images` | Path to image folder |
| `LABELS_FILE` | `data/labels.txt` | Path to label file |
| `ATTR_COLS` | `["Attr1"..."Attr4"]` | Attribute column names |
| `NUM_CLASSES` | `4` | Number of output classes |
| `BATCH_SIZE` | `16` | Training batch size |
| `NUM_EPOCHS` | `50` | Max training epochs |
| `LR_HEAD` | `2e-3` | LR for head-only stage |
| `LR_FINETUNE` | `1e-4` | LR after layer4 unfreeze |
| `UNFREEZE_EPOCH` | `3` | Epoch at which layer4 is unfrozen |
| `EARLY_STOP_PAT` | `5` | Early stopping patience |
| `THRESHOLD` | `0.5` | Sigmoid threshold for binary prediction |

**Outputs saved to `outputs/`:**
- `multilabel_model.pth` — best checkpoint (lowest val loss)
- `loss_curve.png` — iteration-level and epoch-level loss plots

---

## Inference

```bash
# Single image — command line
python inference.py --image data/images/sample.jpg

# With a custom model path
python inference.py --image data/images/sample.jpg --model outputs/multilabel_model.pth
```

**Example output:**
```
──────────────────────────────────────────────────
  Image : data/images/sample.jpg
──────────────────────────────────────────────────
  Attribute    Probability   Status
  ──────────────────────────────────────────────
  Attr1             0.8821   [PRESENT]
  Attr2             0.1034   [absent] 
  Attr3             0.9210   [PRESENT]
  Attr4             0.3301   [absent] 
──────────────────────────────────────────────────
  Attributes present : ['Attr1', 'Attr3']
──────────────────────────────────────────────────
```

---

## Google Colab

1. Upload your dataset zip to Colab and unzip it.
2. Set paths at the top of `train.py`:
   ```python
   IMAGES_DIR  = "/content/images_dataset/images"
   LABELS_FILE = "/content/labels.txt"
   MODEL_SAVE  = "/content/multilabel_model.pth"
   PLOT_SAVE   = "/content/loss_curve.png"
   ```
3. Run `train.py` as a cell: `!python train.py`
4. For inference, set `IMAGE_PATH` at the top of `inference.py` and run it.

---

## Model Architecture

```
ResNet18 backbone (ImageNet pretrained, timm)
    └── Global Average Pool  →  512-dim feature vector
Classification Head:
    BatchNorm1d(512)
    Dropout(0.5)
    Linear(512 → 128)
    ReLU
    Dropout(0.3)
    Linear(128 → num_classes)      ← raw logits (no sigmoid)
```

Loss: `BCEWithLogitsLoss` (sigmoid applied internally) with `pos_weight` for class imbalance and masking for NA labels.

---

## Why ResNet18?

- Skip connections ensure stable gradients on small datasets
- Simple, well-understood architecture that generalises reliably
- Two-stage fine-tuning (head → layer4) avoids overfitting
- Less oscillation than EfficientNetB0 observed in experiments

---

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- See `requirements.txt` for full list

---


