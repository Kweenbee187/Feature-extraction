import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm

# ─────────────────────────────────────────────
# CONFIG  — edit these paths before running
# ─────────────────────────────────────────────
IMAGES_DIR     = "data/images"          # folder containing all .jpg / .png images
LABELS_FILE    = "data/labels.txt"      # whitespace-separated label file
MODEL_SAVE     = "outputs/multilabel_model.pth"
PLOT_SAVE      = "outputs/loss_curve.png"

ATTR_COLS      = ["Attr1", "Attr2", "Attr3", "Attr4"]
NUM_CLASSES    = 4
IMG_SIZE       = 224
BATCH_SIZE     = 16
NUM_EPOCHS     = 50
LR_HEAD        = 2e-3
LR_FINETUNE    = 1e-4       # LR after partial unfreeze
THRESHOLD      = 0.5
UNFREEZE_EPOCH = 3           # unfreeze last ResNet block after this epoch
EARLY_STOP_PAT = 5           # stop if val loss doesn't improve for N epochs
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("outputs", exist_ok=True)


# ─────────────────────────────────────────────
# 1. LABEL PARSING
# ─────────────────────────────────────────────
def parse_labels(labels_file: str) -> pd.DataFrame:
    """
    Reads labels.txt (whitespace-separated).
    Auto-detects whether a header row exists.
    NA values become np.nan; 0/1 stay as float.
    """
    with open(labels_file, "r") as f:
        first_token = f.readline().strip().split()[0]
    has_header = not first_token.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

    col_names = ["image_name"] + ATTR_COLS

    df = pd.read_csv(
        labels_file,
        sep=r"\s+",
        na_values=["NA"],
        dtype=str,
        engine="python",
        header=0 if has_header else None,
        names=None if has_header else col_names,
    )

    if has_header:
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={df.columns[0]: "image_name"})

    for col in ATTR_COLS:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Columns in file: {list(df.columns)}"
            )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows whose image file doesn't exist on disk
    missing = [
        row["image_name"]
        for _, row in df.iterrows()
        if not os.path.exists(os.path.join(IMAGES_DIR, row["image_name"]))
    ]
    if missing:
        print(f"[WARNING] {len(missing)} image(s) not found on disk — skipping.")
        df = df[~df["image_name"].isin(missing)].reset_index(drop=True)

    print(f"\n{'='*55}")
    print(f"  Labels loaded : {len(df)} images from '{labels_file}'")
    print(f"{'='*55}")
    print(f"  {'Attribute':<10} {'Positives':>10} {'Negatives':>10} {'NA':>6}")
    print(f"  {'-'*38}")
    for col in ATTR_COLS:
        pos = int((df[col] == 1.0).sum())
        neg = int((df[col] == 0.0).sum())
        na  = int(df[col].isna().sum())
        print(f"  {col:<10} {pos:>10} {neg:>10} {na:>6}")
    print(f"{'='*55}\n")

    return df


# ─────────────────────────────────────────────
# 2. COMPUTE CLASS WEIGHTS  (handles imbalance)
# ─────────────────────────────────────────────
def compute_pos_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    pos_weight[j] = (#negatives_j) / (#positives_j)
    Passed to BCEWithLogitsLoss — rare positives get higher penalty when missed.
    """
    pos_weights = []
    for col in ATTR_COLS:
        n_pos = max(int((df[col] == 1.0).sum()), 1)
        n_neg = max(int((df[col] == 0.0).sum()), 1)
        pos_weights.append(n_neg / n_pos)

    weights = torch.tensor(pos_weights, dtype=torch.float32)
    print(f"[Weights] pos_weight per attribute: {[round(w, 2) for w in weights.tolist()]}\n")
    return weights


# ─────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────
class MultilabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str, transform=None):
        self.df         = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform  = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["image_name"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Cannot open image: {img_path}") from e

        if self.transform:
            image = self.transform(image)

        # mask[j]=True  → label known, contributes to loss
        # mask[j]=False → label is NA, excluded from loss
        raw    = row[ATTR_COLS].values.astype(np.float32)
        mask   = ~np.isnan(raw)
        labels = np.where(np.isnan(raw), 0.0, raw)

        return (
            image,
            torch.tensor(labels, dtype=torch.float32),
            torch.tensor(mask,   dtype=torch.bool),
        )


# ─────────────────────────────────────────────
# 4. TRANSFORMS / AUGMENTATION
# ─────────────────────────────────────────────
def get_transforms(train: bool) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 24, IMG_SIZE + 24)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),  # simulates occlusion
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


# ─────────────────────────────────────────────
# 5. MODEL  — ResNet18
# ─────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    """
    ResNet18 pretrained on ImageNet via timm.

    Two-stage fine-tuning strategy:
      Stage 1 (epochs 0..UNFREEZE_EPOCH-1): backbone fully frozen, head only.
      Stage 2 (UNFREEZE_EPOCH onward)      : layer4 (last ResNet block) unfrozen.

    Why ResNet18?
      - Skip connections give stable gradients on small datasets.
      - Well-proven on small dataset fine-tuning benchmarks.
      - Less prone to oscillating loss compared to EfficientNetB0.
    """
    model       = timm.create_model("resnet18", pretrained=True, num_classes=0)
    in_features = model.num_features    # 512 for ResNet18

    # Freeze entire backbone
    for param in model.parameters():
        param.requires_grad = False

    # Classification head (no sigmoid — BCEWithLogitsLoss handles it)
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(128, num_classes),
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] ResNet18 ready (backbone frozen).")
    print(f"        in_features={in_features} | Trainable params (head): {trainable:,} | Device: {DEVICE}\n")
    return model


def unfreeze_last_layer(model: nn.Module):
    """Unfreeze only layer4 (~1.3M params) for fine-tuning."""
    for name, param in model.named_parameters():
        if "layer4" in name:
            param.requires_grad = True
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] layer4 UNFROZEN — total trainable params: {total:,}")


# ─────────────────────────────────────────────
# 6. EARLY STOPPING
# ─────────────────────────────────────────────
class EarlyStopping:
    """Stops training and saves best checkpoint when val loss stops improving."""

    def __init__(self, patience: int = 7, model_save: str = MODEL_SAVE):
        self.patience   = patience
        self.model_save = model_save
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_epoch = 0

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_epoch = epoch + 1
            torch.save(
                {
                    "epoch":       epoch + 1,
                    "model_state": model.state_dict(),
                    "val_loss":    val_loss,
                    "attr_cols":   ATTR_COLS,
                    "threshold":   THRESHOLD,
                    "img_size":    IMG_SIZE,
                },
                self.model_save,
            )
            print(f"  ✓ Best model saved (val_loss={val_loss:.4f}, epoch={epoch+1})")
            return False
        else:
            self.counter += 1
            print(
                f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs "
                f"(best={self.best_loss:.4f} @ epoch {self.best_epoch})"
            )
            return self.counter >= self.patience


# ─────────────────────────────────────────────
# 7. MASKED BCE LOSS  (handles NA labels)
# ─────────────────────────────────────────────
class MaskedBCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogitsLoss that skips NA label positions.
      1. Compute element-wise BCE → shape (B, num_classes)
      2. Zero out NA positions via boolean mask
      3. Average over known labels only
    """

    def __init__(self, pos_weight: torch.Tensor = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None

        loss_matrix = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pw, reduction="none"
        )
        loss_matrix = loss_matrix * mask.float()

        n_known = mask.float().sum()
        if n_known == 0:
            return torch.tensor(0.0, requires_grad=True, device=logits.device)
        return loss_matrix.sum() / n_known


# ─────────────────────────────────────────────
# 8. TRAINING — one epoch
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss       = 0.0
    iteration_losses = []

    for i, (images, labels, masks) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        masks  = masks.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels, masks)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        iteration_losses.append(loss.item())

        if (i + 1) % 10 == 0 or (i + 1) == len(loader):
            print(f"  Epoch {epoch+1:02d} | Iter {i+1:03d}/{len(loader)} | Batch Loss: {loss.item():.4f}")

    return total_loss / len(loader), iteration_losses


# ─────────────────────────────────────────────
# 9. VALIDATION
# ─────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion):
    """Returns avg val loss and per-attribute accuracy (known labels only)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_masks = [], [], []

    for images, labels, masks in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        masks  = masks.to(DEVICE)

        logits = model(images)
        loss   = criterion(logits, labels, masks)
        total_loss += loss.item()

        preds = (torch.sigmoid(logits) >= THRESHOLD).float()
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_masks.append(masks.cpu())

    avg_loss   = total_loss / len(loader)
    all_preds  = torch.cat(all_preds,  dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks  = torch.cat(all_masks,  dim=0)

    attr_acc = {}
    for j, col in enumerate(ATTR_COLS):
        known = all_masks[:, j]
        if known.sum() == 0:
            attr_acc[col] = float("nan")
        else:
            correct = (all_preds[known, j] == all_labels[known, j]).float()
            attr_acc[col] = correct.mean().item()

    return avg_loss, attr_acc


# ─────────────────────────────────────────────
# 10. PLOT LOSS CURVE
# ─────────────────────────────────────────────
def plot_loss_curve(all_iter_losses, train_epoch_losses, val_epoch_losses,
                    stopped_at: int = None, save_path: str = PLOT_SAVE):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: iteration-level training loss
    axes[0].plot(range(1, len(all_iter_losses) + 1), all_iter_losses,
                 linewidth=1.0, color="steelblue", alpha=0.8)
    axes[0].set_xlabel("iteration_number")
    axes[0].set_ylabel("training_loss")
    axes[0].set_title("Multilabel Classification — Iteration Loss")
    axes[0].grid(True, alpha=0.3)

    # Right: epoch-level train vs val
    epochs = range(1, len(train_epoch_losses) + 1)
    axes[1].plot(epochs, train_epoch_losses, label="Train Loss", color="steelblue", marker="o", markersize=4)
    axes[1].plot(epochs, val_epoch_losses,   label="Val Loss",   color="tomato",    marker="s", markersize=4)

    if stopped_at and stopped_at < len(train_epoch_losses):
        axes[1].axvline(x=stopped_at, color="gray", linestyle="--", alpha=0.7,
                        label=f"Early stop (epoch {stopped_at})")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Train vs Validation Loss per Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\n[Plot] Saved to '{save_path}'")


# ─────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────
def main():
    print(f"\n{'='*55}")
    print(f"  Multilabel Classification — Training")
    print(f"  Model       : ResNet18 (timm)")
    print(f"  Device      : {DEVICE}")
    print(f"  Images dir  : {IMAGES_DIR}")
    print(f"  Labels file : {LABELS_FILE}")
    print(f"  Max epochs  : {NUM_EPOCHS}  |  Batch size : {BATCH_SIZE}")
    print(f"  Early stop  : patience={EARLY_STOP_PAT} epochs")
    print(f"{'='*55}")

    df = parse_labels(LABELS_FILE)

    # 80/20 train-val split
    df       = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split    = int(0.8 * len(df))
    train_df = df.iloc[:split]
    val_df   = df.iloc[split:]
    print(f"[Split] Train: {len(train_df)} samples  |  Val: {len(val_df)} samples\n")

    train_ds = MultilabelDataset(train_df, IMAGES_DIR, get_transforms(train=True))
    val_ds   = MultilabelDataset(val_df,   IMAGES_DIR, get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model       = build_model(NUM_CLASSES).to(DEVICE)
    pos_weights = compute_pos_weights(train_df)
    criterion   = MaskedBCEWithLogitsLoss(pos_weight=pos_weights)

    early_stopping = EarlyStopping(patience=EARLY_STOP_PAT, model_save=MODEL_SAVE)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    all_iter_losses    = []
    train_epoch_losses = []
    val_epoch_losses   = []
    stopped_at         = None

    for epoch in range(NUM_EPOCHS):

        # Stage 2: unfreeze layer4 after UNFREEZE_EPOCH
        if epoch == UNFREEZE_EPOCH:
            unfreeze_last_layer(model)
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR_FINETUNE, weight_decay=1e-3,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

        train_loss, iter_losses = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        all_iter_losses.extend(iter_losses)
        train_epoch_losses.append(train_loss)

        val_loss, attr_acc = validate(model, val_loader, criterion)
        val_epoch_losses.append(val_loss)
        scheduler.step(val_loss)

        acc_str = "  ".join(
            f"{col}={v:.1%}" if not np.isnan(v) else f"{col}=N/A"
            for col, v in attr_acc.items()
        )
        print(f"\n{'─'*55}")
        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}]  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy  →  {acc_str}")
        print(f"{'─'*55}")

        if early_stopping.step(val_loss, model, epoch):
            stopped_at = epoch + 1
            print(f"\n[EarlyStopping] Triggered at epoch {stopped_at}. "
                  f"Best val loss: {early_stopping.best_loss:.4f} @ epoch {early_stopping.best_epoch}")
            break

    plot_loss_curve(all_iter_losses, train_epoch_losses, val_epoch_losses, stopped_at=stopped_at)

    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Best val loss : {early_stopping.best_loss:.4f}")
    print(f"  Best epoch    : {early_stopping.best_epoch}")
    print(f"  Model saved   : {MODEL_SAVE}")
    print(f"  Loss curve    : {PLOT_SAVE}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
