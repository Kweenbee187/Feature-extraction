"""
Multilabel Image Classification — Inference Script
Architecture : ResNet18 (must match train.py)

── Local / CLI usage ─────────────────────────────────────────
  python inference.py --image path/to/image.jpg
  python inference.py --image path/to/image.jpg --model outputs/multilabel_model.pth

── Google Colab usage ────────────────────────────────────────
  Set IMAGE_PATH and MODEL_PATH below, then run the cell.
"""

import sys
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import timm

# ─────────────────────────────────────────────
# ★ EDIT THESE PATHS (for Colab / notebook use)
# ─────────────────────────────────────────────
IMAGE_PATH = "data/images/sample.jpg"
MODEL_PATH = "outputs/multilabel_model.pth"


# ─────────────────────────────────────────────
# MODEL  — must match train.py exactly
# ─────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    model       = timm.create_model("resnet18", pretrained=False, num_classes=0)
    in_features = model.num_features    # 512 for ResNet18
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(128, num_classes),
    )
    return model


# ─────────────────────────────────────────────
# LOAD CHECKPOINT
# ─────────────────────────────────────────────
def load_model(model_path: str, device: torch.device):
    checkpoint  = torch.load(model_path, map_location=device)
    attr_cols   = checkpoint.get("attr_cols",  ["Attr1", "Attr2", "Attr3", "Attr4"])
    threshold   = checkpoint.get("threshold",  0.5)
    img_size    = checkpoint.get("img_size",   224)
    num_classes = len(attr_cols)

    model = build_model(num_classes).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"[Model] ResNet18 loaded from '{model_path}'")
    print(f"        Epoch {checkpoint.get('epoch', '?')} | "
          f"val_loss = {checkpoint.get('val_loss', float('nan')):.4f}")
    return model, attr_cols, threshold, img_size


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
def predict(image_path: str, model_path: str = MODEL_PATH) -> list:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, attr_cols, threshold, img_size = load_model(model_path, device)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image  = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).squeeze(0)

    present_attrs = []
    print(f"\n{'─'*50}")
    print(f"  Image : {image_path}")
    print(f"{'─'*50}")
    print(f"  {'Attribute':<10} {'Probability':>12}   Status")
    print(f"  {'─'*42}")
    for attr, prob in zip(attr_cols, probs.tolist()):
        status = "[PRESENT]" if prob >= threshold else "[absent] "
        print(f"  {attr:<10} {prob:>11.4f}   {status}")
        if prob >= threshold:
            present_attrs.append(attr)

    print(f"{'─'*50}")
    if present_attrs:
        print(f"  Attributes present : {present_attrs}")
    else:
        print("  No attributes detected above threshold.")
    print(f"{'─'*50}\n")

    return present_attrs


# ─────────────────────────────────────────────
# ENTRY POINT — CLI or Colab cell
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Strip Colab kernel flags (e.g. -f /path/to/kernel.json)
    clean_argv = [a for a in sys.argv[1:]
                  if not a.startswith("-f") and "kernel" not in a]

    if not clean_argv:
        # No CLI args → fall back to paths set at the top
        predict(IMAGE_PATH, MODEL_PATH)
    else:
        import argparse
        parser = argparse.ArgumentParser(description="Multilabel Classification — Inference")
        parser.add_argument("--image", type=str, required=True, help="Path to input image")
        parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to model checkpoint")
        args = parser.parse_args(clean_argv)
        predict(args.image, args.model)
