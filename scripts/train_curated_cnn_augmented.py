"""
Experiment 1 (external-generalization improvement attempt): same architecture/recipe as
train_curated_cnn_balanced.py, same internal-only training data, but with heavier/more
diverse augmentation aimed at reducing sensitivity to source-specific photographic
conventions (framing/zoom, focus, compression) -- since Phase 1 (docs/external_validation_
2026-09-15.md) found the deployed model's failure on external data isn't the original
brightness shortcut. No external data is touched here, so this is directly zero-shot
comparable to Phase 1's SCIN/SkinDisNet numbers.

Saves to models/curated_resnet18_augmented.pt -- never overwrites the deployed
curated_resnet18_balanced.pt.
"""
import csv
import io
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageFilter

from paths import SKINDISEASE_DIR as ROOT, MODELS_DIR
MODELS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
FC_LR = 1e-4
BACKBONE_LR = 1e-5
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_PATH = MODELS_DIR / "curated_resnet18_augmented.pt"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class RandomJPEGDegrade:
    """Simulates a different compression pipeline than whatever produced the source
    archive's images, by re-encoding through JPEG at a random low-ish quality."""
    def __init__(self, p=0.3, quality_range=(30, 80)):
        self.p = p
        self.quality_range = quality_range

    def __call__(self, img):
        if random.random() > self.p:
            return img
        q = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomGaussianBlur:
    def __init__(self, p=0.3, radius_range=(0.3, 2.0)):
        self.p = p
        self.radius_range = radius_range

    def __call__(self, img):
        if random.random() > self.p:
            return img
        r = random.uniform(*self.radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius=r))


class CuratedDataset(Dataset):
    def __init__(self, manifest_path, transform):
        self.rows = []
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.rows.append((row["path"], int(row["label"])))
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label = self.rows[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


def make_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # RandomResizedCrop stands in for the fixed Resize -- simulates different
    # framing/zoom/distance-from-lesion across photographic sources.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.05),
        RandomGaussianBlur(p=0.3),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomEqualize(p=0.15),
        RandomJPEGDegrade(p=0.3),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


FROZEN_SUBMODULES = ["conv1", "bn1", "layer1", "layer2", "layer3"]


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(DEVICE)


def set_train_mode(model):
    model.train()
    for name in FROZEN_SUBMODULES:
        getattr(model, name).eval()


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    set_train_mode(model) if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


def main():
    print(f"Device: {DEVICE}")
    train_tf, eval_tf = make_transforms()
    train_ds = CuratedDataset(ROOT / "manifest_curated_v3_train.csv", train_tf)
    val_ds = CuratedDataset(ROOT / "manifest_curated_v3_val.csv", eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {"params": model.fc.parameters(), "lr": FC_LR},
        {"params": model.layer4.parameters(), "lr": BACKBONE_LR},
    ])

    best_val_acc = 0.0
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        dt = time.time() - t0
        print(f"Epoch {epoch}/{EPOCHS} ({dt:.1f}s) "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUT_PATH)
            print(f"  -> saved new best model (val_acc={val_acc:.4f})")

    total_dt = time.time() - t_start
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Total wall-clock time: {total_dt:.1f}s ({total_dt/60:.1f} min)")
    print(f"Model saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
