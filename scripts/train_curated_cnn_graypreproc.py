"""
Experiment 3 (external-generalization improvement attempt, quick check): a fast gray-world
color-constancy normalization applied to every image (train and eval alike) before the usual
pipeline, to test whether removing per-source color-cast differences helps zero-shot external
performance. Given this project's own documented history of preprocessing fixes NOT fixing
the original shortcut (Section 5.2 of the report -- brightness normalization and cropping
both failed), this is treated as a cheap, quick check, not a major time investment: same
recipe/epochs as the original baseline otherwise, single variable = the gray-world step.

Saves to models/curated_resnet18_graypreproc.pt.
"""
import csv
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from paths import SKINDISEASE_DIR as ROOT, MODELS_DIR
MODELS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
FC_LR = 1e-4
BACKBONE_LR = 1e-5
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_PATH = MODELS_DIR / "curated_resnet18_graypreproc.pt"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class GrayWorldNormalize:
    """Classic gray-world color constancy: scale each RGB channel so its mean matches the
    overall gray-level mean, removing per-image/per-source color-cast differences (e.g. a
    warmer or cooler white balance from a different camera/lighting setup)."""
    def __call__(self, img):
        arr = np.asarray(img, dtype=np.float64)
        means = arr.reshape(-1, 3).mean(axis=0)
        gray_mean = means.mean()
        scale = gray_mean / np.clip(means, 1e-6, None)
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


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
        return self.transform(img), label


def make_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        GrayWorldNormalize(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        GrayWorldNormalize(),
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
