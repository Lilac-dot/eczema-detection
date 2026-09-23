"""
ResNet18 trained from scratch (ImageNet-pretrained init) on SkinDisNet-train only, for the
SCIN-> cross-dataset generalization matrix cells (docs/cross_dataset_matrix_2026-09-16.md).

Same architecture/recipe family as train_curated_cnn_balanced.py (layer4+fc unfrozen,
discriminative LRs, moderate augmentation) deliberately kept identical so matrix cells
differ only in training DATA, not in modeling choices -- this project avoids becoming a
model-comparison study.

SkinDisNet-train is small (853 images after duplicate cleaning, imbalanced 268 Eczema/
585 Other -- see scripts/split_external_manifest.py and
scripts/check_external_duplicates.py) relative to the internal training set (2,327), so
this model should be read as a lower-confidence data point, not put on equal footing.
No class weighting is applied, matching train_curated_cnn_balanced.py's plain
cross-entropy recipe exactly (kept identical on purpose so matrix cells differ only in
training data) -- the imbalance is a real caveat for interpreting this cell, not
something corrected for.
"""
import random
import time
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from paths import ROOT, MODELS_DIR, SKINDISNET_DIR
MODELS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
FC_LR = 1e-4
BACKBONE_LR = 1e-5
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class ManifestDataset(Dataset):
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
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
    train_ds = ManifestDataset(SKINDISNET_DIR / "manifest_skindisnet_train_clean.csv", train_tf)
    val_ds = ManifestDataset(SKINDISNET_DIR / "manifest_skindisnet_val_clean.csv", eval_tf)
    print(f"SkinDisNet-train: {len(train_ds)}  SkinDisNet-val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {"params": model.fc.parameters(), "lr": FC_LR},
        {"params": model.layer4.parameters(), "lr": BACKBONE_LR},
    ])

    best_val_acc = 0.0
    out_path = MODELS_DIR / "skindisnet_resnet18.pt"
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
            torch.save(model.state_dict(), out_path)
            print(f"  -> saved new best model (val_acc={val_acc:.4f})")

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {out_path}")


if __name__ == "__main__":
    main()
