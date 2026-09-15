"""
Experiment 2 (external-generalization improvement attempt): fine-tune the DEPLOYED model
(curated_resnet18_balanced.pt's weights, not from scratch) on internal-train + a train slice
of SCIN + a train slice of SkinDisNet combined, at a conservative learning rate, to test
whether a small amount of cross-source exposure closes the zero-shot generalization gap
found in Phase 1 (docs/external_validation_2026-09-15.md) without forgetting the internal
task. Run split_external_manifest.py on both external manifests first.

Same moderate augmentation recipe as the original train_curated_cnn_balanced.py (not the
heavier one from Experiment 1) -- this keeps "more diverse training sources" as the single
variable under test here, separate from Experiment 1's "heavier augmentation" variable.

Model-selection checkpoint criterion: mean of (internal_val_acc, scin_val_acc,
skindisnet_val_acc) each epoch, so the saved model isn't just good on one domain.

10 epochs, not 15: this is fine-tuning an already-converged model at ~3x lower LR than the
original training run, which typically needs fewer epochs to re-converge, and the combined
dataset is larger than the original (CPU-only machine, wall-clock budget).

Saves to models/curated_resnet18_multisource.pt -- never overwrites the deployed
curated_resnet18_balanced.pt.
"""
import csv
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from PIL import Image

from paths import SKINDISEASE_DIR, SCIN_DIR, SKINDISNET_DIR, MODELS_DIR
MODELS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
FC_LR = 3e-5
BACKBONE_LR = 3e-6
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL_PATH = MODELS_DIR / "curated_resnet18_balanced.pt"
OUT_PATH = MODELS_DIR / "curated_resnet18_multisource.pt"

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
        return self.transform(img), label


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
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=DEVICE))
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True
    return model.to(DEVICE)


def set_train_mode(model):
    model.train()
    for name in FROZEN_SUBMODULES:
        getattr(model, name).eval()


def run_train_epoch(model, loader, criterion, optimizer):
    set_train_mode(model)
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_acc(model, loader):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return correct / total


def main():
    print(f"Device: {DEVICE}")
    train_tf, eval_tf = make_transforms()

    train_ds = ConcatDataset([
        ManifestDataset(SKINDISEASE_DIR / "manifest_curated_v3_train.csv", train_tf),
        ManifestDataset(SCIN_DIR / "manifest_scin_train.csv", train_tf),
        ManifestDataset(SKINDISNET_DIR / "manifest_skindisnet_train.csv", train_tf),
    ])
    print(f"Combined training set: {len(train_ds)} images")

    internal_val = ManifestDataset(SKINDISEASE_DIR / "manifest_curated_v3_val.csv", eval_tf)
    scin_val = ManifestDataset(SCIN_DIR / "manifest_scin_val.csv", eval_tf)
    skindisnet_val = ManifestDataset(SKINDISNET_DIR / "manifest_skindisnet_val.csv", eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    internal_val_loader = DataLoader(internal_val, batch_size=32, shuffle=False, num_workers=0)
    scin_val_loader = DataLoader(scin_val, batch_size=32, shuffle=False, num_workers=0)
    skindisnet_val_loader = DataLoader(skindisnet_val, batch_size=32, shuffle=False, num_workers=0)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {"params": model.fc.parameters(), "lr": FC_LR},
        {"params": model.layer4.parameters(), "lr": BACKBONE_LR},
    ])

    best_combined = 0.0
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_train_epoch(model, train_loader, criterion, optimizer)
        internal_acc = eval_acc(model, internal_val_loader)
        scin_acc = eval_acc(model, scin_val_loader)
        skindisnet_acc = eval_acc(model, skindisnet_val_loader)
        combined = (internal_acc + scin_acc + skindisnet_acc) / 3
        dt = time.time() - t0
        print(f"Epoch {epoch}/{EPOCHS} ({dt:.1f}s) train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} | val: internal={internal_acc:.4f} "
              f"scin={scin_acc:.4f} skindisnet={skindisnet_acc:.4f} combined={combined:.4f}")
        if combined > best_combined:
            best_combined = combined
            torch.save(model.state_dict(), OUT_PATH)
            print(f"  -> saved new best model (combined_val_acc={combined:.4f})")

    total_dt = time.time() - t_start
    print(f"\nBest combined val accuracy: {best_combined:.4f}")
    print(f"Total wall-clock time: {total_dt:.1f}s ({total_dt/60:.1f} min)")
    print(f"Model saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
