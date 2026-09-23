"""Experiment 2b, re-run with SkinDisNet's PATIENT-LEVEL split instead of its image-level
split -- the follow-up motivated by the 2026-09-17 report-revision audit
(docs patient-leakage finding: 81% of SkinDisNet's image-level test partition's patients
also appeared in its training partition). Identical to
train_curated_cnn_multisource_weighted.py in every other respect (same starting weights,
same LRs, same epoch count, same equal-weighted WeightedRandomSampler, same checkpoint
criterion) -- SkinDisNet's manifest paths are the only variable changed, so any difference
in SkinDisNet's held-out result is attributable to the leakage fix, not a different method.

SCIN and internal splits are untouched (SCIN has no usable person-level identifier to
re-split by; internal has no patient identifier at all -- both documented limitations, not
oversights).

Uses scripts/build_skindisnet_patient_split.py's output:
SkinDisNet/manifest_skindisnet_{train,val,test}_patientsplit.csv (868/303/539 images,
208/83/125 patients, zero patient overlap across splits by construction).

Batch size reduced to 8 and a free-RAM safety check added (this development machine has
repeatedly shown volatile free memory -- see docs/finetuning_v2_expanded_2026-09-16.md),
so a memory problem causes a graceful stop rather than an unclean OS kill.

Saves to models/curated_resnet18_multisource_weighted_patientsplit.pt.
"""
import csv
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image

from paths import SKINDISEASE_DIR, SCIN_DIR, SKINDISNET_DIR, MODELS_DIR
from mem_guard import free_mem_mb, require_free_mb

MODELS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10
FC_LR = 3e-5
BACKBONE_LR = 3e-6
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL_PATH = MODELS_DIR / "curated_resnet18_balanced.pt"
OUT_PATH = MODELS_DIR / "curated_resnet18_multisource_weighted_patientsplit.pt"
MIN_FREE_MB = 400.0

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
    start_free = free_mem_mb()
    print(f"Free RAM before start: {start_free:.0f} MB")
    if not require_free_mb(MIN_FREE_MB, "before start"):
        return
    train_tf, eval_tf = make_transforms()

    internal_train = ManifestDataset(SKINDISEASE_DIR / "manifest_curated_v3_train.csv", train_tf)
    scin_train = ManifestDataset(SCIN_DIR / "manifest_scin_train.csv", train_tf)
    skindisnet_train = ManifestDataset(SKINDISNET_DIR / "manifest_skindisnet_train_patientsplit.csv", train_tf)
    train_ds = ConcatDataset([internal_train, scin_train, skindisnet_train])
    print(f"Combined training set: {len(train_ds)} images "
          f"(internal={len(internal_train)}, scin={len(scin_train)}, "
          f"skindisnet_patientsplit={len(skindisnet_train)})")

    sample_weights = (
        [1.0 / len(internal_train)] * len(internal_train)
        + [1.0 / len(scin_train)] * len(scin_train)
        + [1.0 / len(skindisnet_train)] * len(skindisnet_train)
    )
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    internal_val = ManifestDataset(SKINDISEASE_DIR / "manifest_curated_v3_val.csv", eval_tf)
    scin_val = ManifestDataset(SCIN_DIR / "manifest_scin_val.csv", eval_tf)
    skindisnet_val = ManifestDataset(SKINDISNET_DIR / "manifest_skindisnet_val_patientsplit.csv", eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    internal_val_loader = DataLoader(internal_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    scin_val_loader = DataLoader(scin_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    skindisnet_val_loader = DataLoader(skindisnet_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {"params": model.fc.parameters(), "lr": FC_LR},
        {"params": model.layer4.parameters(), "lr": BACKBONE_LR},
    ])

    best_combined = 0.0
    stopped_early_oom_risk = False
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        free_before = free_mem_mb()
        if free_before < MIN_FREE_MB:
            print(f"\nFree RAM before epoch {epoch} is {free_before:.0f} MB, below the "
                  f"{MIN_FREE_MB:.0f} MB safety floor -- stopping gracefully. Last saved "
                  f"checkpoint (if any) is kept.")
            stopped_early_oom_risk = True
            break

        t0 = time.time()
        train_loss, train_acc = run_train_epoch(model, train_loader, criterion, optimizer)
        internal_acc = eval_acc(model, internal_val_loader)
        scin_acc = eval_acc(model, scin_val_loader)
        skindisnet_acc = eval_acc(model, skindisnet_val_loader)
        combined = (internal_acc + scin_acc + skindisnet_acc) / 3
        dt = time.time() - t0
        free_after = free_mem_mb()
        print(f"Epoch {epoch}/{EPOCHS} ({dt:.1f}s) train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} | val: internal={internal_acc:.4f} "
              f"scin={scin_acc:.4f} skindisnet={skindisnet_acc:.4f} combined={combined:.4f} "
              f"| free_ram_before={free_before:.0f}MB free_ram_after={free_after:.0f}MB")
        if combined > best_combined:
            best_combined = combined
            torch.save(model.state_dict(), OUT_PATH)
            print(f"  -> saved new best model (combined_val_acc={combined:.4f})")

    total_dt = time.time() - t_start
    print(f"\nBest combined val accuracy: {best_combined:.4f}")
    print(f"Total wall-clock time: {total_dt:.1f}s ({total_dt/60:.1f} min)")
    print(f"Stopped early due to low memory: {stopped_early_oom_risk}")
    print(f"Model saved to {OUT_PATH}" if OUT_PATH.exists() else "No checkpoint was ever saved.")


if __name__ == "__main__":
    main()
