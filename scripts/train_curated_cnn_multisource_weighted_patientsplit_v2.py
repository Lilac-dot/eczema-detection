"""Second, deliberately more aggressive attempt to improve on the patient-clean SkinDisNet
fine-tune result (Section 4.6.4: AUC 0.4881, +0.0378 vs. baseline, p=0.13 -- not
significant). Two real, non-fabricated changes combined into one run (not isolated as
separate experiments, for time/memory budget on this machine -- reported as a combined
attempt, not a clean single-variable ablation):

1. More real training data where it actually exists: SCIN's EXPANDED train manifest (582
   images, up from 488 -- 47 additional genuine secondary-Eczema cases from SCIN's own
   weighted-label field, built previously by expand_scin_train.py but never successfully
   trained on due to repeated OOM failures). SkinDisNet has no more real data available
   (416 patients is the entire dataset -- verified, the "SkinDisNet_2.zip" archive is the
   same 416 patients' photos plus synthetic augmentation, not new patients), so this lever
   only applies to SCIN.

2. More model capacity / training time: unfreezes layer3 in addition to layer4+fc (one more
   residual block, LR 1e-6 -- 3x smaller than layer4's), and extends training from 10 to 15
   epochs. This tests whether the patient-clean SkinDisNet result was capacity/undertraining-
   limited rather than a genuine ceiling.

SkinDisNet train/val remain the patient-clean split (868/303 images, 208/83 patients, 0%
overlap with the 539-image/125-patient test set). Checkpoint selection unchanged: mean of
internal/SCIN/SkinDisNet validation accuracy each epoch, validation only.

Saves to models/curated_resnet18_multisource_weighted_patientsplit_v2.pt.
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
EPOCHS = 15
FC_LR = 3e-5
LAYER4_LR = 3e-6
LAYER3_LR = 1e-6
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL_PATH = MODELS_DIR / "curated_resnet18_balanced.pt"
OUT_PATH = MODELS_DIR / "curated_resnet18_multisource_weighted_patientsplit_v2.pt"
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


FROZEN_SUBMODULES = ["conv1", "bn1", "layer1", "layer2"]  # layer3 now unfrozen, unlike prior scripts


def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=DEVICE))
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer3.parameters():
        p.requires_grad = True
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
    scin_train = ManifestDataset(SCIN_DIR / "manifest_scin_train_expanded.csv", train_tf)
    skindisnet_train = ManifestDataset(SKINDISNET_DIR / "manifest_skindisnet_train_patientsplit.csv", train_tf)
    train_ds = ConcatDataset([internal_train, scin_train, skindisnet_train])
    print(f"Combined training set: {len(train_ds)} images "
          f"(internal={len(internal_train)}, scin_expanded={len(scin_train)}, "
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
        {"params": model.layer4.parameters(), "lr": LAYER4_LR},
        {"params": model.layer3.parameters(), "lr": LAYER3_LR},
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
