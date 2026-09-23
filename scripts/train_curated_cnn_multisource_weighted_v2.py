"""
Experiment 2c: same domain-equal-weighted multi-source fine-tune as
train_curated_cnn_multisource_weighted.py (Experiment 2b), but using the EXPANDED SCIN
training manifest (488->582 images, via expand_scin_train.py) and the CLEANED SkinDisNet
training manifest (855->853, 2 near-duplicate train/val and val/test pairs removed, see
docs/cross_dataset_matrix_2026-09-15.md Part 1). Only the training data volume/cleanliness
changed -- same starting weights, LRs, epoch count, checkpoint criterion, and sampler logic
as 2b, so any difference in outcome is attributable to the data, not the method.

Run on a memory-constrained machine (~3GB free at time of writing, was OOM-killing training
before a restart) -- batch_size dropped from 32 to 8 and free-memory is logged before
training and after every epoch via a dependency-free ctypes call to the Windows API, so a
memory problem is visible rather than a silent OS kill. If free memory drops under ~400MB
between epochs, training stops gracefully (whatever checkpoint was last saved is kept).

Saves to models/curated_resnet18_multisource_weighted_v2.pt.
"""
import csv
import ctypes
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image

from paths import SKINDISEASE_DIR, SCIN_DIR, SKINDISNET_DIR, MODELS_DIR
MODELS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 8  # reduced from 32 (Experiment 2b) for this machine's current memory constraint
EPOCHS = 10
FC_LR = 3e-5
BACKBONE_LR = 3e-6
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL_PATH = MODELS_DIR / "curated_resnet18_balanced.pt"
OUT_PATH = MODELS_DIR / "curated_resnet18_multisource_weighted_v2.pt"
MIN_FREE_MB = 400.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def free_mem_mb():
    """Dependency-free free-physical-RAM check via the Windows API (no psutil needed)."""
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullAvailPhys / (1024 * 1024)


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
    if start_free < MIN_FREE_MB:
        print(f"Free RAM ({start_free:.0f} MB) is already below the {MIN_FREE_MB:.0f} MB "
              f"safety floor before any data/model has even been loaded -- aborting before "
              f"starting rather than risking an OS OOM kill mid-load.")
        return
    train_tf, eval_tf = make_transforms()

    internal_train = ManifestDataset(SKINDISEASE_DIR / "manifest_curated_v3_train.csv", train_tf)
    scin_train = ManifestDataset(SCIN_DIR / "manifest_scin_train_expanded.csv", train_tf)
    skindisnet_train = ManifestDataset(SKINDISNET_DIR / "manifest_skindisnet_train_clean.csv", train_tf)
    train_ds = ConcatDataset([internal_train, scin_train, skindisnet_train])
    print(f"Combined training set: {len(train_ds)} images "
          f"(internal={len(internal_train)}, scin_expanded={len(scin_train)}, "
          f"skindisnet_clean={len(skindisnet_train)})")

    sample_weights = (
        [1.0 / len(internal_train)] * len(internal_train)
        + [1.0 / len(scin_train)] * len(scin_train)
        + [1.0 / len(skindisnet_train)] * len(skindisnet_train)
    )
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    internal_val = ManifestDataset(SKINDISEASE_DIR / "manifest_curated_v3_val.csv", eval_tf)
    scin_val = ManifestDataset(SCIN_DIR / "manifest_scin_val.csv", eval_tf)
    skindisnet_val = ManifestDataset(SKINDISNET_DIR / "manifest_skindisnet_val_clean.csv", eval_tf)

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
                  f"{MIN_FREE_MB:.0f} MB safety floor -- stopping gracefully instead of "
                  f"risking an OS OOM kill. Last saved checkpoint (if any) is kept.")
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
