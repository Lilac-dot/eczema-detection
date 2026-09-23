"""
Leave-one-dataset-out (LODO): trains a ResNet18 (ImageNet-pretrained init, same
architecture/recipe family as train_scin_resnet18.py / train_skindisnet_resnet18.py --
layer4+fc unfrozen, discriminative LRs, moderate augmentation, kept identical on purpose
so results differ only in training data) on the UNION of two of the three sources'
train splits, with a WeightedRandomSampler giving each of the two combined sources equal
weight regardless of raw size (same equal-weighting principle as
train_curated_cnn_multisource_weighted.py -- config internal_scin combines 2,327
internal + 488 SCIN, an imbalance that would otherwise drown out SCIN's gradient signal).
The third source is held out entirely from training/val and used only as the final,
separately-evaluated test set (see eval_lodo_matrix.py) -- see
docs/cross_dataset_matrix_2026-09-15.md for the corresponding single-source cells this
is meant to sit alongside.

Checkpoint criterion: mean of the two combined sources' own validation accuracy (their
val splits, not the held-out third source's -- using the held-out source's data for
checkpoint selection would defeat the point of holding it out).

Run one config at a time (`python train_lodo_resnet18.py <config_name>`) rather than
looping all three in one process, so each run starts with a clean memory state on this
memory-constrained machine -- see mem_guard.py, reused from
train_curated_cnn_multisource_weighted_v2.py.
"""
import argparse
import random
import time
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image

from paths import MODELS_DIR, SKINDISEASE_DIR, SCIN_DIR, SKINDISNET_DIR
from mem_guard import free_mem_mb, require_free_mb
MODELS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 8  # reduced from the matrix scripts' 32, per this machine's memory constraint
EPOCHS = 15
FC_LR = 1e-4
BACKBONE_LR = 1e-5
SEED = 42
MIN_FREE_MB = 400.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIGS = {
    "internal_scin": dict(
        train_a=(SKINDISEASE_DIR / "manifest_curated_v3_train.csv", "internal"),
        train_b=(SCIN_DIR / "manifest_scin_train.csv", "scin"),
        val_a=SKINDISEASE_DIR / "manifest_curated_v3_val.csv",
        val_b=SCIN_DIR / "manifest_scin_val.csv",
        held_out="skindisnet",
        out_name="lodo_internal_scin_resnet18.pt",
    ),
    "internal_skindisnet": dict(
        train_a=(SKINDISEASE_DIR / "manifest_curated_v3_train.csv", "internal"),
        train_b=(SKINDISNET_DIR / "manifest_skindisnet_train_clean.csv", "skindisnet"),
        val_a=SKINDISEASE_DIR / "manifest_curated_v3_val.csv",
        val_b=SKINDISNET_DIR / "manifest_skindisnet_val_clean.csv",
        held_out="scin",
        out_name="lodo_internal_skindisnet_resnet18.pt",
    ),
    "scin_skindisnet": dict(
        train_a=(SCIN_DIR / "manifest_scin_train.csv", "scin"),
        train_b=(SKINDISNET_DIR / "manifest_skindisnet_train_clean.csv", "skindisnet"),
        val_a=SCIN_DIR / "manifest_scin_val.csv",
        val_b=SKINDISNET_DIR / "manifest_skindisnet_val_clean.csv",
        held_out="internal",
        out_name="lodo_scin_skindisnet_resnet18.pt",
    ),
}

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
    parser = argparse.ArgumentParser()
    parser.add_argument("config", choices=list(CONFIGS.keys()))
    args = parser.parse_args()
    cfg = CONFIGS[args.config]

    print(f"Config: {args.config}  (held out entirely: {cfg['held_out']})")
    print(f"Device: {DEVICE}")
    if not require_free_mb(MIN_FREE_MB, "before start"):
        print("Aborting before loading any data/model.")
        return

    train_tf, eval_tf = make_transforms()
    (path_a, name_a), (path_b, name_b) = cfg["train_a"], cfg["train_b"]
    ds_a = ManifestDataset(path_a, train_tf)
    ds_b = ManifestDataset(path_b, train_tf)
    train_ds = ConcatDataset([ds_a, ds_b])
    print(f"Combined training set: {len(train_ds)} images ({name_a}={len(ds_a)}, {name_b}={len(ds_b)})")

    sample_weights = [1.0 / len(ds_a)] * len(ds_a) + [1.0 / len(ds_b)] * len(ds_b)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    val_a = ManifestDataset(cfg["val_a"], eval_tf)
    val_b = ManifestDataset(cfg["val_b"], eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_a_loader = DataLoader(val_a, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_b_loader = DataLoader(val_b, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {"params": model.fc.parameters(), "lr": FC_LR},
        {"params": model.layer4.parameters(), "lr": BACKBONE_LR},
    ])

    out_path = MODELS_DIR / cfg["out_name"]
    resume_path = MODELS_DIR / (cfg["out_name"] + ".resume.pt")

    # Resume support: this machine's background-task supervisor can kill a long training
    # run mid-way if system-wide memory gets tight at any point during the ~50-65 minute
    # run, even when the process itself never asked for much (see docs/cross_dataset_matrix_
    # 2026-09-15.md Part 5-6 -- confirmed a genuine system memory-pressure kill, not a bug in
    # this script). Saving a resumable checkpoint after every epoch (not just improvements)
    # means a kill loses at most one epoch's work instead of the whole run.
    start_epoch = 1
    best_combined = 0.0
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        best_combined = ckpt["best_combined"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch} (previous best_combined={best_combined:.4f})")
        if start_epoch > EPOCHS:
            print("Resume checkpoint already covers all epochs -- nothing left to train.")
            return

    stopped_early = False
    t_start = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        if not require_free_mb(MIN_FREE_MB, f"before epoch {epoch}"):
            stopped_early = True
            break
        t0 = time.time()
        train_loss, train_acc = run_train_epoch(model, train_loader, criterion, optimizer)
        acc_a = eval_acc(model, val_a_loader)
        acc_b = eval_acc(model, val_b_loader)
        combined = (acc_a + acc_b) / 2
        dt = time.time() - t0
        print(f"Epoch {epoch}/{EPOCHS} ({dt:.1f}s) train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} | val_{name_a}={acc_a:.4f} val_{name_b}={acc_b:.4f} "
              f"combined={combined:.4f} | free_ram_after={free_mem_mb():.0f}MB")
        if combined > best_combined:
            best_combined = combined
            torch.save(model.state_dict(), out_path)
            print(f"  -> saved new best model (combined_val_acc={combined:.4f})")
        # Resumable checkpoint every epoch, regardless of whether it was the best one --
        # this is what a restart loads from, separate from out_path (which only ever holds
        # the best-val-accuracy weights for actual downstream use).
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "best_combined": best_combined},
                   resume_path)

    total_dt = time.time() - t_start
    print(f"\nBest combined val accuracy: {best_combined:.4f}")
    print(f"Total wall-clock time this run: {total_dt:.1f}s ({total_dt/60:.1f} min)")
    print(f"Stopped early due to low memory: {stopped_early}")
    print(f"Model saved to {out_path}" if out_path.exists() else "No checkpoint was ever saved.")
    if not stopped_early:
        resume_path.unlink(missing_ok=True)
        print("Training complete -- resume checkpoint removed.")
    else:
        print(f"Resume checkpoint kept at {resume_path} -- rerun this same command to continue.")


if __name__ == "__main__":
    main()
