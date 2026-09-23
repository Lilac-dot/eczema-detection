"""
Fine-tunes the DEPLOYED internal model (curated_resnet18_balanced.pt) on Internal+SCIN
combined (equal-weighted sampling, same principle as train_curated_cnn_multisource_
weighted.py), holding SkinDisNet out ENTIRELY -- not in training, not in validation/
checkpoint selection. This makes SkinDisNet a genuine, never-touched external test set,
unlike the earlier multisource experiments (docs/external_generalization_improvement_
2026-09-15.md) which included real SkinDisNet training data in the mix.

Why fine-tune rather than train from scratch: train_lodo_resnet18.py's internal_scin
config (ImageNet-pretrained init) is the from-scratch version of this same data
combination -- it reached only AUC 0.498 on SkinDisNet. Every comparison in this project
so far (Experiment 2/2b vs. the LODO results) shows fine-tuning an already-good model
beats training an equivalent architecture from scratch. This script tests whether that
advantage holds when SkinDisNet is a true external test rather than a partially-trained-on
source.

Checkpoint selection: mean of Internal-val and SCIN-val accuracy ONLY. SkinDisNet is not
touched anywhere before the final, single evaluation pass at the end -- no test-set
peeking, no retraining based on the SkinDisNet number.
"""
import random
import time
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image

from paths import MODELS_DIR, SKINDISEASE_DIR, SCIN_DIR
from mem_guard import free_mem_mb, require_free_mb
MODELS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10
FC_LR = 3e-5
BACKBONE_LR = 3e-6
SEED = 42
MIN_FREE_MB = 400.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL_PATH = MODELS_DIR / "curated_resnet18_balanced.pt"
OUT_PATH = MODELS_DIR / "finetune_internal_scin_resnet18.pt"
RESUME_PATH = MODELS_DIR / "finetune_internal_scin_resnet18.pt.resume.pt"

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
    if not require_free_mb(MIN_FREE_MB, "before start"):
        print("Aborting before loading any data/model.")
        return

    train_tf, eval_tf = make_transforms()
    internal_train = ManifestDataset(SKINDISEASE_DIR / "manifest_curated_v3_train.csv", train_tf)
    scin_train = ManifestDataset(SCIN_DIR / "manifest_scin_train.csv", train_tf)
    train_ds = ConcatDataset([internal_train, scin_train])
    print(f"Combined training set: {len(train_ds)} images "
          f"(internal={len(internal_train)}, scin={len(scin_train)}) -- SkinDisNet not used here at all")

    sample_weights = ([1.0 / len(internal_train)] * len(internal_train)
                       + [1.0 / len(scin_train)] * len(scin_train))
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    internal_val = ManifestDataset(SKINDISEASE_DIR / "manifest_curated_v3_val.csv", eval_tf)
    scin_val = ManifestDataset(SCIN_DIR / "manifest_scin_val.csv", eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    internal_val_loader = DataLoader(internal_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    scin_val_loader = DataLoader(scin_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {"params": model.fc.parameters(), "lr": FC_LR},
        {"params": model.layer4.parameters(), "lr": BACKBONE_LR},
    ])

    start_epoch = 1
    best_combined = 0.0
    if RESUME_PATH.exists():
        ckpt = torch.load(RESUME_PATH, map_location=DEVICE)
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
        internal_acc = eval_acc(model, internal_val_loader)
        scin_acc = eval_acc(model, scin_val_loader)
        combined = (internal_acc + scin_acc) / 2
        dt = time.time() - t0
        print(f"Epoch {epoch}/{EPOCHS} ({dt:.1f}s) train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} | val_internal={internal_acc:.4f} "
              f"val_scin={scin_acc:.4f} combined={combined:.4f} | free_ram_after={free_mem_mb():.0f}MB")
        if combined > best_combined:
            best_combined = combined
            torch.save(model.state_dict(), OUT_PATH)
            print(f"  -> saved new best model (combined_val_acc={combined:.4f})")
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "best_combined": best_combined},
                   RESUME_PATH)

    total_dt = time.time() - t_start
    print(f"\nBest combined val accuracy: {best_combined:.4f}")
    print(f"Total wall-clock time this run: {total_dt:.1f}s ({total_dt/60:.1f} min)")
    print(f"Stopped early due to low memory: {stopped_early}")
    if not stopped_early:
        RESUME_PATH.unlink(missing_ok=True)
        print("Training complete -- resume checkpoint removed.")
    else:
        print(f"Resume checkpoint kept at {RESUME_PATH} -- rerun this same command to continue.")
    print(f"Model saved to {OUT_PATH}" if OUT_PATH.exists() else "No checkpoint was ever saved.")


if __name__ == "__main__":
    main()
