"""
Improved version of train_curated_cnn.py. Two changes based on what the first run showed:

1. Partial fine-tuning instead of a fully frozen backbone: unfreeze ResNet18's last
   residual block (layer4) plus the final FC layer, not just the FC layer alone. The
   first run's CNN (69.55% test acc) underperformed the colour-feature LightGBM model
   (82.43%) -- with everything but the last layer frozen, the network could only
   recombine generic ImageNet features, not adapt to this task's specific texture cues.
   layer4 gets a 10x smaller learning rate than the FC layer (standard discriminative
   fine-tuning -- avoids wrecking the pretrained features with too-large updates).
2. Softer class-imbalance weighting: the first run weighted the loss by the raw
   class-count ratio (~4.3:1), which likely explains its low precision (35%) / high
   recall (71%) -- it over-predicted Eczema. Using the square root of that ratio (~2.1:1)
   is a gentler correction, still accounting for imbalance without overcorrecting.

Also runs for 15 epochs instead of 10, since fine-tuning newly-unfrozen layers typically
needs a bit more time to settle than training a single linear layer.
"""
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

ROOT = Path(r"C:\Users\tishy\Documents\Honors\SkinDisease")
MODELS_DIR = Path(r"C:\Users\tishy\Documents\Honors\models")
MODELS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
FC_LR = 1e-4
BACKBONE_LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
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
    train_ds = CuratedDataset(ROOT / "manifest_curated_train.csv", train_tf)
    val_ds = CuratedDataset(ROOT / "manifest_curated_val.csv", eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model()
    pos_weight = (3120 / 723) ** 0.5  # softened from the raw ~4.3x ratio used last time
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(DEVICE)
    print(f"Class weight for Eczema: {pos_weight:.3f} (was 4.315 last run)")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam([
        {"params": model.fc.parameters(), "lr": FC_LR},
        {"params": model.layer4.parameters(), "lr": BACKBONE_LR},
    ])

    best_val_acc = 0.0
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
            torch.save(model.state_dict(), MODELS_DIR / "curated_resnet18_v2.pt")
            print(f"  -> saved new best model (val_acc={val_acc:.4f})")

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {MODELS_DIR / 'curated_resnet18_v2.pt'}")


if __name__ == "__main__":
    main()
