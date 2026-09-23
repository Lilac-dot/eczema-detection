"""Two-stage transfer learning for the eczema severity proxy (Mild/Moderate/Severe),
adapted from train_transfer_cnn.py's binary disease-classifier pipeline for 3-class
classification. Reuses that script's MODEL_CONFIGS (architecture shape descriptions)
and overall stage1-frozen / stage2-finetune structure; see that file's own docstring for
why architectures need per-model config rather than one shared code path.

Labels: docs/severity_scale_selection_2026-09-21.md's EASI-signs proxy, 3 bands (Mild/
Moderate/Severe) derived from the empirical score distribution, not a validated clinical
severity score -- see that doc before citing this model's accuracy as anything more than
agreement with this project's own labeling heuristic.

TEST-SET ISOLATION: this script only ever opens severity_train.csv and severity_val.csv.
severity_test.csv is never referenced anywhere in this file -- see
docs/severity_test_set_protocol_2026-09-21.md, committed before this script was first run.

Usage:
    python train_severity_cnn.py --model shufflenet_v2_x0_5
    python train_severity_cnn.py --model efficientnet_b0
"""
import argparse
import csv
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import timm
except ImportError:
    timm = None

from paths import ROOT as PROJECT_ROOT
from mem_guard import require_free_mb
from train_transfer_cnn import MODEL_CONFIGS, _first_linear_in_features

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
WEIGHT_DECAY = 1e-4
STAGE1_LR = 1e-4
STAGE2_LR = 1e-5
STAGE1_MAX_EPOCHS = 25
STAGE2_MAX_EPOCHS = 25
PATIENCE = 5
MIN_FREE_MB = 400
NUM_CLASSES = 3
CLASS_NAMES = ["Mild", "Moderate", "Severe"]

SEVERITY_DIR = PROJECT_ROOT / "dataset" / "severity_labels"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SeverityDataset(Dataset):
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
        img = Image.open(PROJECT_ROOT / path).convert("RGB")
        return self.transform(img), label


def make_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def build_model(model_name):
    """Same backbone/head-replacement logic as train_transfer_cnn.build_model, but the
    final Linear outputs NUM_CLASSES logits (softmax via CrossEntropyLoss) instead of a
    single sigmoid logit."""
    cfg = MODEL_CONFIGS[model_name]

    if cfg["source"] == "torchvision":
        ctor = getattr(models, cfg["ctor"])
        backbone = ctor(weights="DEFAULT")
    elif cfg["source"] == "timm":
        if timm is None:
            raise RuntimeError(
                f"model {model_name!r} requires the 'timm' package, which is not installed")
        backbone = timm.create_model(cfg["ctor"], pretrained=True, num_classes=1000)
    else:
        raise ValueError(f"unknown source {cfg['source']!r} for {model_name}")

    final_attr = cfg["final_attr"]
    existing_final = getattr(backbone, final_attr)

    if cfg["pattern"] == "squeezenet":
        new_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cfg["in_features"], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES),
        )
    else:
        in_features = cfg["in_features"] or _first_linear_in_features(existing_final)
        new_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES),
        )

    setattr(backbone, final_attr, new_head)

    for name in cfg["backbone_names"]:
        for p in getattr(backbone, name).parameters():
            p.requires_grad = False

    return backbone.to(DEVICE)


def _under_any_prefix(name, prefixes):
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def set_frozen_eval(model, cfg, unfrozen_paths):
    model.train()
    trainable_prefixes = list(unfrozen_paths) + [cfg["final_attr"]]
    for name, module in model.named_modules():
        if name and not _under_any_prefix(name, trainable_prefixes):
            module.eval()


def unfreeze_backbone(model, cfg):
    for path in cfg["stage2_unfreeze"]:
        for p in model.get_submodule(path).parameters():
            p.requires_grad = True


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_epoch(model, loader, criterion, optimizer, cfg, unfrozen_paths):
    is_train = optimizer is not None
    if is_train:
        set_frozen_eval(model, cfg, unfrozen_paths)
    else:
        model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []
    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels_t = labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels_t)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            preds = logits.detach().argmax(dim=1).cpu().numpy()
            correct += (preds == labels.numpy()).sum()
            total += imgs.size(0)
            total_loss += loss.item() * imgs.size(0)
            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.tolist())
    metrics = {"loss": total_loss / total, "accuracy": correct / total}
    metrics["precision_macro"] = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    metrics["f1_macro"] = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    metrics["_labels"] = all_labels
    metrics["_preds"] = all_preds
    return metrics


def train_stage(model, train_loader, val_loader, cfg, unfrozen_paths, lr, max_epochs,
                 stage_name, history, history_live_path):
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = None
    epochs_without_improvement = 0
    t_stage_start = time.time()

    for epoch in range(1, max_epochs + 1):
        if not require_free_mb(MIN_FREE_MB, context=f"{stage_name} epoch {epoch}"):
            print(f"Stopping {stage_name} early at epoch {epoch} due to low free RAM.")
            break
        t0 = time.time()
        train_m = run_epoch(model, train_loader, criterion, optimizer, cfg, unfrozen_paths)
        val_m = run_epoch(model, val_loader, criterion, optimizer=None, cfg=cfg, unfrozen_paths=unfrozen_paths)
        dt = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "stage": stage_name, "epoch": epoch, "lr": current_lr, "time_s": dt,
            "train_loss": train_m["loss"], "train_accuracy": train_m["accuracy"],
            "train_f1_macro": train_m["f1_macro"],
            "val_loss": val_m["loss"], "val_accuracy": val_m["accuracy"],
            "val_precision_macro": val_m["precision_macro"], "val_recall_macro": val_m["recall_macro"],
            "val_f1_macro": val_m["f1_macro"],
        })
        print(f"[{stage_name}] epoch {epoch}/{max_epochs} ({dt:.1f}s) lr={current_lr:.2e} "
              f"train_loss={train_m['loss']:.4f} train_acc={train_m['accuracy']:.4f} "
              f"val_loss={val_m['loss']:.4f} val_acc={val_m['accuracy']:.4f} "
              f"val_f1_macro={val_m['f1_macro']:.4f}", flush=True)
        with open(history_live_path, "w") as hf:
            json.dump(history, hf, indent=2)

        scheduler.step(val_m["loss"])

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
            print(f"  -> new best ({stage_name}), val_loss={best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"  -> early stopping ({stage_name}) at epoch {epoch}, "
                      f"no val_loss improvement for {PATIENCE} epochs")
                break

    if best_state is None:
        raise RuntimeError(
            f"{stage_name} completed zero epochs before stopping (likely low free RAM) -- "
            f"no checkpoint to restore. Free up RAM and rerun this model; nothing was trained."
        )
    model.load_state_dict(best_state)
    stage_time = time.time() - t_stage_start
    return model, best_epoch, best_val_loss, stage_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()
    cfg = MODEL_CONFIGS[args.model]

    out_dir = EXPERIMENTS_DIR / f"{args.model}_severity"
    (out_dir / "config").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "history").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Model: {cfg['display_name']} (severity, {NUM_CLASSES}-class: {CLASS_NAMES})")

    train_tf, eval_tf = make_transforms()
    train_ds = SeverityDataset(SEVERITY_DIR / "severity_train.csv", train_tf)
    val_ds = SeverityDataset(SEVERITY_DIR / "severity_val.csv", eval_tf)
    train_eval_ds = SeverityDataset(SEVERITY_DIR / "severity_train.csv", eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model(args.model)
    total_params, trainable_stage1 = count_params(model)
    print(f"Total params: {total_params:,}  Trainable (stage 1, head only): {trainable_stage1:,}")

    history = []
    history_live_path = out_dir / "history" / "training_history_live.json"

    model, s1_best_epoch, s1_best_val_loss, s1_time = train_stage(
        model, train_loader, val_loader, cfg=cfg, unfrozen_paths=[],
        lr=STAGE1_LR, max_epochs=STAGE1_MAX_EPOCHS, stage_name="stage1_frozen",
        history=history, history_live_path=history_live_path,
    )
    torch.save(model.state_dict(), out_dir / "checkpoints" / "stage1_best.pt")

    unfreeze_backbone(model, cfg)
    _, trainable_stage2 = count_params(model)
    print(f"Trainable (stage 2, head + unfrozen blocks {cfg['stage2_unfreeze']}): {trainable_stage2:,}")

    model, s2_best_epoch, s2_best_val_loss, s2_time = train_stage(
        model, train_loader, val_loader, cfg=cfg, unfrozen_paths=cfg["stage2_unfreeze"],
        lr=STAGE2_LR, max_epochs=STAGE2_MAX_EPOCHS, stage_name="stage2_finetune",
        history=history, history_live_path=history_live_path,
    )
    final_ckpt_path = out_dir / "checkpoints" / "best_overall.pt"
    torch.save(model.state_dict(), final_ckpt_path)
    model_size_mb = final_ckpt_path.stat().st_size / (1024 * 1024)

    model.eval()
    warmup = 3
    n_timed = min(50, len(val_ds))
    with torch.no_grad():
        for i in range(warmup):
            img, _ = val_ds[i]
            model(img.unsqueeze(0).to(DEVICE))
        t0 = time.time()
        for i in range(n_timed):
            img, _ = val_ds[i]
            model(img.unsqueeze(0).to(DEVICE))
        inference_time_ms_per_image = (time.time() - t0) / n_timed * 1000
    print(f"Inference time: {inference_time_ms_per_image:.2f} ms/image "
          f"(batch size 1, {DEVICE}, mean over {n_timed} val images)")

    final_val_m = run_epoch(model, val_loader, nn.CrossEntropyLoss(), optimizer=None,
                             cfg=cfg, unfrozen_paths=cfg["stage2_unfreeze"])
    final_train_m = run_epoch(model, train_eval_loader, nn.CrossEntropyLoss(), optimizer=None,
                               cfg=cfg, unfrozen_paths=cfg["stage2_unfreeze"])

    with open(out_dir / "history" / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    epochs_x = list(range(1, len(history) + 1))
    n_stage1 = sum(1 for h in history if h["stage"] == "stage1_frozen")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs_x, [h["train_loss"] for h in history], label="Train loss")
    ax.plot(epochs_x, [h["val_loss"] for h in history], label="Val loss")
    ax.axvline(n_stage1 + 0.5, color="gray", linestyle="--", label="Stage 1 -> Stage 2")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"{cfg['display_name']} (severity) -- Loss (train/val only)")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "plots" / "loss_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs_x, [h["train_accuracy"] for h in history], label="Train accuracy")
    ax.plot(epochs_x, [h["val_accuracy"] for h in history], label="Val accuracy")
    ax.axvline(n_stage1 + 0.5, color="gray", linestyle="--", label="Stage 1 -> Stage 2")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title(f"{cfg['display_name']} (severity) -- Accuracy (train/val only)")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "plots" / "accuracy_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs_x, [h["train_f1_macro"] for h in history], label="Train F1 (macro)")
    ax.plot(epochs_x, [h["val_f1_macro"] for h in history], label="Val F1 (macro)")
    ax.axvline(n_stage1 + 0.5, color="gray", linestyle="--", label="Stage 1 -> Stage 2")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1 (macro)")
    ax.set_title(f"{cfg['display_name']} (severity) -- F1 (train/val only)")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "plots" / "f1_curve.png", dpi=150)
    plt.close(fig)

    labels = final_val_m["_labels"]
    preds = final_val_m["_preds"]
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{cfg['display_name']} (severity) -- Validation confusion matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "val_confusion_matrix.png", dpi=150)
    plt.close(fig)

    config = {
        "model": args.model,
        "display_name": cfg["display_name"],
        "task": "eczema severity proxy (EASI-signs, 3-class: Mild/Moderate/Severe)",
        "labels_doc": "docs/severity_scale_selection_2026-09-21.md",
        "seed": SEED,
        "image_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "device": str(DEVICE),
        "dataset": {
            "train_manifest": str(SEVERITY_DIR / "severity_train.csv"),
            "val_manifest": str(SEVERITY_DIR / "severity_val.csv"),
            "test_manifest_used": False,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "class_names": CLASS_NAMES,
        },
        "architecture": {
            "backbone": cfg["display_name"],
            "pretrained": "ImageNet1K",
            "head": f"GlobalAvgPool -> Linear(->128) -> ReLU -> Dropout(0.3) -> Linear(128->{NUM_CLASSES}) (CrossEntropyLoss)",
            "total_params": total_params,
            "trainable_params_stage1": trainable_stage1,
            "trainable_params_stage2": trainable_stage2,
            "unfrozen_backbone_blocks_stage2": cfg["stage2_unfreeze"],
            "model_size_mb": round(model_size_mb, 2),
        },
        "training": {
            "stage1": {"lr": STAGE1_LR, "max_epochs": STAGE1_MAX_EPOCHS, "patience": PATIENCE,
                       "best_epoch": s1_best_epoch, "best_val_loss": s1_best_val_loss,
                       "time_s": round(s1_time, 1)},
            "stage2": {"lr": STAGE2_LR, "max_epochs": STAGE2_MAX_EPOCHS, "patience": PATIENCE,
                       "best_epoch": s2_best_epoch, "best_val_loss": s2_best_val_loss,
                       "time_s": round(s2_time, 1)},
            "optimizer": "AdamW", "weight_decay": WEIGHT_DECAY,
            "lr_scheduler": "ReduceLROnPlateau(mode=min, factor=0.5, patience=2, monitors val_loss)",
            "class_weighting": "none -- train split is ~32/37/31% Mild/Moderate/Severe, already balanced",
            "total_time_s": round(s1_time + s2_time, 1),
        },
        "inference_time_ms_per_image": round(inference_time_ms_per_image, 3),
        "final_train_metrics": {
            "loss": final_train_m["loss"], "accuracy": final_train_m["accuracy"],
            "precision_macro": final_train_m["precision_macro"], "recall_macro": final_train_m["recall_macro"],
            "f1_macro": final_train_m["f1_macro"],
        },
        "final_validation_metrics": {
            "loss": final_val_m["loss"], "accuracy": final_val_m["accuracy"],
            "precision_macro": final_val_m["precision_macro"], "recall_macro": final_val_m["recall_macro"],
            "f1_macro": final_val_m["f1_macro"],
        },
        "val_confusion_matrix": cm.tolist(),
        "train_val_gap": {
            "accuracy_gap": round(final_train_m["accuracy"] - final_val_m["accuracy"], 4),
            "f1_macro_gap": round(final_train_m["f1_macro"] - final_val_m["f1_macro"], 4),
        },
    }
    with open(out_dir / "config" / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n=== DONE ===")
    print("Final train metrics:", json.dumps(config["final_train_metrics"], indent=2))
    print("Final val metrics:", json.dumps(config["final_validation_metrics"], indent=2))
    print("Val confusion matrix:", cm.tolist())
    print("Train-val gap:", json.dumps(config["train_val_gap"], indent=2))
    print("TEST SET HAS NOT BEEN ACCESSED.")


if __name__ == "__main__":
    main()
