"""
Two-stage transfer learning for lightweight CNNs on the existing curated
Eczema-vs-7-similar-diseases dataset (SkinDisease/manifest_curated_v3_*).

TEST-SET ISOLATION: this script only ever opens manifest_curated_v3_train.csv and
manifest_curated_v3_val.csv. manifest_curated_v3_test.csv is never referenced anywhere
in this file -- test-set isolation is enforced structurally (the file path is simply
never opened), not just by convention, so there is no code path that could accidentally
touch it during training, tuning, or checkpoint selection.

Usage:
    python train_transfer_cnn.py --model efficientnet_b0
    python train_transfer_cnn.py --model <any key in MODEL_CONFIGS>

Writes to experiments/<model>/{config,checkpoints,history,plots}/.

Architectures don't share one internal structure (EfficientNet/MobileNet use a
single `features` Sequential + `classifier`; ShuffleNetV2 uses separate named
stages + `fc`; SqueezeNet's `classifier` does its own pooling; timm models expose
`blocks` alongside a separate stem/head) -- see MODEL_CONFIGS below, which
describes each architecture's shape once, and the generic build_model/
unfreeze_backbone/set_frozen_eval functions work off that description rather than
assuming one fixed layout.
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
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import timm
except ImportError:
    timm = None

from paths import SKINDISEASE_DIR as ROOT, ROOT as PROJECT_ROOT
from mem_guard import require_free_mb

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

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Each entry fully describes one architecture's shape, since they are not uniform:
#   pattern: "features_classifier" (features:Sequential + classifier ending in a
#            Linear; EfficientNet/MobileNet family) | "squeezenet" (classifier does
#            its own GAP+Conv2d, no separate Linear to reuse) | "stages_fc"
#            (ShuffleNetV2: several named top-level stages + a plain fc Linear)
#   source: "torchvision" or "timm"
#   ctor: the model function/name to construct
#   backbone_names: top-level attribute names that make up the backbone (frozen
#            entirely in stage 1)
#   final_attr: the attribute name to replace with our own classifier head
#   stage2_unfreeze: dotted submodule paths (via model.get_submodule) to unfreeze
#            in stage 2 -- picked as roughly the last ~15-20% of each backbone's
#            depth, so the comparison across architectures uses a comparable
#            relative amount of fine-tuning, not an identical absolute layer count
#   in_features: fixed input width to our new head; None means auto-detect from
#            the architecture's own existing final Linear layer
MODEL_CONFIGS = {
    "efficientnet_b0": dict(
        display_name="EfficientNet-B0", pattern="features_classifier",
        source="torchvision", ctor="efficientnet_b0",
        backbone_names=["features"], final_attr="classifier",
        stage2_unfreeze=["features.7", "features.8"], in_features=None,
    ),
    "mobilenetv3_small": dict(
        display_name="MobileNetV3-Small", pattern="features_classifier",
        source="torchvision", ctor="mobilenet_v3_small",
        backbone_names=["features"], final_attr="classifier",
        stage2_unfreeze=["features.11", "features.12"], in_features=None,
    ),
    "mobilenetv3_large": dict(
        display_name="MobileNetV3-Large", pattern="features_classifier",
        source="torchvision", ctor="mobilenet_v3_large",
        backbone_names=["features"], final_attr="classifier",
        stage2_unfreeze=["features.15", "features.16"], in_features=None,
    ),
    "mobilenetv2_100": dict(
        # Substitute for the originally-requested MobileNetV2 0.35x: neither
        # torchvision nor timm ships real ImageNet-pretrained weights at that
        # width (confirmed -- both mark it "untrained"), so training it would be
        # random-init, not transfer learning, and not a fair comparison against
        # the other 8 models. MobileNetV2 1.0x has real pretrained weights.
        display_name="MobileNetV2 1.0x (substitute for MobileNetV2 0.35x -- no "
                      "real pretrained weights exist for 0.35x in torchvision or timm)",
        pattern="features_classifier", source="torchvision", ctor="mobilenet_v2",
        backbone_names=["features"], final_attr="classifier",
        stage2_unfreeze=["features.16", "features.17", "features.18"], in_features=None,
    ),
    "squeezenet1_1": dict(
        display_name="SqueezeNet 1.1", pattern="squeezenet",
        source="torchvision", ctor="squeezenet1_1",
        backbone_names=["features"], final_attr="classifier",
        stage2_unfreeze=["features.11", "features.12"], in_features=512,
    ),
    "shufflenet_v2_x0_5": dict(
        display_name="ShuffleNetV2 0.5x", pattern="stages_fc",
        source="torchvision", ctor="shufflenet_v2_x0_5",
        backbone_names=["conv1", "maxpool", "stage2", "stage3", "stage4", "conv5"],
        final_attr="fc", stage2_unfreeze=["stage4", "conv5"], in_features=None,
    ),
    "shufflenet_v2_x1_0": dict(
        display_name="ShuffleNetV2 1.0x", pattern="stages_fc",
        source="torchvision", ctor="shufflenet_v2_x1_0",
        backbone_names=["conv1", "maxpool", "stage2", "stage3", "stage4", "conv5"],
        final_attr="fc", stage2_unfreeze=["stage4", "conv5"], in_features=None,
    ),
    "repghostnet_050": dict(
        # Substitute for the originally-requested GhostNet 0.5x: confirmed "untrained"
        # (no real pretrained weights) in timm; RepGhostNet is an architecturally
        # related, actually-pretrained alternative at the same 0.5x width.
        display_name="RepGhostNet 0.5x (substitute for GhostNet 0.5x -- no real "
                      "pretrained weights exist for GhostNet 0.5x in torchvision or timm)",
        pattern="timm_blocks", source="timm", ctor="repghostnet_050",
        backbone_names=["conv_stem", "bn1", "act1", "blocks", "conv_head", "act2"],
        final_attr="classifier", stage2_unfreeze=["blocks.8", "blocks.9", "conv_head"],
        in_features=None,
    ),
    "efficientnet_lite0": dict(
        display_name="EfficientNet-Lite0", pattern="timm_blocks",
        source="timm", ctor="efficientnet_lite0",
        backbone_names=["conv_stem", "bn1", "blocks", "conv_head", "bn2"],
        final_attr="classifier", stage2_unfreeze=["blocks.5", "blocks.6", "conv_head"],
        in_features=None,
    ),
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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
        return self.transform(img), label


def make_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Moderate augmentation, training data only. Deliberately avoids anything that could
    # alter clinically relevant lesion characteristics (no strong color/hue shifts, no
    # vertical flip since it has no clinical meaning here but horizontal does not distort
    # lesion appearance).
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        normalize,
    ])
    # Validation preprocessing is deterministic: resize + normalize only, no augmentation.
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def _first_linear_in_features(module):
    """Find the input width of the first nn.Linear in a module -- either the
    module itself (ShuffleNetV2's `fc`, timm's `classifier`) or its first Linear
    child in forward order (EfficientNet/MobileNet's `classifier` Sequential,
    where a Dropout or a Linear(960->1280) intermediate layer may come first)."""
    if isinstance(module, nn.Linear):
        return module.in_features
    for child in module.children():
        if isinstance(child, nn.Linear):
            return child.in_features
    raise RuntimeError(f"no nn.Linear found in or under {module!r}")


def build_model(model_name):
    """Builds any architecture in MODEL_CONFIGS, replacing its final classification
    layer(s) with GAP (already part of each backbone's own forward pass, except
    SqueezeNet where it's added explicitly below) -> Dense(128) -> ReLU ->
    Dropout(0.3) -> Dense(1). Sigmoid is applied separately at inference/metrics
    time; training uses BCEWithLogitsLoss on the raw logit, the numerically stable
    equivalent of Sigmoid + BCELoss."""
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
        # SqueezeNet's own classifier does its own GAP+Conv2d(512->1000) instead of
        # GAP-then-Linear -- explicitly add the pool/flatten this replacement head
        # needs, since (unlike every other architecture here) nothing upstream of
        # `classifier` already pools spatial dimensions down to a vector.
        new_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cfg["in_features"], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
    else:
        in_features = cfg["in_features"] or _first_linear_in_features(existing_final)
        new_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    setattr(backbone, final_attr, new_head)

    for name in cfg["backbone_names"]:
        for p in getattr(backbone, name).parameters():
            p.requires_grad = False

    return backbone.to(DEVICE)


def _under_any_prefix(name, prefixes):
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def set_frozen_eval(model, cfg, unfrozen_paths):
    """Keep frozen backbone submodules' BatchNorm in eval mode even while the rest
    of the model is in train() -- otherwise their running_mean/running_var keep
    drifting on this dataset despite those submodules' weights never updating
    (same fix already used in train_curated_cnn_balanced.py's set_train_mode)."""
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
    all_labels, all_probs = [], []
    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels_f = labels.float().unsqueeze(1).to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels_f)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
            preds = (probs >= 0.5).astype(int)
            correct += (preds == labels.numpy()).sum()
            total += imgs.size(0)
            total_loss += loss.item() * imgs.size(0)
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())
    metrics = {"loss": total_loss / total, "accuracy": correct / total}
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    metrics["precision"] = precision_score(all_labels, preds, zero_division=0)
    metrics["recall"] = recall_score(all_labels, preds, zero_division=0)
    metrics["f1"] = f1_score(all_labels, preds, zero_division=0)
    try:
        metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    metrics["_labels"] = all_labels
    metrics["_probs"] = all_probs
    return metrics


def train_stage(model, train_loader, val_loader, cfg, unfrozen_paths, lr, max_epochs,
                 stage_name, history, history_live_path):
    criterion = nn.BCEWithLogitsLoss()
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
            "train_precision": train_m.get("precision"), "train_recall": train_m.get("recall"),
            "train_f1": train_m.get("f1"),
            "val_loss": val_m["loss"], "val_accuracy": val_m["accuracy"],
            "val_precision": val_m["precision"], "val_recall": val_m["recall"],
            "val_f1": val_m["f1"], "val_roc_auc": val_m["roc_auc"],
        })
        print(f"[{stage_name}] epoch {epoch}/{max_epochs} ({dt:.1f}s) lr={current_lr:.2e} "
              f"train_loss={train_m['loss']:.4f} train_acc={train_m['accuracy']:.4f} "
              f"val_loss={val_m['loss']:.4f} val_acc={val_m['accuracy']:.4f} "
              f"val_f1={val_m['f1']:.4f} val_auc={val_m['roc_auc']:.4f}", flush=True)
        # Write history incrementally so progress survives a crash/kill mid-run.
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
        # The mem_guard free-RAM check can fire before epoch 1 ever runs, in which case
        # no checkpoint was ever recorded. Fail with a clear, actionable message instead of
        # letting `load_state_dict(None)` raise a TypeError that points at this line rather
        # than the actual cause (see docs/transfer_cnn_test_set_access_2026-09-18.md, where
        # this crashed squeezenet1_1's run at 287 MB free RAM).
        raise RuntimeError(
            f"{stage_name} completed zero epochs before stopping (likely low free RAM -- "
            f"see the 'Stopping ... due to low free RAM' line above) -- no checkpoint to "
            f"restore. Free up RAM and rerun this model; nothing was trained."
        )
    model.load_state_dict(best_state)
    stage_time = time.time() - t_stage_start
    return model, best_epoch, best_val_loss, stage_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()
    cfg = MODEL_CONFIGS[args.model]

    out_dir = EXPERIMENTS_DIR / args.model
    (out_dir / "config").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "history").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Model: {cfg['display_name']}")

    train_tf, eval_tf = make_transforms()
    train_ds = CuratedDataset(ROOT / "manifest_curated_v3_train.csv", train_tf)
    val_ds = CuratedDataset(ROOT / "manifest_curated_v3_val.csv", eval_tf)
    # Deterministic (no-augmentation) view of the training set, used only for a
    # clean final train-vs-val comparison after training -- separate from
    # train_loader, whose per-epoch "train_accuracy" is computed under augmentation
    # (the standard, noisier in-loop training signal).
    train_eval_ds = CuratedDataset(ROOT / "manifest_curated_v3_train.csv", eval_tf)
    # num_workers=4 was tried to speed up data loading (single-threaded loading was
    # the dominant per-epoch cost, not model compute), but it introduced a silent
    # multiprocessing failure on this Windows machine (process died with no
    # traceback before finishing even one epoch). Reverted to num_workers=0, which
    # already proved reliable (one full, correct epoch completed under it) -- slower
    # but trustworthy matters more than fast but flaky here.
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

    # Inference-time benchmark (val images only -- never touches test).
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

    final_val_m = run_epoch(model, val_loader, nn.BCEWithLogitsLoss(), optimizer=None,
                             cfg=cfg, unfrozen_paths=cfg["stage2_unfreeze"])
    # Clean (no-augmentation) train-set readout of the SAME best-checkpoint model,
    # for an apples-to-apples train-vs-val comparison -- never touches test.
    final_train_m = run_epoch(model, train_eval_loader, nn.BCEWithLogitsLoss(), optimizer=None,
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
    ax.set_title(f"{cfg['display_name']} -- Loss (train/val only)")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "plots" / "loss_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs_x, [h["train_accuracy"] for h in history], label="Train accuracy")
    ax.plot(epochs_x, [h["val_accuracy"] for h in history], label="Val accuracy")
    ax.axvline(n_stage1 + 0.5, color="gray", linestyle="--", label="Stage 1 -> Stage 2")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title(f"{cfg['display_name']} -- Accuracy (train/val only)")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "plots" / "accuracy_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs_x, [h["train_f1"] for h in history], label="Train F1")
    ax.plot(epochs_x, [h["val_f1"] for h in history], label="Val F1")
    ax.axvline(n_stage1 + 0.5, color="gray", linestyle="--", label="Stage 1 -> Stage 2")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1 (Eczema)")
    ax.set_title(f"{cfg['display_name']} -- F1 (train/val only)")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "plots" / "f1_curve.png", dpi=150)
    plt.close(fig)

    labels = final_val_m["_labels"]
    probs = final_val_m["_probs"]
    preds = [1 if p >= 0.5 else 0 for p in probs]
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Other", "Eczema"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Other", "Eczema"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{cfg['display_name']} -- Validation confusion matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "val_confusion_matrix.png", dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC={final_val_m['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"{cfg['display_name']} -- Validation ROC")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "plots" / "val_roc_curve.png", dpi=150)
    plt.close(fig)

    config = {
        "model": args.model,
        "display_name": cfg["display_name"],
        "seed": SEED,
        "image_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "device": str(DEVICE),
        "dataset": {
            "train_manifest": str(ROOT / "manifest_curated_v3_train.csv"),
            "val_manifest": str(ROOT / "manifest_curated_v3_val.csv"),
            "test_manifest_used": False,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
        },
        "architecture": {
            "backbone": cfg["display_name"],
            "pretrained": "ImageNet1K",
            "head": "GlobalAvgPool -> Linear(->128) -> ReLU -> Dropout(0.3) -> Linear(128->1) -> Sigmoid (BCEWithLogitsLoss)",
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
            "class_weighting": "none -- train/val splits are already ~50/50 balanced",
            "total_time_s": round(s1_time + s2_time, 1),
        },
        "inference_time_ms_per_image": round(inference_time_ms_per_image, 3),
        "final_train_metrics": {
            "loss": final_train_m["loss"], "accuracy": final_train_m["accuracy"],
            "precision": final_train_m["precision"], "recall": final_train_m["recall"],
            "f1": final_train_m["f1"], "roc_auc": final_train_m["roc_auc"],
        },
        "final_validation_metrics": {
            "loss": final_val_m["loss"], "accuracy": final_val_m["accuracy"],
            "precision": final_val_m["precision"], "recall": final_val_m["recall"],
            "f1": final_val_m["f1"], "roc_auc": final_val_m["roc_auc"],
        },
        "train_val_gap": {
            "accuracy_gap": round(final_train_m["accuracy"] - final_val_m["accuracy"], 4),
            "f1_gap": round(final_train_m["f1"] - final_val_m["f1"], 4),
        },
    }
    with open(out_dir / "config" / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n=== DONE ===")
    print("Final train metrics:", json.dumps(config["final_train_metrics"], indent=2))
    print("Final val metrics:", json.dumps(config["final_validation_metrics"], indent=2))
    print("Train-val gap:", json.dumps(config["train_val_gap"], indent=2))
    print("TEST SET HAS NOT BEEN ACCESSED.")


if __name__ == "__main__":
    main()
