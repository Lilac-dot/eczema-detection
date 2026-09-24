"""
Fine-tunes a curated-set Stage B checkpoint on the pooled skin-tone dataset
(SCIN + Fitzpatrick17k-C + DermaCon-IN; dataset/skintone/manifest_{train,val}.csv from
split_skintone.py). This is "option 2" of the skin-tone compression audit: start from
the model this project already has (experiments/<model>/checkpoints/best_overall.pt,
trained on the curated DermNet-style set) rather than from ImageNet.

  --init curated  (default) start from the curated checkpoint
  --init imagenet           ImageNet-only start, for comparison

Two stages, same shape as train_transfer_cnn.py:
  stage 1: head + the architecture's last blocks (MODEL_CONFIGS stage2_unfreeze), lr 1e-4
  stage 2: lr 1e-5, unfreezing either the whole network (--stage2-unfreeze all, default)
           or only the named blocks, e.g. --stage2-unfreeze stage3 stage4 conv5 for
           ShuffleNetV2: early generic layers (conv1, stage2) stay frozen, with their
           BatchNorm statistics in eval mode, which suits a small target dataset.
Checkpoint selection on validation MACRO AUC across sources (mean of per-source AUCs),
so a model can't win selection by doing well on the largest source only.

Source shortcut guard: sources have different eczema prevalence
(check_skintone_merge.py), so each training image is weighted so that every
source x label cell contributes equally to the loss -- the model gains nothing from
learning which source an image came from.

TEST-SET ISOLATION: manifest_test.csv is never opened here.

Writes experiments/skintone/<model>_<init>/{best.pt, history.json, config.json}.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from paths import ROOT, DATASET_DIR
from train_transfer_cnn import MODEL_CONFIGS, build_model, make_transforms, set_frozen_eval

SPLIT_DIR = DATASET_DIR / "skintone"
OUT_ROOT = ROOT / "experiments" / "skintone"
BATCH_SIZE = 32
SEED = 42
STAGE1_LR, STAGE2_LR = 1e-4, 1e-5
STAGE1_MAX_EPOCHS, STAGE2_MAX_EPOCHS = 20, 20
PATIENCE = 5


class ManifestDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = [str(ROOT / p) for p in df["path"]]
        self.labels = df["label"].astype(np.float32).values
        self.weights = df["weight"].astype(np.float32).values if "weight" in df else \
            np.ones(len(df), np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.transform(Image.open(self.paths[i]).convert("RGB"))
        return img, self.labels[i], self.weights[i]


def cell_weights(df):
    """Inverse-frequency weight per source x label cell, normalised to mean 1."""
    cell = df["source"] + "|" + df["label"].astype(str)
    w = 1.0 / cell.map(cell.value_counts())
    return w / w.mean()


def set_trainable(model, cfg, unfrozen):
    """unfrozen: list of block paths to train alongside the head, or None for everything."""
    for p in model.parameters():
        p.requires_grad = unfrozen is None
    if unfrozen is not None:
        for path in unfrozen + [cfg["final_attr"]]:
            for p in model.get_submodule(path).parameters():
                p.requires_grad = True


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs = []
    for x, _, _ in loader:
        probs.append(torch.sigmoid(model(x)).numpy().ravel())
    return np.concatenate(probs)


def macro_auc(df, probs):
    per = {}
    for s, idx in df.groupby("source").indices.items():
        y = df["label"].values[idx]
        if len(np.unique(y)) == 2:
            per[s] = roc_auc_score(y, probs[idx])
    return float(np.mean(list(per.values()))), per


def run_stage(model, cfg, stage, unfrozen, train_loader, val_loader, val_df, lr, max_epochs,
              out_dir, history, best):
    set_trainable(model, cfg, unfrozen)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                            weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(reduction="none")
    stale = 0
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        if unfrozen is not None:
            # frozen blocks' BatchNorm stays in eval mode, as in train_transfer_cnn.py
            set_frozen_eval(model, cfg, unfrozen)
        else:
            model.train()
        total, n = 0.0, 0
        for x, y, w in train_loader:
            loss = (crit(model(x).squeeze(1), y) * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        probs = predict(model, val_loader)
        m_auc, per = macro_auc(val_df, probs)
        history.append(dict(stage=stage, epoch=epoch, train_loss=total / n, val_macro_auc=m_auc,
                            val_auc_by_source=per, secs=round(time.time() - t0)))
        print(f"stage {stage} epoch {epoch}: loss {total / n:.4f}  val macro-AUC {m_auc:.4f} "
              f"{ {k: round(v, 3) for k, v in per.items()} }", flush=True)
        if m_auc > best["auc"]:
            best.update(auc=m_auc, stage=stage, epoch=epoch)
            torch.save(model.state_dict(), out_dir / "best.pt")
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="shufflenet_v2_x0_5", choices=list(MODEL_CONFIGS))
    ap.add_argument("--init", default="curated", choices=["curated", "imagenet"])
    ap.add_argument("--stage2-unfreeze", nargs="+", default=["all"],
                    help="blocks to unfreeze in stage 2, or 'all'")
    ap.add_argument("--tag", default="", help="suffix for the output folder")
    args = ap.parse_args()
    stage2_unfrozen = None if args.stage2_unfreeze == ["all"] else args.stage2_unfreeze
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg = MODEL_CONFIGS[args.model]
    out_dir = OUT_ROOT / f"{args.model}_{args.init}{'_' + args.tag if args.tag else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(SPLIT_DIR / "manifest_train.csv")
    val_df = pd.read_csv(SPLIT_DIR / "manifest_val.csv")
    train_df["weight"] = cell_weights(train_df)

    train_tf, eval_tf = make_transforms()
    train_loader = DataLoader(ManifestDataset(train_df, train_tf), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=6, persistent_workers=True)
    val_loader = DataLoader(ManifestDataset(val_df, eval_tf), batch_size=64, num_workers=6,
                            persistent_workers=True)

    model = build_model(args.model)
    if args.init == "curated":
        ckpt = ROOT / "experiments" / args.model / "checkpoints" / "best_overall.pt"
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to("cpu")

    start_auc, start_per = macro_auc(val_df, predict(model, val_loader))
    print(f"before fine-tuning: val macro-AUC {start_auc:.4f} {start_per}", flush=True)

    history, best = [], dict(auc=-1.0)
    run_stage(model, cfg, 1, cfg["stage2_unfreeze"], train_loader, val_loader, val_df,
              STAGE1_LR, STAGE1_MAX_EPOCHS, out_dir, history, best)
    model.load_state_dict(torch.load(out_dir / "best.pt"))
    run_stage(model, cfg, 2, stage2_unfrozen, train_loader, val_loader, val_df,
              STAGE2_LR, STAGE2_MAX_EPOCHS, out_dir, history, best)

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(dict(model=args.model, init=args.init, stage2_unfreeze=args.stage2_unfreeze,
                       seed=SEED, batch_size=BATCH_SIZE,
                       stage1_lr=STAGE1_LR, stage2_lr=STAGE2_LR, patience=PATIENCE,
                       n_train=len(train_df), n_val=len(val_df), test_manifest_used=False,
                       val_macro_auc_before=start_auc, val_auc_before_by_source=start_per,
                       best=best), f, indent=2)
    print(f"best: {best}")


if __name__ == "__main__":
    main()
