"""
5-fold cross-validated fine-tuning for the skin-tone compression audit.

Data: the DEVELOPMENT set = dataset/skintone/manifest_train.csv + manifest_val.csv
(split_skintone.py). manifest_test.csv is the frozen test split and is never opened here.

Folds: StratifiedGroupKFold(5) over the development set -- groups are split_group (same
SCIN case / DermaCon-IN subject / merged near-duplicates never straddle folds), strata are
source x label x Fitzpatrick group. For each fold k:
  - outer fold k is held out and ONLY used for the reported out-of-fold predictions;
  - the other 4 folds are split again (group-aware, ~1/6 of groups) into inner-train and
    inner-val; inner-val alone drives checkpoint selection and early stopping.
So no image is ever both used to pick a checkpoint and to score it.

Model: skintone_models.load_start_model (default: curated Stage B ResNet18, "option 2" --
start from what the project already has). Two stages:
  stage 1: head + last block (ResNet18: layer4), lr 1e-4
  stage 2: head + last two blocks (layer3, layer4), lr 1e-5 -- early generic layers stay
           frozen with BatchNorm in eval mode (small target data; full unfreezing didn't
           help in the SCIN-only ShuffleNet runs).
Loss: BCE, each image weighted so every source x label cell contributes equally (guards
against learning source identity as an eczema shortcut). Selection: inner-val macro AUC
(mean of per-source AUCs). Images: aspect-preserving resize + crop
(skintone_models.make_skintone_transforms) from the 320-px cache.

Options added after the pooled run hurt SCIN (literature:
Compton et al. MLHC 2023; Idrissi et al. CLeaR 2022; Byrd & Lipton ICML 2019):
  --balance subsample  randomly drop images from inner-TRAIN so that, within every
                       source x Fitzpatrick-group cell, eczema and other are equal
                       (loss reweighting alone fades once a deep net fits its training
                       data, subsampling doesn't). Outer folds are never subsampled.
  --select worst       pick checkpoints by the WORST per-source inner-val AUC instead of
                       the mean, so easy gains on one source can't hide losses on another.

Writes experiments/skintone_cv/<model>_<init>[_tag]/:
  fold<k>_best.pt, oof_predictions.csv, history.json, summary.json
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
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from paths import ROOT, DATASET_DIR
from skintone_models import BLOCKS, HEAD, load_start_model, make_skintone_transforms

SPLIT_DIR = DATASET_DIR / "skintone"
OUT_ROOT = ROOT / "experiments" / "skintone_cv"
SEED = 42
N_FOLDS = 5
BATCH_SIZE = 32
STAGES = [  # (blocks to unfreeze besides the head, lr, max epochs): roughly the last ~20%
    # of each backbone in stage 1, the last ~40% in stage 2 (early layers always frozen)
    ({"resnet18": ["layer4"],
      "shufflenet_v2_x0_5": ["stage4", "conv5"], "shufflenet_v2_x1_0": ["stage4", "conv5"],
      "squeezenet1_1": ["features.11", "features.12"]}, 1e-4, 15),
    ({"resnet18": ["layer3", "layer4"],
      "shufflenet_v2_x0_5": ["stage3", "stage4", "conv5"],
      "shufflenet_v2_x1_0": ["stage3", "stage4", "conv5"],
      "squeezenet1_1": ["features.9", "features.10", "features.11", "features.12"]}, 1e-5, 15),
]
PATIENCE = 4
N_BOOT = 1000


class ManifestDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = [str(ROOT / p) for p in df["path_small"]]
        self.labels = df["label"].astype(np.float32).values
        self.weights = df["weight"].astype(np.float32).values if "weight" in df else \
            np.ones(len(df), np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.transform(Image.open(self.paths[i]).convert("RGB"))
        return img, self.labels[i], self.weights[i]


def strata(df):
    return df["source"] + "|" + df["label"].astype(str) + "|" + df["fst_group"].fillna("unk")


def cell_weights(df):
    cell = df["source"] + "|" + df["label"].astype(str)
    w = 1.0 / cell.map(cell.value_counts())
    return w / w.mean()


def balance_subsample(df, seed):
    """Within each source x Fitzpatrick-group cell, downsample the larger label to the
    size of the smaller one."""
    rng = np.random.RandomState(seed)
    keep = []
    for _, cell in df.groupby([df["source"], df["fst_group"].fillna("unk")]):
        pos, neg = cell.index[cell["label"] == 1], cell.index[cell["label"] == 0]
        n = min(len(pos), len(neg))
        keep += list(rng.choice(pos, n, replace=False)) + list(rng.choice(neg, n, replace=False))
    return df.loc[sorted(keep)].reset_index(drop=True)


def make_folds(dev):
    """Yields (k, inner_train, inner_val, outer) exactly as used for training, so other
    scripts (e.g. dfr_skintone.py) can rebuild the same splits."""
    outer = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for k, (tr_idx, te_idx) in enumerate(outer.split(dev, strata(dev), dev["split_group"])):
        train_k = dev.iloc[tr_idx].reset_index(drop=True)
        test_k = dev.iloc[te_idx].reset_index(drop=True)
        inner = StratifiedGroupKFold(6, shuffle=True, random_state=SEED + k)
        itr, iva = next(inner.split(train_k, strata(train_k), train_k["split_group"]))
        yield (k, train_k.iloc[itr].reset_index(drop=True), train_k.iloc[iva].reset_index(drop=True),
               test_k)


def load_dev(sources=None):
    dev = pd.concat([pd.read_csv(SPLIT_DIR / "manifest_train.csv"),
                     pd.read_csv(SPLIT_DIR / "manifest_val.csv")], ignore_index=True)
    if sources:
        dev = dev[dev["source"].isin(sources)].reset_index(drop=True)
    return dev


def macro_auc(df, probs):
    per = {}
    for s, idx in df.groupby("source").indices.items():
        y = df["label"].values[idx]
        if len(np.unique(y)) == 2:
            per[s] = float(roc_auc_score(y, probs[idx]))
    return float(np.mean(list(per.values()))), per


def set_trainable(model, name, unfrozen):
    for p in model.parameters():
        p.requires_grad = False
    for path in unfrozen + [HEAD[name]]:
        for p in model.get_submodule(path).parameters():
            p.requires_grad = True


def train_mode(model, name, unfrozen):
    """Train mode for unfrozen blocks + head only; frozen blocks' BatchNorm stays in eval."""
    model.train()
    for block in BLOCKS[name]:
        if block not in unfrozen:
            model.get_submodule(block).eval()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for x, _, _ in loader:
        out.append(torch.sigmoid(model(x.to(device))).float().cpu().numpy().ravel())
    return np.concatenate(out)


def loader(df, tf, shuffle):
    return DataLoader(ManifestDataset(df, tf), batch_size=BATCH_SIZE if shuffle else 64,
                      shuffle=shuffle, num_workers=6, persistent_workers=True)


def train_fold(args, k, inner_train, inner_val, device, out_dir, history):
    train_tf, eval_tf = make_skintone_transforms()
    inner_train = inner_train.copy()
    if args.balance == "subsample":
        n0 = len(inner_train)
        inner_train = balance_subsample(inner_train, SEED + k)
        print(f"fold {k}: balanced inner-train {n0} -> {len(inner_train)} images", flush=True)
    inner_train["weight"] = cell_weights(inner_train)
    tl, vl = loader(inner_train, train_tf, True), loader(inner_val, eval_tf, False)

    model = load_start_model(args.model, args.init).to(device)
    crit = nn.BCEWithLogitsLoss(reduction="none")
    best = dict(auc=-1.0)
    ckpt = out_dir / f"fold{k}_best.pt"
    for stage, (blocks, lr, max_epochs) in enumerate(STAGES, 1):
        if stage > 1:
            model.load_state_dict(torch.load(ckpt, map_location=device))
        unfrozen = blocks[args.model]
        set_trainable(model, args.model, unfrozen)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                                weight_decay=1e-4)
        stale = 0
        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            train_mode(model, args.model, unfrozen)
            total, n = 0.0, 0
            for x, y, w in tl:
                x, y, w = x.to(device), y.to(device), w.to(device)
                loss = (crit(model(x).squeeze(1), y) * w).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(y)
                n += len(y)
            m_auc, per = macro_auc(inner_val, predict(model, vl, device))
            score = min(per.values()) if args.select == "worst" else m_auc
            history.append(dict(fold=k, stage=stage, epoch=epoch, train_loss=total / n,
                                inner_val_macro_auc=m_auc, inner_val_auc_by_source=per,
                                secs=round(time.time() - t0)))
            print(f"fold {k} stage {stage} epoch {epoch}: loss {total / n:.4f} inner-val "
                  f"macro-AUC {m_auc:.4f} { {s: round(v, 3) for s, v in per.items()} }", flush=True)
            if score > best["auc"]:
                best = dict(auc=score, stage=stage, epoch=epoch, select=args.select)
                torch.save(model.state_dict(), ckpt)
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model, best


def boot_auc(df, col="prob", seed=SEED):
    """AUC with a 95% CI from bootstrapping CASES (split_group), not images, since images
    of one case are correlated."""
    y, p = df["label"].values, df[col].values
    if len(np.unique(y)) < 2:
        return None
    groups = df["split_group"].values
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(N_BOOT):
        idx = np.concatenate([idx_by_group[g] for g in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], p[idx]))
    return dict(auc=round(float(roc_auc_score(y, p)), 4),
                ci_low=round(float(np.percentile(boots, 2.5)), 4),
                ci_high=round(float(np.percentile(boots, 97.5)), 4),
                n_images=int(len(y)), n_cases=int(len(uniq)), n_pos=int(y.sum()))


def summarize(oof, per_fold):
    s = {"per_fold": per_fold,
         "fold_macro_auc_mean": round(float(np.mean([f["outer_macro_auc"] for f in per_fold])), 4),
         "fold_macro_auc_sd": round(float(np.std([f["outer_macro_auc"] for f in per_fold])), 4),
         "pooled_oof": {"all": boot_auc(oof), "zero_shot_all": boot_auc(oof, "prob_start")}}
    for src, d in oof.groupby("source"):
        s["pooled_oof"][src] = {"all": boot_auc(d), "zero_shot": boot_auc(d, "prob_start")}
        for g, dg in d.dropna(subset=["fst_group"]).groupby("fst_group"):
            s["pooled_oof"][src][g] = boot_auc(dg)
    s["pooled_oof"]["by_fst_all_sources"] = {
        g: boot_auc(dg) for g, dg in oof.dropna(subset=["fst_group"]).groupby("fst_group")}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18", choices=list(BLOCKS))
    ap.add_argument("--init", default="curated", choices=["curated", "imagenet"])
    ap.add_argument("--sources", nargs="+", default=None, help="restrict to these sources")
    ap.add_argument("--balance", default="none", choices=["none", "subsample"])
    ap.add_argument("--select", default="macro", choices=["macro", "worst"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    dev = load_dev(args.sources)
    out_dir = OUT_ROOT / f"{args.model}_{args.init}{'_' + args.tag if args.tag else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device {device}; development set {len(dev)} images, "
          f"{dev['split_group'].nunique()} groups; sources {sorted(dev['source'].unique())}")

    history, per_fold, oof_parts = [], [], []
    _, eval_tf = make_skintone_transforms()
    for k, inner_train, inner_val, test_k in make_folds(dev):
        print(f"\n=== fold {k}: inner-train {len(inner_train)}, inner-val {len(inner_val)}, "
              f"outer {len(test_k)} images ===", flush=True)

        tel = loader(test_k, eval_tf, False)
        start = load_start_model(args.model, args.init).to(device)
        p_start = predict(start, tel, device)
        model, best = train_fold(args, k, inner_train, inner_val, device, out_dir, history)
        p = predict(model, tel, device)
        m_auc, per = macro_auc(test_k, p)
        per_fold.append(dict(fold=k, outer_macro_auc=round(m_auc, 4),
                             outer_auc_by_source={s: round(v, 4) for s, v in per.items()},
                             best=best, n_outer=len(test_k)))
        print(f"fold {k} OUTER macro-AUC {m_auc:.4f} {per}", flush=True)
        part = test_k[["path", "source", "split_group", "label", "disease_class", "fst", "fst_group", "mst"]].copy()
        part["fold"], part["prob"], part["prob_start"] = k, p, p_start
        oof_parts.append(part)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)
    summary = summarize(oof, per_fold)
    summary["config"] = dict(model=args.model, init=args.init, sources=args.sources, seed=SEED,
                             balance=args.balance, select=args.select,
                             stages=[(b[args.model], lr, e) for b, lr, e in STAGES],
                             patience=PATIENCE, test_manifest_used=False, device=str(device))
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["pooled_oof"], indent=2))
    print(f"per-fold macro-AUC mean {summary['fold_macro_auc_mean']} sd {summary['fold_macro_auc_sd']}")


if __name__ == "__main__":
    main()
