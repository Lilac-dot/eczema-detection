"""
Is it safe to pool SCIN, Fitzpatrick17k-C and DermaCon-IN into one fine-tuning set?
Runs on the DEVELOPMENT images only (dataset/skintone/manifest_train.csv + manifest_val.csv
from split_skintone.py -- the frozen test split is never read here) and the fine-tuning
STARTING checkpoint (the curated-set model, skintone_models.load_start_model; ResNet18 by
default, --model to change). Nothing here trains the eczema model.

Checks, in the order they matter for "can these be merged":
  1. Composition: class mix, eczema prevalence and skin-type mix per source. A source
     whose eczema prevalence differs a lot from the others lets a model score well by
     recognising the SOURCE rather than the disease (a shortcut, the same failure mode
     as the original Eczema-vs-Normal dataset -- docs/dataset_usability_check_2026-08-26.md).
  2. Source-only shortcut ceiling: AUC of predicting eczema from source identity
     alone. Near 0.5 = source carries no label information.
  3. Image statistics per source (resolution, brightness, mean colour).
  4. Domain separability in the starting model's own feature space: 5-fold origin
     classifier accuracy and pairwise proxy A-distance (Ben-David et al. 2010, same
     measure as domain_shift_analysis.py), including the curated training set.
  5. Is the label signal shared across sources? A linear eczema probe trained on one
     source's embeddings and tested on each other source (group-aware split). If
     transfer is near chance, the sources disagree about what "eczema" looks like and
     pooling them mixes incompatible label definitions -- the actual merge blocker.
  6. Zero-shot performance of the starting checkpoint per source x skin-type group.

Writes dataset/skintone/merge_check.json and prints a summary.
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import argparse

from paths import ROOT, DATASET_DIR
from skintone_models import load_start_model, make_skintone_transforms
from clean_skintone_manifest import CURATED_MANIFEST, local_path

OUT_DIR = DATASET_DIR / "skintone"
N_CURATED = 1000  # curated-set sample for the domain comparison (train split only)
SEED = 42
N_BOOT = 1000


class PathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def embed_and_score(model, paths):
    """Returns (1024-d backbone embeddings, eczema probabilities) for each path."""
    _, eval_tf = make_skintone_transforms()
    head = model.fc
    model.fc = nn.Identity()
    model.eval()
    feats, probs = [], []
    for x in DataLoader(PathDataset(paths, eval_tf), batch_size=64, num_workers=4):
        f = model(x)
        feats.append(f.numpy())
        probs.append(torch.sigmoid(head(f)).numpy().ravel())
    model.fc = head
    return np.concatenate(feats), np.concatenate(probs)


def auc_ci(y, p, seed=SEED):
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2:
        return None
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], p[idx]))
    return dict(auc=round(roc_auc_score(y, p), 4), ci_low=round(np.percentile(boots, 2.5), 4),
                ci_high=round(np.percentile(boots, 97.5), 4), n=int(len(y)), n_pos=int(y.sum()))


def origin_accuracy(X, src):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    pred = cross_val_predict(clf, X, src, cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    return float((pred == src).mean())


def proxy_a_distance(Xa, Xb):
    """2(1 - 2*err) of a domain classifier; 0 = indistinguishable, 2 = fully separable."""
    n = min(len(Xa), len(Xb))
    rng = np.random.RandomState(SEED)
    Xa, Xb = Xa[rng.choice(len(Xa), n, False)], Xb[rng.choice(len(Xb), n, False)]
    acc = origin_accuracy(np.vstack([Xa, Xb]), np.array([0] * n + [1] * n))
    return round(2 * (1 - 2 * (1 - acc)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18")
    args = ap.parse_args()
    df = pd.concat([pd.read_csv(OUT_DIR / "manifest_train.csv"),
                    pd.read_csv(OUT_DIR / "manifest_val.csv")], ignore_index=True)
    out = {"model": args.model, "n_dev_images": len(df)}

    # 1. Composition
    out["class_mix"] = pd.crosstab(df["source"], df["disease_class"]).to_dict("index")
    out["eczema_prevalence"] = df.groupby("source")["label"].mean().round(3).to_dict()
    out["fst_mix"] = pd.crosstab(df["source"], df["fst_group"].fillna("unknown"),
                                 normalize="index").round(3).to_dict("index")
    out["eczema_by_fst"] = pd.crosstab([df["source"], df["fst_group"].fillna("unknown")],
                                       df["label"]).rename(columns={0: "other", 1: "eczema"}) \
        .reset_index().to_dict("records")

    # 2. Source-only shortcut ceiling
    out["source_only_auc"] = round(roc_auc_score(df["label"], df["source"].map(
        out["eczema_prevalence"])), 4)

    # 3. Image statistics
    out["image_stats"] = df.groupby("source")[
        ["width", "height", "brightness", "mean_r", "mean_g", "mean_b"]].median().round(1) \
        .to_dict("index")

    # 4-6 need embeddings from the starting checkpoint
    model = load_start_model(args.model, "curated")
    X, p0 = embed_and_score(model, [str(ROOT / p) for p in df["path_small"]])

    cur = pd.read_csv(CURATED_MANIFEST.with_name("manifest_curated_v3_train.csv"))
    cur = cur.sample(min(N_CURATED, len(cur)), random_state=SEED)
    Xc, _ = embed_and_score(model, [str(local_path(p)) for p in cur["path"]])

    srcs = sorted(df["source"].unique())
    Xs = {s: X[(df["source"] == s).values] for s in srcs}
    Xs["Curated (start)"] = Xc
    allX = np.vstack(list(Xs.values()))
    allsrc = np.concatenate([[s] * len(v) for s, v in Xs.items()])
    out["origin_classifier_accuracy"] = round(origin_accuracy(allX, allsrc), 4)
    out["origin_chance"] = round(pd.Series(allsrc).value_counts(normalize=True).max(), 4)
    names = list(Xs)
    out["proxy_a_distance"] = {f"{a} vs {b}": proxy_a_distance(Xs[a], Xs[b])
                               for i, a in enumerate(names) for b in names[i + 1:]}

    # 5. Cross-source label-probe transfer
    probe = {}
    for s in srcs:
        m = (df["source"] == s).values
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
        # within-source AUC: group-aware 5-fold so same-case images never straddle folds
        pw = cross_val_predict(clf, X[m], df.loc[m, "label"], groups=df.loc[m, "split_group"],
                               cv=GroupKFold(5), method="predict_proba")[:, 1]
        row = {f"within {s}": round(roc_auc_score(df.loc[m, "label"], pw), 4)}
        clf.fit(X[m], df.loc[m, "label"])
        for t in srcs:
            if t != s:
                mt = (df["source"] == t).values
                row[f"-> {t}"] = round(roc_auc_score(df.loc[mt, "label"],
                                                     clf.predict_proba(X[mt])[:, 1]), 4)
        probe[f"train on {s}"] = row
    out["label_probe_transfer_auc"] = probe

    # 6. Zero-shot performance of the starting checkpoint
    zs = {}
    df["p0"] = p0
    for s in srcs:
        d = df[df["source"] == s]
        zs[s] = {"all": auc_ci(d["label"], d["p0"])}
        for g, dg in d.dropna(subset=["fst_group"]).groupby("fst_group"):
            zs[s][g] = auc_ci(dg["label"], dg["p0"])
    out["zero_shot_start_checkpoint"] = zs

    with open(OUT_DIR / f"merge_check_{args.model}.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
