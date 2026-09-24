"""
Deep Feature Reweighting (Kirichenko, Izmailov & Wilson, ICML 2022) on the pooled
ResNet18 CV run: keep each fold's fine-tuned backbone (experiments/skintone_cv/
resnet18_curated_pooled/fold<k>_best.pt) and retrain ONLY the last layer, as a logistic
regression on the 512-d penultimate features, using a BALANCED subset of that fold's
inner-validation set (never used to train the backbone -- the DFR "held-out reweighting
set" protocol). Balance = equal eczema/other within every source x Fitzpatrick-group cell
(train_skintone_cv.balance_subsample).

The regularisation strength C is chosen by group-aware 5-fold CV inside that balanced
set, on the WORST per-source AUC. Outer folds are rebuilt with train_skintone_cv.make_folds
(identical to the backbone run) and are only used for the reported predictions.

Writes experiments/skintone_cv/resnet18_curated_pooled_dfr/{oof_predictions.csv, summary.json}.
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from paths import ROOT
from skintone_models import load_start_model, make_skintone_transforms
from train_skintone_cv import (OUT_ROOT, SEED, balance_subsample, load_dev, loader, make_folds,
                               summarize)

BACKBONE_RUN = OUT_ROOT / "resnet18_curated_pooled"
OUT_DIR = OUT_ROOT / "resnet18_curated_pooled_dfr"
CS = [0.001, 0.01, 0.1, 1.0]


@torch.no_grad()
def features(model, df):
    _, eval_tf = make_skintone_transforms()
    model.eval()
    out = []
    for x, _, _ in loader(df, eval_tf, False):
        out.append(model(x).numpy())
    return np.concatenate(out)


def worst_source_auc(df, p):
    aucs = [roc_auc_score(d["label"], p[d.index]) for _, d in df.groupby("source")
            if d["label"].nunique() == 2]
    return min(aucs)


def pick_c(X, df):
    scores = {}
    for c in CS:
        p = np.zeros(len(df))
        for tr, te in GroupKFold(5).split(X, df["label"], df["split_group"]):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
            clf.fit(X[tr], df["label"].values[tr])
            p[te] = clf.predict_proba(X[te])[:, 1]
        scores[c] = worst_source_auc(df, p)
    return max(scores, key=scores.get), scores


def main():
    torch.set_num_threads(4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = load_dev()
    parts, per_fold = [], []
    for k, _, inner_val, outer in make_folds(dev):
        model = load_start_model("resnet18", "curated")
        model.load_state_dict(torch.load(BACKBONE_RUN / f"fold{k}_best.pt", map_location="cpu"))
        head = model.fc
        model.fc = nn.Identity()

        bal = balance_subsample(inner_val, SEED + k)
        Xb, Xo = features(model, bal), features(model, outer)
        c, scores = pick_c(Xb, bal)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
        clf.fit(Xb, bal["label"])
        p = clf.predict_proba(Xo)[:, 1]
        with torch.no_grad():
            p_before = torch.sigmoid(head(torch.from_numpy(Xo))).numpy().ravel()

        per = {s: round(float(roc_auc_score(d["label"], p[d.index])), 4)
               for s, d in outer.groupby("source")}
        per_fold.append(dict(fold=k, outer_macro_auc=round(float(np.mean(list(per.values()))), 4),
                             outer_auc_by_source=per, C=c, inner_cv_worst_auc=scores,
                             n_reweight_set=len(bal)))
        print(f"fold {k}: reweighting set {len(bal)} images, C={c}, outer {per}", flush=True)
        part = outer[["path", "source", "split_group", "label", "disease_class", "fst", "fst_group", "mst"]].copy()
        part["fold"], part["prob"], part["prob_start"] = k, p, p_before  # prob_start = pooled head
        parts.append(part)

    oof = pd.concat(parts, ignore_index=True)
    oof.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    summary = summarize(oof, per_fold)
    summary["config"] = dict(method="DFR last-layer retraining", backbone_run=str(BACKBONE_RUN.relative_to(ROOT)),
                             reweighting_set="balanced inner-val", Cs=CS, selection="worst-source AUC",
                             note="prob_start in oof = original pooled head, for comparison")
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["pooled_oof"], indent=2))


if __name__ == "__main__":
    main()
