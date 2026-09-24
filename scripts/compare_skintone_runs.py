"""
Paired comparison of cross-validated skin-tone runs on the SAME held-out images.

Every run in experiments/skintone_cv/ writes out-of-fold predictions for the development
images it covered. Two runs are compared only on the images both scored (e.g. SCIN-only
vs pooled -> the SCIN development images), with a paired bootstrap over CASES
(split_group) for the AUC difference, overall and within each Fitzpatrick group.

Usage:
    python compare_skintone_runs.py <reference_run> <run> [<run> ...]
    e.g. python compare_skintone_runs.py resnet18_curated_scin_only resnet18_curated_pooled

Writes experiments/skintone_cv/comparison_<reference_run>.json.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from train_skintone_cv import OUT_ROOT

N_BOOT = 2000


def paired(m, a, b, seed=0):
    y = m["label"].values
    if len(np.unique(y)) < 2:
        return None
    g = m["split_group"].values
    uniq = np.unique(g)
    idx = {k: np.flatnonzero(g == k) for k in uniq}
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(N_BOOT):
        i = np.concatenate([idx[k] for k in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[i])) == 2:
            diffs.append(roc_auc_score(y[i], m[b].values[i]) - roc_auc_score(y[i], m[a].values[i]))
    diffs = np.array(diffs)
    return dict(auc_ref=round(float(roc_auc_score(y, m[a])), 4),
                auc_run=round(float(roc_auc_score(y, m[b])), 4),
                diff=round(float(roc_auc_score(y, m[b]) - roc_auc_score(y, m[a])), 4),
                ci_low=round(float(np.percentile(diffs, 2.5)), 4),
                ci_high=round(float(np.percentile(diffs, 97.5)), 4),
                p_run_not_better=round(float((diffs <= 0).mean()), 4),
                n_cases=int(len(uniq)))


def main():
    ref, runs = sys.argv[1], sys.argv[2:]
    a = pd.read_csv(OUT_ROOT / ref / "oof_predictions.csv")
    out = {}
    for run in runs:
        b = pd.read_csv(OUT_ROOT / run / "oof_predictions.csv")
        m = a.merge(b[["path", "prob"]], on="path", suffixes=("_ref", "_run"))
        res = {}
        for src, d in m.groupby("source"):
            res[src] = {"all": paired(d, "prob_ref", "prob_run")}
            for fg, dg in d.dropna(subset=["fst_group"]).groupby("fst_group"):
                res[src][fg] = paired(dg, "prob_ref", "prob_run")
        out[run] = res
        print(f"\n{run} vs {ref} (n images compared: {len(m)})")
        for src, r in res.items():
            for k, v in r.items():
                if v:
                    print(f"  {src:15s} {k:7s} ref {v['auc_ref']:.3f}  run {v['auc_run']:.3f}  "
                          f"diff {v['diff']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]  "
                          f"({v['n_cases']} cases)")
    with open(OUT_ROOT / f"comparison_{ref}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
