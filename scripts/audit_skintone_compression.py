"""
Skin-tone compression audit: does compressing an eczema classifier cost darker-skinned
patients more than lighter-skinned ones?

Runs on the per-source cross-validated models (train_skintone_cv.py, one model per
source -- pooling hurt SCIN and Fitzpatrick17k, see compare_skintone_runs.py). For every
fold, the fold's FP32 checkpoint and its compressed variants score that fold's OUTER
images only (never seen in training or checkpoint selection), so every image gets exactly
one out-of-fold prediction per variant. The frozen test split is never opened.

Variants (reusing the compression study's code, edge_ai_extended_analysis.py and
qnnpack_relu6_layout_fix_2026_09_23.py):
  fp32                 the fine-tuned model
  int8                 static INT8 PTQ, qnnpack (ARM -- the phone backend; this Mac has
                       no fbgemm), default qconfig, 256 calibration images drawn at random
                       from the fold's inner-train split, ReLU6 channels-last fix applied
  int8_calib_light     same, calibration images only from the source's lightest
                       Fitzpatrick group (I-II; III-IV for DermaCon-IN, which has no I-II)
  int8_calib_dark      same, calibration images only from FST V-VI
  prune30/50/70        one-shot global L1 unstructured magnitude pruning of every
                       Conv2d/Linear weight, no fine-tuning
  prune30_int8         30% pruning followed by INT8 (default calibration)

Calibration images are sampled with a fixed seed per fold; when a skin-tone group has
fewer than 256 inner-train images, all of them are used (the count is recorded).

Analysis (per model x source x variant): AUC overall and per Fitzpatrick group, the
change vs FP32 with a paired case-level bootstrap, the prediction flip rate vs FP32 at
threshold 0.5 (Hooker et al. 2020's "compression identified exemplars" view -- a
compressed model can keep AUC while changing individual decisions), and the key
disparity statistic: delta-AUC(darkest group) minus delta-AUC(lightest group), with a
bootstrap CI. Negative = compression hurts darker skin more.

Usage:
    python audit_skintone_compression.py --models resnet18 shufflenet_v2_x0_5 ...
Writes experiments/skintone_audit/{predictions.csv, audit_summary.json}.
"""
import argparse
import copy
import json
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

import edge_ai_extended_analysis as ext
from paths import ROOT
from qnnpack_relu6_layout_fix_2026_09_23 import fix_layout
from skintone_models import load_start_model, make_skintone_transforms
from train_skintone_cv import OUT_ROOT, SEED, ManifestDataset, load_dev, make_folds

warnings.filterwarnings("ignore")
torch.backends.quantized.engine = "qnnpack"
ext.QUANT_BACKEND = "qnnpack"

AUDIT_DIR = ROOT / "experiments" / "skintone_audit"
SOURCES = ["SCIN", "DermaCon-IN", "Fitzpatrick17k"]
N_CALIB = 256
PRUNE_LEVELS = [0.3, 0.5, 0.7]
N_BOOT = 1000
FST_ORDER = ["I-II", "III-IV", "V-VI"]


@torch.no_grad()
def predict(model, df):
    _, eval_tf = make_skintone_transforms()
    model.eval()
    probs = []
    for x, _, _ in DataLoader(ManifestDataset(df, eval_tf), batch_size=64, num_workers=4):
        probs.append(torch.sigmoid(model(x)).numpy().ravel())
    return np.concatenate(probs)


def calib_loader(df, seed):
    _, eval_tf = make_skintone_transforms()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(df), min(N_CALIB, len(df)), replace=False)
    return DataLoader(Subset(ManifestDataset(df.reset_index(drop=True), eval_tf), idx.tolist()),
                      batch_size=32, shuffle=False)


def int8(model, loader):
    # quantize_static iterates (images, labels) pairs; our dataset also yields a weight
    batches = [(x, y) for x, y, _ in loader]
    q = ext.quantize_static(copy.deepcopy(model).eval(), batches, n_calib_batches=len(batches))
    return fix_layout(q)[0]


def pruned(model, sparsity):
    return ext.prune_to_sparsity(copy.deepcopy(model), sparsity)


def variants_for_fold(model, inner_train, source, k):
    light = "III-IV" if source == "DermaCon-IN" else "I-II"
    calib_sets = {
        "int8": inner_train,
        "int8_calib_light": inner_train[inner_train["fst_group"] == light],
        "int8_calib_dark": inner_train[inner_train["fst_group"] == "V-VI"],
    }
    out = {"fp32": (model, None)}
    for name, df in calib_sets.items():
        out[name] = (int8(model, calib_loader(df, SEED + k)), min(N_CALIB, len(df)))
    for s in PRUNE_LEVELS:
        out[f"prune{int(s * 100)}"] = (pruned(model, s), None)
    out["prune30_int8"] = (int8(pruned(model, 0.3), calib_loader(inner_train, SEED + k)), N_CALIB)
    return out


# ---------------------------------------------------------------- analysis
def case_index(df):
    g = df["split_group"].values
    uniq = np.unique(g)
    return uniq, {u: np.flatnonzero(g == u) for u in uniq}


def auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def analyse(wide, variant):
    """wide: one row per image with columns fp32 and <variant>. Returns per-group AUCs,
    deltas, flip rates and the dark-minus-light delta, all with case-bootstrap CIs."""
    groups = [g for g in FST_ORDER if (wide["fst_group"] == g).any()]
    lightest, darkest = groups[0], groups[-1]
    uniq, idx = case_index(wide)
    y, a, b = wide["label"].values, wide["fp32"].values, wide[variant].values
    fg = wide["fst_group"].values
    flips = (a >= 0.5) != (b >= 0.5)

    def stats(i):
        r = {"all": auc(y[i], b[i]) - auc(y[i], a[i])}
        for g in groups:
            m = i[fg[i] == g]
            r[g] = auc(y[m], b[m]) - auc(y[m], a[m])
        r["dark_minus_light"] = r[darkest] - r[lightest]
        return r

    point = stats(np.arange(len(wide)))
    rng = np.random.RandomState(SEED)
    boots = [stats(np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))]))
             for _ in range(N_BOOT)]
    res = {"groups_compared": [lightest, darkest]}
    for key in point:
        vals = np.array([bt[key] for bt in boots], dtype=float)
        vals = vals[~np.isnan(vals)]
        res[f"delta_auc_{key}"] = dict(point=round(float(point[key]), 4),
                                       ci_low=round(float(np.percentile(vals, 2.5)), 4),
                                       ci_high=round(float(np.percentile(vals, 97.5)), 4))
    res["auc"] = {"all": round(float(auc(y, b)), 4),
                  **{g: round(float(auc(y[fg == g], b[fg == g])), 4) for g in groups}}
    res["flip_rate"] = {"all": round(float(flips.mean()), 4),
                        **{g: round(float(flips[fg == g].mean()), 4) for g in groups}}
    return res


def summarize(preds):
    summary = {}
    for (model, source), d in preds.groupby(["model", "source"]):
        wide = d.pivot_table(index=["path", "split_group", "label", "fst_group"], columns="variant",
                             values="prob").reset_index()
        wide = wide.dropna(subset=["fst_group"])
        summary.setdefault(model, {})[source] = {
            "fp32_auc": {"all": round(float(auc(wide["label"], wide["fp32"])), 4),
                         **{g: round(float(auc(w["label"], w["fp32"])), 4)
                            for g, w in wide.groupby("fst_group")}},
            "n_cases_by_fst": wide.groupby("fst_group")["split_group"].nunique().to_dict(),
            "variants": {v: analyse(wide, v) for v in wide.columns
                         if v not in ("path", "split_group", "label", "fst_group", "fp32")},
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18"])
    ap.add_argument("--sources", nargs="+", default=SOURCES)
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(6)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = AUDIT_DIR / "predictions.csv"
    preds = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()

    if not args.analyse_only:
        for model_name in args.models:
            for source in args.sources:
                run = OUT_ROOT / f"{model_name}_curated_{source}_only"
                if not (run / "summary.json").exists():
                    print(f"skip {model_name} {source}: CV run not finished")
                    continue
                if len(preds) and ((preds["model"] == model_name) & (preds["source"] == source)).any():
                    print(f"skip {model_name} {source}: already audited")
                    continue
                calib_n = {}
                for k, inner_train, _, outer in make_folds(load_dev([source])):
                    model = load_start_model(model_name, "curated")
                    model.load_state_dict(torch.load(run / f"fold{k}_best.pt", map_location="cpu"))
                    model.eval()
                    for variant, (m, n_cal) in variants_for_fold(model, inner_train, source, k).items():
                        part = outer[["path", "split_group", "label", "fst_group"]].copy()
                        part["model"], part["source"], part["fold"], part["variant"] = \
                            model_name, source, k, variant
                        part["prob"] = predict(m, outer)
                        preds = pd.concat([preds, part], ignore_index=True)
                        calib_n[f"fold{k}_{variant}"] = n_cal
                    print(f"{model_name} {source} fold {k} done", flush=True)
                preds.to_csv(pred_path, index=False)
                with open(AUDIT_DIR / f"calib_counts_{model_name}_{source}.json", "w") as f:
                    json.dump(calib_n, f, indent=2)

    summary = summarize(preds)
    with open(AUDIT_DIR / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    for model_name, by_src in summary.items():
        for source, s in by_src.items():
            print(f"\n{model_name} / {source}  FP32 AUC {s['fp32_auc']}  cases {s['n_cases_by_fst']}")
            for v, r in s["variants"].items():
                dml = r["delta_auc_dark_minus_light"]
                print(f"  {v:18s} dAUC all {r['delta_auc_all']['point']:+.3f}  "
                      + "  ".join(f"{g} {r[f'delta_auc_{g}']['point']:+.3f}"
                                  for g in FST_ORDER if f"delta_auc_{g}" in r)
                      + f"  | dark-light {dml['point']:+.3f} [{dml['ci_low']:+.3f}, {dml['ci_high']:+.3f}]"
                      + f"  | flips {r['flip_rate']['all']:.3f}")


if __name__ == "__main__":
    main()
