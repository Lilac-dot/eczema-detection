"""
The ONE script allowed to load wesad_comparison_split.TEST_SUBJECTS' data for the Stage A
(stress) model comparison -- same two-gate pattern as
scripts/eval_transfer_cnn_finalists.py on the image side.

GATE 1: refuses to run unless all 4 trained models exist on disk.
GATE 2: refuses to overwrite a previous result file unless --force is passed.

IMPORTANT ASYMMETRY, stated plainly rather than glossed over: the 4 models were trained on
two DIFFERENT sample representations of the same test subjects --
  - XGBoost, CatBoost, Random Forest, LightGBM(bonus): 139 rows from
    wesad_wrist_features.csv (52 hand-engineered features per row)
  - CNN-LSTM: 566 raw BVP windows from wesad_raw_windows.npz (different windowing scheme,
    not the same rows)
These are NOT the same test instances, so CNN-LSTM's predictions cannot be paired
row-for-row against the other four's. This script therefore:
  1. Runs full paired significance testing (McNemar, paired bootstrap, Holm correction)
     among the four tabular-feature models, which DO share identical test rows.
  2. Reports CNN-LSTM's test metrics independently, with its own (unpaired) bootstrap CI,
     and does not force a McNemar/paired comparison that would require fabricating a
     pairing that doesn't exist.
"""
import argparse
import datetime
import json
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2
from scipy.signal import butter, filtfilt
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, precision_score,
                              recall_score)

from paths import ROOT
from wesad_comparison_split import TEST_SUBJECTS
from train_wesad_lstm import CNNLSTM, bandpass_bvp, BVP_FS

FEATURES_PATH = ROOT / "dataset" / "WESAD" / "wesad_wrist_features.csv"
RAW_PATH = ROOT / "dataset" / "WESAD" / "wesad_raw_windows.npz"
MODEL_DIR = ROOT / "experiments" / "wesad_stress_comparison"
TODAY = datetime.date.today().isoformat()
OUT_JSON = ROOT / f"wesad_stress_final_comparison_{TODAY}.json"
N_BOOTSTRAP = 2000
SEED = 42

TABULAR_MODELS = ["xgboost", "catboost", "random_forest", "lightgbm_uncalibrated"]
REQUIRED_FILES = {
    "xgboost": MODEL_DIR / "xgboost.json",
    "catboost": MODEL_DIR / "catboost.cbm",
    "random_forest": MODEL_DIR / "random_forest.joblib",
    "lightgbm_uncalibrated": MODEL_DIR / "lightgbm_uncalibrated.txt",
    "cnn_lstm": MODEL_DIR / "cnn_lstm_v2.pt",
}


def check_all_trained():
    return [name for name, path in REQUIRED_FILES.items() if not path.exists()]


def metrics_from_probs(y_true, y_prob, threshold=0.5):
    pred = (y_prob >= threshold).astype(int)
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def bootstrap_ci_auc(y_true, y_prob, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    vals = []
    for _ in range(n):
        idx = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar_test(y_true, pred_a, pred_b):
    correct_a, correct_b = (pred_a == y_true), (pred_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "statistic": float(statistic), "p_value": float(chi2.sf(statistic, df=1))}


def paired_bootstrap_diff(y_true, prob_a, prob_b, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    observed = roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)
    diffs = []
    for _ in range(n):
        idx = rng.randint(0, n_samples, n_samples)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        diffs.append(roc_auc_score(yt, prob_a[idx]) - roc_auc_score(yt, prob_b[idx]))
    diffs = np.array(diffs)
    ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
    p = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"observed_diff": float(observed), "ci": ci, "p_value": p}


def holm_correction(pvalues):
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running_max = {}, 0.0
    for rank, (label, p_raw) in enumerate(items):
        adjusted = min((m - rank) * p_raw, 1.0)
        running_max = max(running_max, adjusted)
        out[label] = {"p_raw": p_raw, "p_holm": running_max, "significant": running_max < 0.05}
    return out


def load_tabular_test():
    df = pd.read_csv(FEATURES_PATH)
    df = df[df["subject_id"].isin(TEST_SUBJECTS)].reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ("subject_id", "label", "condition")]
    return df[feature_cols].values, df["label"].values


def load_cnn_lstm_test():
    d = np.load(RAW_PATH, allow_pickle=True)
    mask = np.isin(d["subject"], TEST_SUBJECTS)
    bvp = bandpass_bvp(d["BVP"][mask, :, 0], fs=BVP_FS)
    bvp = (bvp - bvp.mean(axis=1, keepdims=True)) / (bvp.std(axis=1, keepdims=True) + 1e-8)
    return bvp, d["label"][mask].astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    missing = check_all_trained()
    if missing and not args.force:
        print("REFUSING TO RUN: the following models have no saved checkpoint yet:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)
    if OUT_JSON.exists() and not args.force:
        print(f"REFUSING TO RUN: {OUT_JSON} already exists. This is meant to be a single, "
              f"final test-set read. Pass --force and document why if you need to redo it.")
        sys.exit(1)

    print(f"Gate passed. Opening TEST_SUBJECTS data now: {TEST_SUBJECTS}\n")

    # --- tabular models (paired, share identical test rows) ---
    X_test, y_test = load_tabular_test()
    print(f"Tabular test rows: {len(y_test)}\n")

    probs = {}

    import xgboost as xgb
    clf = xgb.XGBClassifier()
    clf.load_model(MODEL_DIR / "xgboost.json")
    probs["xgboost"] = clf.predict_proba(X_test)[:, 1]

    from catboost import CatBoostClassifier
    clf = CatBoostClassifier()
    clf.load_model(str(MODEL_DIR / "catboost.cbm"))
    probs["catboost"] = clf.predict_proba(X_test)[:, 1]

    import joblib
    clf = joblib.load(MODEL_DIR / "random_forest.joblib")
    probs["random_forest"] = clf.predict_proba(X_test)[:, 1]

    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(MODEL_DIR / "lightgbm_uncalibrated.txt"))
    probs["lightgbm_uncalibrated"] = booster.predict(X_test)

    tabular_metrics = {}
    for name in TABULAR_MODELS:
        m = metrics_from_probs(y_test, probs[name])
        lo, hi = bootstrap_ci_auc(y_test, probs[name])
        m["auc_95ci"] = [lo, hi]
        tabular_metrics[name] = m
        print(f"{name}: acc={m['accuracy']:.3f} auc={m['roc_auc']:.3f} "
              f"(95% CI [{lo:.3f}, {hi:.3f}]) f1={m['f1']:.3f}")

    print("\n=== Pairwise (tabular models only -- same test rows) ===")
    pairwise, raw_p = {}, {}
    for i in range(len(TABULAR_MODELS)):
        for j in range(i + 1, len(TABULAR_MODELS)):
            a, b = TABULAR_MODELS[i], TABULAR_MODELS[j]
            label = f"{a}_vs_{b}"
            pred_a = (probs[a] >= 0.5).astype(int)
            pred_b = (probs[b] >= 0.5).astype(int)
            mcnemar = mcnemar_test(y_test, pred_a, pred_b)
            auc_diff = paired_bootstrap_diff(y_test, probs[a], probs[b])
            pairwise[label] = {"mcnemar": mcnemar, "auc_diff": auc_diff}
            raw_p[label] = mcnemar["p_value"]
            print(f"{label}: mcnemar_p={mcnemar['p_value']:.4f}  "
                  f"auc_diff={auc_diff['observed_diff']:+.4f} (p={auc_diff['p_value']:.4f})")
    holm = holm_correction(raw_p)

    # --- CNN-LSTM (different windowing -- reported standalone, not paired) ---
    print("\n=== CNN-LSTM (different test-window structure -- see module docstring; "
          "reported standalone, not paired against the tabular models) ===")
    X_lstm, y_lstm = load_cnn_lstm_test()
    print(f"CNN-LSTM test windows: {len(y_lstm)}")
    model = CNNLSTM()
    model.load_state_dict(torch.load(MODEL_DIR / "cnn_lstm_v2.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_lstm).unsqueeze(1))
        lstm_probs = torch.sigmoid(logits).numpy().ravel()
    lstm_metrics = metrics_from_probs(y_lstm, lstm_probs)
    lo, hi = bootstrap_ci_auc(y_lstm, lstm_probs)
    lstm_metrics["auc_95ci"] = [lo, hi]
    print(f"cnn_lstm: acc={lstm_metrics['accuracy']:.3f} auc={lstm_metrics['roc_auc']:.3f} "
          f"(95% CI [{lo:.3f}, {hi:.3f}]) f1={lstm_metrics['f1']:.3f}")

    out = {
        "date": TODAY,
        "test_subjects": TEST_SUBJECTS,
        "tabular_test_n": len(y_test),
        "cnn_lstm_test_n": len(y_lstm),
        "tabular_metrics": tabular_metrics,
        "tabular_pairwise": pairwise,
        "tabular_holm_correction": holm,
        "cnn_lstm_metrics": lstm_metrics,
        "note": "CNN-LSTM not included in pairwise/Holm testing -- different test-window "
                "structure than the tabular models (566 raw windows vs 139 feature rows "
                "for the same 4 subjects), so predictions cannot be paired row-for-row.",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_JSON}")
    print("This was a single, official test-set read. Do not rerun.")


if __name__ == "__main__":
    main()
