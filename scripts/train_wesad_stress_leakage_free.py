"""
Leakage-free variant of the personal-baseline-calibrated WESAD LOSO-CV pipeline.

LEAKAGE FOUND (see docs/wesad_calibration_deep_dive_2026-09-16.md Section 6 for full
detail): build_wesad_features.py's add_personal_baseline_features() computes each
subject's baseline profile as the mean of ALL of that subject's condition==1 (baseline)
windows, then subtracts that same profile from those same windows to make their `_rel`
features. Since baseline windows are themselves part of the evaluated set (label=0
includes both condition==1 and condition==3), every baseline window's `_rel` value is
computed using a profile that includes the window's own contribution -- a mild
self-referential leakage (diluted ~1/19 per subject, since each subject has ~18-19
baseline windows, but present for every subject, train and test alike).

Fix: for each TEST subject in each LOSO fold, split their baseline-condition windows
in half by index order (first half = calibration-only set, used ONLY to compute that
subject's baseline profile; second half = evaluation set). The evaluated set for that
subject becomes: second-half baseline windows + ALL amusement windows + ALL stress
windows (all `_rel`-transformed using the calibration-half-derived profile only).
Training subjects are unaffected -- they are never evaluated, so using their own full
baseline for their own `_rel` features is not leakage for them (mirrors real deployment:
a new wearer's calibration recording is separate from what gets monitored afterward,
which is exactly what this script now does for the test subject specifically).

Writes dataset/WESAD/wesad_loso_fold_results_leakage_free.csv -- does NOT touch
wesad_loso_fold_results.csv (the original, leakage-present "after" file) or any
deployed model. Both are kept side by side deliberately per the user's instruction.
"""
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from paths import WESAD_DIR
from loso_report import print_reliability_summary
from train_wesad_stress import PARAMS, SEED, best_f1_threshold, evaluate

FEATURES_CSV = WESAD_DIR / "wesad_wrist_features.csv"
# Split-seed is separate from the model-training SEED so repeated runs with different
# calibration/evaluation half-splits can be compared (Section 7 robustness check) without
# also changing LightGBM's own randomness.
SPLIT_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else SEED
OUT_CSV = WESAD_DIR / f"wesad_loso_fold_results_leakage_free_split{SPLIT_SEED}.csv"

RAW_FEAT_COLS = None  # set in main() once columns are known


def rebuild_rel_for_test_subject(df, test_subj, raw_cols, rng):
    """Returns a copy of test_subj's rows with `_rel` columns recomputed from a
    baseline profile built ONLY from a random half of that subject's own baseline
    (condition==1) windows -- the other half plus all amusement/stress windows form
    the evaluated set."""
    subj_df = df[df["subject_id"] == test_subj].copy()
    baseline_idx = subj_df.index[subj_df["condition"] == 1].to_numpy().copy()
    rng.shuffle(baseline_idx)
    half = len(baseline_idx) // 2
    calib_idx, eval_baseline_idx = baseline_idx[:half], baseline_idx[half:]

    calib_profile = subj_df.loc[calib_idx, raw_cols].mean()

    eval_idx = np.concatenate([
        eval_baseline_idx,
        subj_df.index[subj_df["condition"] != 1].to_numpy(),
    ])
    eval_df = subj_df.loc[eval_idx].copy()
    for c in raw_cols:
        eval_df[f"{c}_rel"] = eval_df[c] - calib_profile[c]
    return eval_df


def main():
    df = pd.read_csv(FEATURES_CSV)
    raw_cols = [c for c in df.columns
                if c not in ("subject_id", "label", "condition") and not c.endswith("_rel")]
    feat_cols = raw_cols + [f"{c}_rel" for c in raw_cols]
    subjects = sorted(df["subject_id"].unique(), key=lambda s: int(s[1:]))
    rng = np.random.RandomState(SPLIT_SEED)

    fold_rows = []
    for i, test_subj in enumerate(subjects):
        remaining = [s for s in subjects if s != test_subj]
        val_subj = remaining[i % len(remaining)]
        train_subjs = [s for s in remaining if s != val_subj]

        # Training and validation subjects: unchanged (their own full baseline is not
        # leakage for themselves -- see module docstring).
        train_df = df[df["subject_id"].isin(train_subjs)]
        val_df = df[df["subject_id"] == val_subj]
        # Test subject: leakage-free half-split reconstruction.
        test_df = rebuild_rel_for_test_subject(df, test_subj, raw_cols, rng)

        X_train, y_train = train_df[feat_cols].values, train_df["label"].values
        X_val, y_val = val_df[feat_cols].values, val_df["label"].values
        X_test, y_test = test_df[feat_cols].values, test_df["label"].values

        train_set = lgb.Dataset(X_train, label=y_train, feature_name=feat_cols)
        val_set = lgb.Dataset(X_val, label=y_val, feature_name=feat_cols, reference=train_set)

        model = lgb.train(
            PARAMS, train_set, num_boost_round=300,
            valid_sets=[val_set], valid_names=["val"],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )

        val_probs = model.predict(X_val, num_iteration=model.best_iteration)
        test_probs = model.predict(X_test, num_iteration=model.best_iteration)

        thr, val_f1 = best_f1_threshold(val_probs, y_val)
        test_auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else float("nan")
        result = evaluate(test_probs, y_test, thr)
        result.update(subject=test_subj, val_subject=val_subj, threshold=thr,
                       val_f1=val_f1, test_auc=test_auc, n_test=len(y_test),
                       best_iteration=model.best_iteration)
        fold_rows.append(result)
        print(f"[{test_subj}] n_test={len(y_test)} (halved baseline) test_auc={test_auc:.4f} "
              f"test_f1={result['f1']:.4f}")

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT_CSV, index=False)
    print(f"\nMean AUC: {fold_df['test_auc'].mean():.4f}  Mean F1: {fold_df['f1'].mean():.4f}")
    print_reliability_summary(fold_df)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
