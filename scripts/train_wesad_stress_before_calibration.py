"""
Reconstructs the PRE-calibration LOSO-CV result for paired statistical comparison
against the deployed personal-baseline-calibrated model (dataset/WESAD/wesad_loso_fold_results.csv,
mean AUC 0.9405 -- see docs/wesad_personal_baseline_calibration_2026-09-12.md).

wesad_wrist_features.csv already contains BOTH the absolute per-window features
(EDA_mean, TEMP_slope, ...) and their personal-baseline-relative "_rel" counterparts
(add_personal_baseline_features() in build_wesad_features.py computes both from the
same underlying windows -- nothing was recomputed or re-extracted for this script).
"Before calibration" is reconstructed by training on ONLY the absolute columns, i.e.
exactly the feature set the original (pre-fix) model used -- same data, same LOSO
protocol, same LightGBM code path as train_wesad_stress.py, the single difference being
which columns are excluded from feat_cols. This is not a new experiment design, it's the
ablation this project's own doc (Section "Update 2026-09-12") describes narratively but
never re-ran to produce a saved, subject-matched CSV for formal paired testing.

Writes dataset/WESAD/wesad_loso_fold_results_before_calibration.csv -- does NOT touch
wesad_loso_fold_results.csv (the current "after" file) or any deployed model.
"""
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import numpy as np

from paths import WESAD_DIR
from loso_report import print_reliability_summary
from train_wesad_stress import PARAMS, SEED, best_f1_threshold, evaluate

FEATURES_CSV = WESAD_DIR / "wesad_wrist_features.csv"
OUT_CSV = WESAD_DIR / "wesad_loso_fold_results_before_calibration.csv"


def main():
    df = pd.read_csv(FEATURES_CSV)
    # Only the absolute columns -- excludes every "_rel" column, reproducing the
    # pre-calibration feature set exactly (see module docstring).
    feat_cols = [c for c in df.columns
                 if c not in ("subject_id", "label", "condition") and not c.endswith("_rel")]
    subjects = sorted(df["subject_id"].unique(), key=lambda s: int(s[1:]))
    print(f"Subjects: {len(subjects)}  Features (absolute only): {len(feat_cols)}")

    fold_rows = []
    for i, test_subj in enumerate(subjects):
        remaining = [s for s in subjects if s != test_subj]
        val_subj = remaining[i % len(remaining)]
        train_subjs = [s for s in remaining if s != val_subj]

        train_df = df[df["subject_id"].isin(train_subjs)]
        val_df = df[df["subject_id"] == val_subj]
        test_df = df[df["subject_id"] == test_subj]

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
        print(f"[{test_subj}] test_auc={test_auc:.4f} test_f1={result['f1']:.4f}")

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT_CSV, index=False)
    print(f"\nMean AUC: {fold_df['test_auc'].mean():.4f}  Mean F1: {fold_df['f1'].mean():.4f}")
    print_reliability_summary(fold_df)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
