"""
Train the WESAD stress classifier: LightGBM binary classifier on wrist-only windowed
features (dataset/WESAD/wesad_wrist_features.csv), predicting stress vs not-stress
(baseline/amusement).

Evaluation: LEAVE-ONE-SUBJECT-OUT cross-validation (LOSO-CV), not a single train/val/test
split -- WESAD has only 15 subjects, too few for one held-out split to give a reliable
estimate (a single unlucky test subject could swing the number a lot). This is also the
same protocol Chun et al. 2021 uses for the closely related scratch-detection task, and
matches the subject-level-split principle already established for Stage A (WISDM) --
see docs/paper_review_adam_sensor_2026-08-26.md.

For each of the 15 folds: hold out one subject as test, hold out one more (rotating,
deterministic given seed) as validation for early stopping and threshold tuning, train on
the remaining 13. Report per-fold AUC/F1 plus the mean +/- std across folds (the standard
way LOSO-CV results are reported in this literature) and a pooled confusion matrix.

Also trains and saves a FINAL model on all 15 subjects for actual use in the Stage C
fusion pipeline (models/wesad_stress_lightgbm.txt) -- its decision threshold is the
median of the 15 per-fold thresholds, since a model trained on all subjects has no
independent validation subject left to tune one directly.
"""
import csv
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = Path(r"C:\Users\tishy\Documents\Honors")
FEATURES_CSV = ROOT / "dataset" / "WESAD" / "wesad_wrist_features.csv"
FOLD_RESULTS_CSV = ROOT / "dataset" / "WESAD" / "wesad_loso_fold_results.csv"
SEED = 42


def best_f1_threshold(probs, y):
    candidates = np.linspace(0.01, 0.99, 197)
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        pred = (probs >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def evaluate(probs, y, threshold):
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (preds == y).mean()
    return dict(acc=acc, tp=tp, tn=tn, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "seed": SEED,
    "learning_rate": 0.05,
    "num_leaves": 15,  # small: only ~13 subjects/few thousand windows per fold, keep it modest
    "is_unbalance": True,
}


def main():
    df = pd.read_csv(FEATURES_CSV)
    feat_cols = [c for c in df.columns if c not in ("subject_id", "label")]
    subjects = sorted(df["subject_id"].unique(), key=lambda s: int(s[1:]))
    print(f"Subjects: {len(subjects)} -> {subjects}")
    print(f"Windows: {len(df)}  Stress: {df['label'].sum()} ({100*df['label'].mean():.1f}%)")
    print(f"Features: {len(feat_cols)}")

    fold_rows = []
    pooled_probs, pooled_y = [], []

    for i, test_subj in enumerate(subjects):
        remaining = [s for s in subjects if s != test_subj]
        val_subj = remaining[i % len(remaining)]  # deterministic rotating choice
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
        pooled_probs.append(test_probs)
        pooled_y.append(y_test)

        print(f"[{test_subj}] val_subj={val_subj} thr={thr:.3f} "
              f"test_auc={test_auc:.4f} test_f1={result['f1']:.4f} test_acc={result['acc']:.4f}")

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(FOLD_RESULTS_CSV, index=False)

    print("\n=== LOSO-CV summary across 15 folds ===")
    print(f"AUC:       mean={fold_df['test_auc'].mean():.4f}  std={fold_df['test_auc'].std():.4f}")
    print(f"F1:        mean={fold_df['f1'].mean():.4f}  std={fold_df['f1'].std():.4f}")
    print(f"Accuracy:  mean={fold_df['acc'].mean():.4f}  std={fold_df['acc'].std():.4f}")
    print(f"Precision: mean={fold_df['precision'].mean():.4f}")
    print(f"Recall:    mean={fold_df['recall'].mean():.4f}")

    pooled_probs = np.concatenate(pooled_probs)
    pooled_y = np.concatenate(pooled_y)
    pooled_thr = float(fold_df["threshold"].median())
    pooled = evaluate(pooled_probs, pooled_y, pooled_thr)
    pooled_auc = roc_auc_score(pooled_y, pooled_probs)
    print(f"\n=== Pooled (all test folds concatenated, median threshold={pooled_thr:.3f}) ===")
    print(f"AUC={pooled_auc:.4f} Accuracy={pooled['acc']:.4f} "
          f"Precision={pooled['precision']:.4f} Recall={pooled['recall']:.4f} F1={pooled['f1']:.4f}")
    print(f"TN={pooled['tn']} FP={pooled['fp']} FN={pooled['fn']} TP={pooled['tp']}")

    # Final deployable model: train on ALL subjects. No held-out set left for early
    # stopping, so use the median best_iteration across the 15 CV folds instead --
    # a principled stand-in for the round count that would otherwise be tuned live.
    final_rounds = int(fold_df["best_iteration"].median())
    X_all, y_all = df[feat_cols].values, df["label"].values
    final_set = lgb.Dataset(X_all, label=y_all, feature_name=feat_cols)
    final_model = lgb.train(PARAMS, final_set, num_boost_round=final_rounds)
    final_model.save_model(str(ROOT / "models" / "wesad_stress_lightgbm.txt"))

    with open(ROOT / "models" / "wesad_stress_threshold.txt", "w") as f:
        f.write(str(pooled_thr))

    print(f"\nFinal model saved: models/wesad_stress_lightgbm.txt")
    print(f"Deployment threshold saved: models/wesad_stress_threshold.txt ({pooled_thr:.3f})")

    importances = sorted(zip(feat_cols, final_model.feature_importance(importance_type="gain")),
                          key=lambda x: -x[1])
    print("\nTop 15 features by gain (final model):")
    for name, imp in importances[:15]:
        print(f"  {name}: {imp:.1f}")


if __name__ == "__main__":
    main()
