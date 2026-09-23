"""
Ablation: is the personal-baseline calibration benefit driven by EDA specifically, or
is it genuinely multimodal? Reruns the same LOSO-CV LightGBM pipeline as
train_wesad_stress.py (full feature set, mean AUC 0.9405) restricted to three feature
subsets: EDA-only (+ EDA_rel), TEMP-only (+ TEMP_rel), and EDA+TEMP combined (+ both
_rel sets). BVP/ACC are excluded per the user's scope (fast to add, but not requested).

Writes dataset/WESAD/wesad_ablation_results.csv (one row per modality subset, per-fold
detail not kept -- summary only, as requested). Does not touch the main fold-results CSVs.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from paths import WESAD_DIR
from train_wesad_stress import PARAMS, best_f1_threshold, evaluate

FEATURES_CSV = WESAD_DIR / "wesad_wrist_features.csv"
OUT_CSV = WESAD_DIR / "wesad_ablation_results.csv"

SUBSETS = {
    "EDA_only": ["EDA"],
    "TEMP_only": ["TEMP"],
    "EDA_TEMP": ["EDA", "TEMP"],
}


def run_loso(df, feat_cols):
    subjects = sorted(df["subject_id"].unique(), key=lambda s: int(s[1:]))
    aucs, f1s = [], []
    for i, test_subj in enumerate(subjects):
        remaining = [s for s in subjects if s != test_subj]
        val_subj = remaining[i % len(remaining)]
        train_subjs = [s for s in remaining if s != val_subj]

        train_df, val_df, test_df = (df[df["subject_id"].isin(train_subjs)],
                                      df[df["subject_id"] == val_subj],
                                      df[df["subject_id"] == test_subj])
        X_train, y_train = train_df[feat_cols].values, train_df["label"].values
        X_val, y_val = val_df[feat_cols].values, val_df["label"].values
        X_test, y_test = test_df[feat_cols].values, test_df["label"].values

        train_set = lgb.Dataset(X_train, label=y_train, feature_name=feat_cols)
        val_set = lgb.Dataset(X_val, label=y_val, feature_name=feat_cols, reference=train_set)
        model = lgb.train(PARAMS, train_set, num_boost_round=300, valid_sets=[val_set],
                           valid_names=["val"],
                           callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])

        val_probs = model.predict(X_val, num_iteration=model.best_iteration)
        test_probs = model.predict(X_test, num_iteration=model.best_iteration)
        thr, _ = best_f1_threshold(val_probs, y_val)
        auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else float("nan")
        f1 = evaluate(test_probs, y_test, thr)["f1"]
        aucs.append(auc); f1s.append(f1)
    return np.array(aucs), np.array(f1s)


def main():
    df = pd.read_csv(FEATURES_CSV)
    rows = []
    for name, mods in SUBSETS.items():
        cols = []
        for m in mods:
            cols += [c for c in df.columns if c.startswith(m + "_") and not c.endswith("_rel")]
        feat_cols = cols + [f"{c}_rel" for c in cols]
        aucs, f1s = run_loso(df, feat_cols)
        rows.append(dict(subset=name, n_features=len(feat_cols),
                          mean_auc=aucs.mean(), std_auc=aucs.std(),
                          mean_f1=f1s.mean(), std_f1=f1s.std()))
        print(f"{name}: n_feat={len(feat_cols)} mean_auc={aucs.mean():.4f} mean_f1={f1s.mean():.4f}")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nFull-feature-set reference (all modalities, from train_wesad_stress.py): mean_auc=0.9405")
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
