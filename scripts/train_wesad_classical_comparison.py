"""
Part 2 of the architecture-comparison work (Stage A / stress): three classical models
compared against the deployed personal-baseline-calibrated LightGBM, on the same
52-feature wrist-wearable feature set (dataset/WESAD/wesad_wrist_features.csv).

TEST-SET ISOLATION: only wesad_comparison_split.TRAIN_SUBJECTS rows are ever loaded from
the features file -- TEST_SUBJECTS' rows are never read into memory by this script. Model
selection/monitoring uses a further subject-wise split of TRAIN_SUBJECTS
(INTERNAL_TRAIN_SUBJECTS / INTERNAL_VAL_SUBJECTS), never TEST_SUBJECTS.

REVISION 2026-09-18 (v2): applies the same discipline as the CNN-LSTM fix (v2) -- model
selection driven by validation AUC with early stopping, not a fixed round count picked in
advance. XGBoost and CatBoost both support this natively (eval_set + early_stopping_rounds)
and now use it: MAX_ROUNDS=500 as a ceiling, early_stopping_rounds=20, monitored on AUC.
Random Forest has no equivalent mechanism in scikit-learn's implementation (no staged/
incremental fit with a validation-triggered stop) -- left at a fixed N_ROUNDS=120 rather
than faking early stopping for it.
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import lightgbm as lgb

from paths import ROOT
from wesad_comparison_split import TRAIN_SUBJECTS, INTERNAL_TRAIN_SUBJECTS, INTERNAL_VAL_SUBJECTS

FEATURES_PATH = ROOT / "dataset" / "WESAD" / "wesad_wrist_features.csv"
OUT_DIR = ROOT / "experiments" / "wesad_stress_comparison"
N_ROUNDS = 120          # Random Forest only -- no early stopping available for it
MAX_BOOST_ROUNDS = 500  # ceiling for XGBoost/CatBoost/LightGBM; early stopping decides the
                         # actual round count used
EARLY_STOPPING_ROUNDS = 20
SEED = 42


def load_data():
    df = pd.read_csv(FEATURES_PATH)
    df = df[df["subject_id"].isin(TRAIN_SUBJECTS)].reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ("subject_id", "label", "condition")]
    return df, feature_cols


def split_train_val(df):
    train = df[df["subject_id"].isin(INTERNAL_TRAIN_SUBJECTS)]
    val = df[df["subject_id"].isin(INTERNAL_VAL_SUBJECTS)]
    return train, val


def evaluate(name, model, X_val, y_val, predict_proba_fn=None):
    probs = predict_proba_fn(model, X_val) if predict_proba_fn else model.predict_proba(X_val)[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_val, preds),
        "roc_auc": roc_auc_score(y_val, probs),
        "f1": f1_score(y_val, preds, zero_division=0),
    }
    print(f"{name}: acc={metrics['accuracy']:.4f} auc={metrics['roc_auc']:.4f} f1={metrics['f1']:.4f}")
    return metrics


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, feature_cols = load_data()
    train_df, val_df = split_train_val(df)
    print(f"Train rows: {len(train_df)} (subjects {INTERNAL_TRAIN_SUBJECTS})")
    print(f"Internal val rows: {len(val_df)} (subjects {INTERNAL_VAL_SUBJECTS})")
    print(f"TEST_SUBJECTS not loaded at any point in this script.\n")

    X_train, y_train = train_df[feature_cols].values, train_df["label"].values
    X_val, y_val = val_df[feature_cols].values, val_df["label"].values

    results = {}

    # --- XGBoost (v2: validation-AUC early stopping, not a fixed round count) ---
    import xgboost as xgb
    clf = xgb.XGBClassifier(n_estimators=MAX_BOOST_ROUNDS, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                             eval_metric="auc", early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"XGBoost stopped at round {clf.best_iteration} "
          f"(best val AUC {clf.best_score:.4f}, ceiling was {MAX_BOOST_ROUNDS})")
    results["xgboost"] = evaluate("XGBoost", clf, X_val, y_val)
    clf.save_model(OUT_DIR / "xgboost.json")

    # --- CatBoost (v2: same, native early stopping on AUC) ---
    from catboost import CatBoostClassifier
    clf = CatBoostClassifier(iterations=MAX_BOOST_ROUNDS, depth=4, learning_rate=0.05,
                              random_seed=SEED, verbose=False, eval_metric="AUC",
                              early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    clf.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    print(f"CatBoost stopped at round {clf.get_best_iteration()} "
          f"(best val AUC {clf.get_best_score()['validation']['AUC']:.4f}, "
          f"ceiling was {MAX_BOOST_ROUNDS})")
    results["catboost"] = evaluate("CatBoost", clf, X_val, y_val)
    clf.save_model(str(OUT_DIR / "catboost.cbm"))

    # --- Random Forest ---
    clf = RandomForestClassifier(n_estimators=N_ROUNDS, max_depth=6, random_state=SEED,
                                  n_jobs=-1)
    clf.fit(X_train, y_train)
    results["random_forest"] = evaluate("Random Forest", clf, X_val, y_val)
    import joblib
    joblib.dump(clf, OUT_DIR / "random_forest.joblib")

    # --- LightGBM (from-scratch, same protocol, for apples-to-apples against the deployed
    #     one -- NOT the deployed personal-baseline-calibrated model itself, which uses a
    #     different, LOSO, subject-calibration protocol documented in
    #     docs/wesad_calibration_significance_2026-09-16.md) ---
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    booster = lgb.train(
        {"objective": "binary", "metric": "auc", "max_depth": 4, "learning_rate": 0.05,
         "verbosity": -1, "seed": SEED},
        train_set, num_boost_round=MAX_BOOST_ROUNDS, valid_sets=[val_set],
        callbacks=[lgb.log_evaluation(0), lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    print(f"LightGBM stopped at round {booster.best_iteration} "
          f"(ceiling was {MAX_BOOST_ROUNDS})")
    probs = booster.predict(X_val, num_iteration=booster.best_iteration)
    results["lightgbm_uncalibrated"] = evaluate(
        "LightGBM (from scratch, same split, NOT the deployed calibrated model)",
        None, X_val, y_val, predict_proba_fn=lambda m, X: probs)
    booster.save_model(str(OUT_DIR / "lightgbm_uncalibrated.txt"))

    with open(OUT_DIR / "internal_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/internal_validation_results.json")
    print("These are INTERNAL VALIDATION numbers (3 held-out train-side subjects), not the "
          "final test-set comparison. TEST_SUBJECTS remain untouched.")


if __name__ == "__main__":
    main()
