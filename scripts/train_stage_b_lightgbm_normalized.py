"""
Same as train_stage_b_lightgbm.py, but trained on dataset/color_features_normalized.csv
(brightness/exposure-normalized colour features, see extract_color_features_normalized.py)
instead of the raw colour features. Uses the identical train/val/test split so this is a
direct, controlled comparison against the original 95.18% test accuracy: if brightness
was the shortcut, this number should drop noticeably.
"""
import csv
from pathlib import Path

import numpy as np
import lightgbm as lgb

ROOT = Path(r"C:\Users\tishy\Documents\Honors\dataset")
FEATURES_CSV = ROOT / "color_features_normalized.csv"
CLASS_TO_IDX = {"Normal": 0, "Eczema": 1}


def load_features():
    with open(FEATURES_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        feat_names = header[2:]
        by_path = {}
        for row in reader:
            path, cls = row[0], row[1]
            feats = np.array([float(v) for v in row[2:]], dtype=np.float64)
            by_path[path] = (feats, CLASS_TO_IDX[cls])
    return by_path, feat_names


def load_split(split_name, by_path):
    manifest = ROOT / f"manifest_{split_name}.csv"
    X, y = [], []
    with open(manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            feats, label = by_path[row["path"]]
            X.append(feats)
            y.append(label)
    return np.array(X), np.array(y)


def main():
    by_path, feat_names = load_features()
    X_train, y_train = load_split("train", by_path)
    X_val, y_val = load_split("val", by_path)
    X_test, y_test = load_split("test", by_path)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feat_names)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=feat_names, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "seed": 42,
        "learning_rate": 0.05,
        "num_leaves": 31,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=25)],
    )

    print(f"\nBest iteration: {model.best_iteration}")

    def evaluate(X, y, name):
        preds = (model.predict(X, num_iteration=model.best_iteration) >= 0.5).astype(int)
        acc = (preds == y).mean()
        tp = int(((preds == 1) & (y == 1)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"\n[{name}] n={len(y)} accuracy={acc:.4f}")
        print(f"  Normal: TN={tn} FP={fp}")
        print(f"  Eczema: FN={fn} TP={tp}")
        print(f"  precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
        return acc, tn, fp, fn, tp, precision, recall, f1

    evaluate(X_val, y_val, "validation")
    evaluate(X_test, y_test, "test")

    importances = sorted(zip(feat_names, model.feature_importance(importance_type="gain")),
                          key=lambda x: -x[1])
    print("\nTop 10 features by gain:")
    for name, imp in importances[:10]:
        print(f"  {name}: {imp:.1f}")

    model.save_model(str(Path(r"C:\Users\tishy\Documents\Honors\models\stage_b_lightgbm_normalized.txt")))
    print("\nModel saved to models/stage_b_lightgbm_normalized.txt")


if __name__ == "__main__":
    main()
