"""
Shortcut/domain-gap diagnostic for external-validation manifests, mirroring the exact
methodology that originally caught the photographic-source shortcut in Section 5.1 of the
report (mean whole-image brightness gap between classes) and the suspiciously-close
CNN/colour-feature-LightGBM agreement that was the first red flag.

For a given manifest:
  1. Mean whole-image brightness (grayscale mean / 255, computed on the same 128x128 resize
     extract_color_features.py uses) per class, so the gap is directly comparable to the
     original dataset's known 0.15 vs 0.58 confound.
  2. Runs the existing curated_lightgbm_v3 colour-feature model (trained only on internal
     data, never fine-tuned here) on the manifest's images and reports its accuracy/F1
     alongside the CNN's -- if this simpler, more shortcut-prone model does unexpectedly
     well on data it's never seen, that's a signal the CNN's result might not be trustworthy
     either.
"""
import csv
import sys
from collections import defaultdict

import lightgbm as lgb
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(__file__.rsplit("\\", 1)[0]))
from extract_color_features import compute_spaces, feature_names
from paths import MODELS_DIR

RESIZE_TO = (128, 128)


def load_rgb01(path):
    img = Image.open(path).convert("RGB").resize(RESIZE_TO)
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr[..., 0], arr[..., 1], arr[..., 2]


def brightness(path):
    R, G, B = load_rgb01(path)
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    return float(gray.mean())


def color_features(path):
    R, G, B = load_rgb01(path)
    spaces = compute_spaces(R, G, B)
    feats = []
    for space_name in ["RGB", "NRGB", "YCbCr", "HSV", "HLS", "XYZ", "LAB", "LCH", "LUV",
                        "OPPONENT", "CMY", "YUV", "YIQ", "YDbDr", "YPbPr"]:
        for channel in spaces[space_name]:
            feats.append(float(np.mean(channel)))
            feats.append(float(np.std(channel)))
    return feats


def diagnose(manifest_path, dataset_label):
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"\n=== Shortcut diagnostic: {dataset_label} ===")

    # 1. Brightness gap
    by_label_brightness = defaultdict(list)
    for row in rows:
        try:
            b = brightness(row["path"])
        except Exception as e:
            continue
        by_label_brightness[int(row["label"])].append(b)

    mean_eczema = np.mean(by_label_brightness[1])
    mean_other = np.mean(by_label_brightness[0])
    print(f"Mean brightness -- Eczema: {mean_eczema:.4f}, Other: {mean_other:.4f}, "
          f"gap: {abs(mean_eczema - mean_other):.4f}")
    print("(Original dataset's known shortcut-confound gap was 0.15 vs 0.58 = 0.43)")

    # 2. Colour-feature LightGBM (trained only on internal data) on this manifest
    model_path = MODELS_DIR / "curated_lightgbm_v3.txt"
    model = lgb.Booster(model_file=str(model_path))
    names = feature_names()

    X, y = [], []
    for row in rows:
        try:
            feats = color_features(row["path"])
        except Exception:
            continue
        X.append(feats)
        y.append(int(row["label"]))
    X = np.array(X)
    y = np.array(y)

    probs = model.predict(X, num_iteration=model.best_iteration)
    preds = (probs >= 0.5).astype(int)
    acc = (preds == y).mean()
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    auc = roc_auc_score(y, probs)
    print(f"Colour-feature LightGBM (curated_lightgbm_v3, unmodified) on this manifest:")
    print(f"  accuracy={acc:.4f} auc={auc:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")

    return dict(mean_eczema_brightness=float(mean_eczema), mean_other_brightness=float(mean_other),
                lgbm_acc=float(acc), lgbm_auc=float(auc), lgbm_f1=float(f1))


if __name__ == "__main__":
    from paths import SKINDISEASE_DIR, SCIN_DIR

    diagnose(SKINDISEASE_DIR / "manifest_curated_v3_test.csv", "Internal test set (baseline)")
    diagnose(SCIN_DIR / "manifest_scin.csv", "SCIN (external)")
