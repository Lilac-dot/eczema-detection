"""
Same 90 colour features as extract_color_features.py, but with an exposure/brightness
normalization step applied to each image FIRST: each RGB channel is independently
rescaled to a fixed target mean and standard deviation across the whole image, before
any colour-space conversion.

Purpose: this is a direct test of the shortcut-learning finding in
Progress_Log_2026-08-26.docx section 8 (MEAN_XYZ_2 / overall brightness was the single
most important LightGBM feature, and differs ~4x between classes). Forcing every image
to the same global mean/std brightness makes that specific cue impossible to use --
MEAN_RGB_x / STD_RGB_x become (by construction) identical across every image and
therefore useless to a classifier. If the model trained on these normalized features
still scores near the original ~95%, the shortcut wasn't just brightness; if accuracy
drops sharply, brightness/exposure was doing most of the work.

This does NOT fix composition, framing, or watermarks -- only global brightness/contrast.

Writes dataset/color_features_normalized.csv (same schema as color_features.csv).
"""
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from extract_color_features import compute_spaces, feature_names

ROOT = Path(r"C:\Users\tishy\Documents\Honors\dataset")
MANIFEST = ROOT / "manifest_clean.csv"
OUT_CSV = ROOT / "color_features_normalized.csv"
RESIZE_TO = (128, 128)
EPS = 1e-8

TARGET_MEAN = 0.5
TARGET_STD = 0.15


def load_rgb01_normalized(path):
    img = Image.open(path).convert("RGB").resize(RESIZE_TO)
    arr = np.asarray(img, dtype=np.float64) / 255.0
    for c in range(3):
        ch = arr[..., c]
        mean, std = ch.mean(), ch.std() + EPS
        arr[..., c] = np.clip((ch - mean) / std * TARGET_STD + TARGET_MEAN, 0.0, 1.0)
    return arr[..., 0], arr[..., 1], arr[..., 2]


def extract(path):
    R, G, B = load_rgb01_normalized(path)
    spaces = compute_spaces(R, G, B)
    feats = []
    for space_name in ["RGB", "NRGB", "YCbCr", "HSV", "HLS", "XYZ", "LAB", "LCH", "LUV",
                        "OPPONENT", "CMY", "YUV", "YIQ", "YDbDr", "YPbPr"]:
        for channel in spaces[space_name]:
            feats.append(float(np.mean(channel)))
            feats.append(float(np.std(channel)))
    return feats


def main():
    names = feature_names()
    rows = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    print(f"Extracting {len(names)} brightness-normalized colour features from {len(reader)} images...")
    for i, row in enumerate(reader):
        path = row["path"]
        try:
            feats = extract(path)
        except Exception as e:
            print(f"  skip {path}: {e}")
            continue
        rows.append([path, row["class"]] + feats)
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(reader)} done")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class"] + names)
        w.writerows(rows)

    print(f"Saved {len(rows)} feature rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
