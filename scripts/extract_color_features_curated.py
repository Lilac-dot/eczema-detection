"""
Same 90 colour features as extract_color_features.py, computed for the curated
Eczema-vs-other-disease subset (SkinDisease/manifest_curated.csv).

Writes SkinDisease/color_features_curated.csv (path, disease_class, label, 90 feature columns).
"""
import csv
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from extract_color_features import compute_spaces, feature_names

ROOT = Path(r"C:\Users\tishy\Documents\Honors\SkinDisease")
MANIFEST = ROOT / "manifest_curated.csv"
OUT_CSV = ROOT / "color_features_curated.csv"
RESIZE_TO = (128, 128)


def load_rgb01(path):
    img = Image.open(path).convert("RGB").resize(RESIZE_TO)
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr[..., 0], arr[..., 1], arr[..., 2]


def extract(path):
    R, G, B = load_rgb01(path)
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

    print(f"Extracting {len(names)} colour features from {len(reader)} images...")
    for i, row in enumerate(reader):
        path = row["path"]
        try:
            feats = extract(path)
        except Exception as e:
            print(f"  skip {path}: {e}")
            continue
        rows.append([path, row["disease_class"], row["label"]] + feats)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(reader)} done")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "disease_class", "label"] + names)
        w.writerows(rows)

    print(f"Saved {len(rows)} feature rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
