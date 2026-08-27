"""
Extract the 90 hand-engineered colour features described in Maulana et al. (2024),
"Enhanced Prediction of Atopic Dermatitis Severity Using Advanced Machine Learning
Techniques", for every image in the cleaned Eczema/Normal manifest.

For each image, computes mean + standard deviation of each channel across 15 colour
representations: RGB, normalized RGB, YCbCr, HSV, HLS, CIE XYZ, CIE LAB, CIE LUV,
CIE LCH, Opponent, CMY, YUV, YIQ, YDbDr, YPbPr.
15 spaces x 3 channels x 2 stats = 90 features.

Writes dataset/color_features.csv (path, class, 90 named feature columns).
"""
import csv
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(r"C:\Users\tishy\Documents\Honors\dataset")
MANIFEST = ROOT / "manifest_clean.csv"
OUT_CSV = ROOT / "color_features.csv"
RESIZE_TO = (128, 128)
EPS = 1e-8


def load_rgb01(path):
    img = Image.open(path).convert("RGB").resize(RESIZE_TO)
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr[..., 0], arr[..., 1], arr[..., 2]  # R, G, B in [0,1]


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def compute_spaces(R, G, B):
    spaces = {}

    spaces["RGB"] = (R, G, B)

    s = R + G + B + EPS
    spaces["NRGB"] = (R / s, G / s, B / s)

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 0.5
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 0.5
    spaces["YCbCr"] = (Y, Cb, Cr)

    maxc = np.maximum(np.maximum(R, G), B)
    minc = np.minimum(np.minimum(R, G), B)
    diff = maxc - minc + EPS
    Vv = maxc
    Sv = np.where(maxc > 0, diff / (maxc + EPS), 0.0)
    Hv = np.zeros_like(R)
    r_is_max = maxc == R
    g_is_max = (maxc == G) & (~r_is_max)
    b_is_max = (~r_is_max) & (~g_is_max)
    Hv = np.where(r_is_max, (60 * ((G - B) / diff) + 360) % 360, Hv)
    Hv = np.where(g_is_max, (60 * ((B - R) / diff) + 120), Hv)
    Hv = np.where(b_is_max, (60 * ((R - G) / diff) + 240), Hv)
    spaces["HSV"] = (Hv, Sv, Vv)

    Lh = (maxc + minc) / 2.0
    Sh = np.where(
        diff < EPS, 0.0,
        np.where(Lh > 0.5, diff / (2.0 - maxc - minc + EPS), diff / (maxc + minc + EPS)),
    )
    spaces["HLS"] = (Hv, Lh, Sh)

    Rl, Gl, Bl = srgb_to_linear(R), srgb_to_linear(G), srgb_to_linear(B)
    X = 0.4124564 * Rl + 0.3575761 * Gl + 0.1804375 * Bl
    Yx = 0.2126729 * Rl + 0.7151522 * Gl + 0.0721750 * Bl
    Z = 0.0193339 * Rl + 0.1191920 * Gl + 0.9503041 * Bl
    spaces["XYZ"] = (X, Yx, Z)

    Xn, Yn, Zn = 0.95047, 1.0, 1.08883

    def f(t):
        return np.where(t > (6/29) ** 3, np.cbrt(t), (t / (3 * (6/29) ** 2)) + 4/29)

    fx, fy, fz = f(X / Xn), f(Yx / Yn), f(Z / Zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    spaces["LAB"] = (L, a, b)

    C = np.sqrt(a ** 2 + b ** 2)
    H = np.degrees(np.arctan2(b, a)) % 360
    spaces["LCH"] = (L, C, H)

    up = 4 * X / (X + 15 * Yx + 3 * Z + EPS)
    vp = 9 * Yx / (X + 15 * Yx + 3 * Z + EPS)
    un = 4 * Xn / (Xn + 15 * Yn + 3 * Zn)
    vn = 9 * Yn / (Xn + 15 * Yn + 3 * Zn)
    Lu = np.where(Yx / Yn > (6/29) ** 3, 116 * np.cbrt(Yx / Yn) - 16, (29/3) ** 3 * (Yx / Yn))
    Uu = 13 * Lu * (up - un)
    Vu = 13 * Lu * (vp - vn)
    spaces["LUV"] = (Lu, Uu, Vu)

    O1 = (R - G) / np.sqrt(2)
    O2 = (R + G - 2 * B) / np.sqrt(6)
    O3 = (R + G + B) / np.sqrt(3)
    spaces["OPPONENT"] = (O1, O2, O3)

    spaces["CMY"] = (1 - R, 1 - G, 1 - B)

    Yyuv = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.14713 * R - 0.28886 * G + 0.436 * B
    V = 0.615 * R - 0.51499 * G - 0.10001 * B
    spaces["YUV"] = (Yyuv, U, V)

    Yiq = 0.299 * R + 0.587 * G + 0.114 * B
    I = 0.595716 * R - 0.274453 * G - 0.321263 * B
    Q = 0.211456 * R - 0.522591 * G + 0.311135 * B
    spaces["YIQ"] = (Yiq, I, Q)

    Ydd = 0.299 * R + 0.587 * G + 0.114 * B
    Db = -0.450 * R - 0.883 * G + 1.333 * B
    Dr = -1.333 * R + 1.116 * G + 0.217 * B
    spaces["YDbDr"] = (Ydd, Db, Dr)

    Ypp = 0.299 * R + 0.587 * G + 0.114 * B
    Pb = -0.168736 * R - 0.331264 * G + 0.5 * B
    Pr = 0.5 * R - 0.418688 * G - 0.081312 * B
    spaces["YPbPr"] = (Ypp, Pb, Pr)

    return spaces


def feature_names():
    names = []
    for space in ["RGB", "NRGB", "YCbCr", "HSV", "HLS", "XYZ", "LAB", "LCH", "LUV",
                  "OPPONENT", "CMY", "YUV", "YIQ", "YDbDr", "YPbPr"]:
        for ch in range(3):
            names.append(f"MEAN_{space}_{ch}")
            names.append(f"STD_{space}_{ch}")
    return names


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
