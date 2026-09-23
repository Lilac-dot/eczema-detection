"""
Stock-photo / non-skin contamination screen for SCIN and SkinDisNet, reusing the same
YCbCr skin-colour-fraction heuristic that caught the SkinDisease dataset's contaminated
Normal class (scripts/check_normal_class_content.py). Applied class-blind (Eczema and
Other alike) to both external datasets used in the cross-dataset generalization matrix.

Does not move/delete anything -- diagnostic only. Writes:
  docs/external_skin_content_check_2026-09-16.csv
"""
import csv
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(r"C:\Users\tishy\Documents\Honors")
WORK_SIZE = 128


def skin_fraction(img_small):
    ycbcr = np.asarray(img_small.convert("YCbCr"), dtype=np.int16)
    _, cb, cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
    mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    return float(mask.mean())


def load_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def screen(manifest_path, dataset_label):
    rows = load_manifest(manifest_path)
    out = []
    for row in rows:
        p = row["path"]
        label = "Eczema" if row["label"] == "1" else "Other"
        try:
            img = Image.open(p).convert("RGB").resize((WORK_SIZE, WORK_SIZE))
            frac = skin_fraction(img)
            out.append([dataset_label, p, label, frac, ""])
        except Exception as e:
            out.append([dataset_label, p, label, "", f"error:{e}"])
    return out


def summarize(rows, dataset_label):
    from collections import defaultdict
    by_class = defaultdict(list)
    for ds, p, cls, frac, err in rows:
        if ds == dataset_label and frac != "":
            by_class[cls].append(float(frac))
    print(f"\n=== {dataset_label} skin-fraction summary ===")
    print(f"{'class':10s} {'n':>6s} {'mean':>8s} {'median':>8s} {'%below_0.02':>12s} {'%below_0.05':>12s}")
    for cls in sorted(by_class):
        vals = np.array(by_class[cls])
        pct_below_2 = 100 * (vals < 0.02).mean()
        pct_below_5 = 100 * (vals < 0.05).mean()
        print(f"{cls:10s} {len(vals):6d} {vals.mean():8.3f} {np.median(vals):8.3f} {pct_below_2:12.1f} {pct_below_5:12.1f}")


def main():
    rows = []
    rows += screen(ROOT / "dataset" / "SCIN" / "manifest_scin.csv", "SCIN")
    rows += screen(ROOT / "SkinDisNet" / "manifest_skindisnet.csv", "SkinDisNet")

    out_path = ROOT / "docs" / "external_skin_content_check_2026-09-16.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "path", "class", "skin_fraction", "error"])
        w.writerows(rows)

    summarize(rows, "SCIN")
    summarize(rows, "SkinDisNet")
    print(f"\nReport written: {out_path}")


if __name__ == "__main__":
    main()
