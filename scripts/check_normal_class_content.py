"""
Content sanity check for the new SkinDisease dataset, prompted by manual spot-checking
that turned up several "Normal" class images that aren't photos of skin at all (a drill
kit, cherries on a branch, a dish rack, a frying pan, a vacuum cleaner, a train, a cat,
an elephant, folded towels...).

Reuses the YCbCr skin-colour mask from scripts/crop_images.py as an automated, class-blind
screen: for every image in every class/split, compute what fraction of pixels are
plausibly skin-toned. Applied identically to ALL classes (not just Normal) so this is a
uniform content check, not a rule invented to specifically target one class.

Does NOT move or delete anything -- this is a diagnostic report to decide whether/how much
of a problem this is before deciding what to quarantine. Writes:
  SkinDisease/skin_content_check.csv  -- path, split, class, skin_fraction
"""
import csv
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(r"C:\Users\tishy\Documents\Honors\SkinDisease")
SPLITS = ["train", "valid", "test"]
WORK_SIZE = 128


def skin_fraction(img_small):
    ycbcr = np.asarray(img_small.convert("YCbCr"), dtype=np.int16)
    _, cb, cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
    mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    return float(mask.mean())


def main():
    rows = []
    for split in SPLITS:
        split_dir = ROOT / split
        if not split_dir.exists():
            continue
        classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        for cls in classes:
            cls_dir = split_dir / cls
            files = sorted(p for p in cls_dir.iterdir() if p.is_file())
            print(f"[{split}/{cls}] checking {len(files)} files...")
            for path in files:
                try:
                    img = Image.open(path).convert("RGB").resize((WORK_SIZE, WORK_SIZE))
                    frac = skin_fraction(img)
                except FileNotFoundError:
                    continue  # already moved by the concurrent corruption/dup cleaning pass
                except Exception as e:
                    rows.append([str(path), split, cls, "", f"error:{e}"])
                    continue
                rows.append([str(path), split, cls, frac, ""])

    with open(ROOT / "skin_content_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "split", "class", "skin_fraction", "error"])
        w.writerows(rows)

    # Summary per class
    from collections import defaultdict
    by_class = defaultdict(list)
    for path, split, cls, frac, err in rows:
        if frac != "":
            by_class[cls].append(float(frac))

    print("\n=== Skin-fraction summary by class ===")
    print(f"{'class':22s} {'n':>6s} {'mean':>8s} {'median':>8s} {'%below_0.02':>12s} {'%below_0.05':>12s}")
    for cls in sorted(by_class):
        vals = np.array(by_class[cls])
        pct_below_2 = 100 * (vals < 0.02).mean()
        pct_below_5 = 100 * (vals < 0.05).mean()
        print(f"{cls:22s} {len(vals):6d} {vals.mean():8.3f} {np.median(vals):8.3f} {pct_below_2:12.1f} {pct_below_5:12.1f}")


if __name__ == "__main__":
    main()
