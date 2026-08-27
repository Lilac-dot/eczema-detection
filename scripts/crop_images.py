"""
Auto-crop every image in manifest_clean.csv down to its largest skin-coloured region,
to strip out the background/composition confound documented in
docs/brightness_normalization_experiment_2026-08-26.md (Eczema = tight clinical crops,
Normal = varied stock-photo scenes with people/backgrounds).

No manual bounding-box annotations exist for this dataset and no pretrained
skin/lesion-segmentation model is available offline, so this uses a standard, well-known
heuristic: YCbCr skin-colour thresholding (Chai & Ngan, 1999: 133<=Cr<=173, 77<=Cb<=127),
connected-component analysis to find the single largest skin-coloured blob, then crop to
its bounding box with a margin. This is a cheap, explainable, class-agnostic rule -- it is
run identically on Eczema and Normal, so it cannot itself introduce a new class-specific
shortcut the way the original framing did.

Images where no sufficiently large skin region is found (mask covers < MIN_MASK_FRACTION
of the image) are NOT cropped -- the original image is copied through unchanged, and this
is logged, so a systematic detector failure on one class would be visible rather than
silently distorting that class's images.

Writes:
  dataset/cropped/<class>/<filename>       -- cropped (or, on fallback, copied) images
  dataset/manifest_cropped.csv             -- path, class, width, height, was_cropped, orig_path
  dataset/manifest_cropped_{train,val,test}.csv -- same split membership as the original
                                                    manifest_{train,val,test}.csv (by orig_path)
"""
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(r"C:\Users\tishy\Documents\Honors\dataset")
CROPPED_ROOT = ROOT / "cropped"
WORK_SIZE = 256          # mask computed at this resolution, then scaled back up
MARGIN_FRAC = 0.10        # extra margin added around the detected bounding box
MIN_MASK_FRACTION = 0.02  # if the largest skin blob is smaller than this fraction of the
                          # image, treat detection as failed and don't crop


def skin_mask(img_small):
    ycbcr = np.asarray(img_small.convert("YCbCr"), dtype=np.int16)
    _, cb, cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
    return (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)


def largest_component_bbox(mask):
    """Pick the skin blob that is both large AND central, not just large.

    Raw largest-connected-component picks up out-of-focus foreground/background skin
    (e.g. a blurred shoulder in a stock photo's bokeh) over the actual in-focus subject,
    because a smooth blurred patch is often a bigger uniform blob than a face broken up
    by eyes/hair/mouth. Weighting by distance from the image center fixes this in
    practice (subjects -- faces, hands, lesions -- are usually roughly centered in both
    the clinical and stock photos here) without needing a real segmentation model.
    """
    labeled, n = ndimage.label(mask)
    if n == 0:
        return None
    h, w = mask.shape
    cy0, cx0 = h / 2, w / 2
    max_dist = np.hypot(cy0, cx0)

    sizes = ndimage.sum(mask, labeled, index=range(1, n + 1))
    centers = ndimage.center_of_mass(mask, labeled, index=range(1, n + 1))

    best_label, best_score = None, -1.0
    for i in range(n):
        cy, cx = centers[i]
        dist = np.hypot(cy - cy0, cx - cx0)
        centrality = max(0.0, 1.0 - dist / max_dist)
        score = sizes[i] * centrality
        if score > best_score:
            best_score, best_label = score, i + 1

    if sizes[best_label - 1] / mask.size < MIN_MASK_FRACTION:
        return None
    ys, xs = np.where(labeled == best_label)
    return xs.min(), ys.min(), xs.max(), ys.max()


def crop_with_margin(path, out_path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    small = img.resize((WORK_SIZE, WORK_SIZE))
    bbox = largest_component_bbox(skin_mask(small))

    if bbox is None:
        shutil.copy(path, out_path)
        return False, w, h

    x0, y0, x1, y1 = bbox
    # scale bbox from WORK_SIZE space back to original resolution
    sx, sy = w / WORK_SIZE, h / WORK_SIZE
    x0, x1 = x0 * sx, x1 * sx
    y0, y1 = y0 * sy, y1 * sy
    bw, bh = x1 - x0, y1 - y0
    x0 = max(0, x0 - bw * MARGIN_FRAC)
    y0 = max(0, y0 - bh * MARGIN_FRAC)
    x1 = min(w, x1 + bw * MARGIN_FRAC)
    y1 = min(h, y1 + bh * MARGIN_FRAC)

    cropped = img.crop((int(x0), int(y0), int(x1), int(y1)))
    cropped.save(out_path)
    return True, cropped.width, cropped.height


def main():
    with open(ROOT / "manifest_clean.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    n_cropped = 0
    n_fallback = 0
    for i, row in enumerate(rows):
        src = Path(row["path"])
        cls = row["class"]
        dest_dir = CROPPED_ROOT / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name

        was_cropped, w, h = crop_with_margin(src, dest)
        n_cropped += was_cropped
        n_fallback += not was_cropped
        out_rows.append([str(dest), cls, w, h, was_cropped, str(src)])

        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(rows)} done")

    with open(ROOT / "manifest_cropped.csv", "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow(["path", "class", "width", "height", "was_cropped", "orig_path"])
        w_.writerows(out_rows)

    by_orig = {r[5]: r for r in out_rows}
    for split in ("train", "val", "test"):
        with open(ROOT / f"manifest_{split}.csv", newline="", encoding="utf-8") as f:
            split_rows = list(csv.DictReader(f))
        with open(ROOT / f"manifest_cropped_{split}.csv", "w", newline="", encoding="utf-8") as f:
            w_ = csv.writer(f)
            w_.writerow(["path", "class", "width", "height"])
            for r in split_rows:
                cropped_row = by_orig[r["path"]]
                w_.writerow([cropped_row[0], cropped_row[1], cropped_row[2], cropped_row[3]])

    from collections import Counter
    class_cropped = Counter()
    class_fallback = Counter()
    for r in out_rows:
        if r[4]:
            class_cropped[r[1]] += 1
        else:
            class_fallback[r[1]] += 1

    print(f"\nTotal images: {len(out_rows)}")
    print(f"Cropped: {n_cropped} ({100*n_cropped/len(out_rows):.1f}%)")
    print(f"Fallback (no crop, skin not confidently detected): {n_fallback} ({100*n_fallback/len(out_rows):.1f}%)")
    print("By class -- cropped:", dict(class_cropped))
    print("By class -- fallback:", dict(class_fallback))


if __name__ == "__main__":
    main()
