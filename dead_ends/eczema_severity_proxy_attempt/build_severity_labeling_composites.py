"""Build 3x3 composite grids of eczema images for severity labeling by vision-based agents.

Each composite bundles 9 images into one file with a numbered label per cell, so a
single vision read can score 9 images at once instead of 9 separate reads. Cuts the
labeling job's image-read cost by ~9x. See docs/severity_scale_selection_2026-09-21.md
for the EASI-signs rubric these composites are labeled against.
"""
import csv
import os
from PIL import Image, ImageDraw, ImageFont

CELL_SIZE = 280
GRID = 3
PAD = 6
LABEL_H = 28

ROOT = "/Users/tishy/Documents/Honors"
IMAGES_CSV = f"{ROOT}/dataset/severity_labels/eczema_images_to_label.csv"
OUT_DIR = f"{ROOT}/dataset/severity_labels/composites"
MANIFEST_OUT = f"{ROOT}/dataset/severity_labels/composite_manifest.csv"


def load_font(size):
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def center_crop_resize(img, size):
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img.resize((size, size), Image.LANCZOS)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(IMAGES_CSV) as f:
        rows = list(csv.DictReader(f))

    font = load_font(20)
    manifest_rows = []
    n_per_grid = GRID * GRID
    composite_idx = 0

    for start in range(0, len(rows), n_per_grid):
        batch = rows[start:start + n_per_grid]
        composite_idx += 1
        cell_full = CELL_SIZE + PAD * 2
        canvas_w = cell_full * GRID
        canvas_h = (cell_full + LABEL_H) * GRID
        canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
        draw = ImageDraw.Draw(canvas)

        for i, row in enumerate(batch):
            r, c = divmod(i, GRID)
            x0 = c * cell_full
            y0 = r * (cell_full + LABEL_H)
            path = os.path.join(ROOT, row["path"])
            try:
                img = Image.open(path).convert("RGB")
                img = center_crop_resize(img, CELL_SIZE)
            except Exception as e:
                img = Image.new("RGB", (CELL_SIZE, CELL_SIZE), (80, 0, 0))
            canvas.paste(img, (x0 + PAD, y0 + LABEL_H))
            label = f"#{i + 1}"
            draw.rectangle([x0, y0, x0 + cell_full, y0 + LABEL_H], fill=(0, 0, 0))
            draw.text((x0 + 8, y0 + 4), label, fill=(255, 255, 0), font=font)

            manifest_rows.append({
                "composite_file": f"composite_{composite_idx:04d}.jpg",
                "cell_number": i + 1,
                "path": row["path"],
                "split": row["split"],
            })

        out_path = os.path.join(OUT_DIR, f"composite_{composite_idx:04d}.jpg")
        canvas.save(out_path, quality=85)

    with open(MANIFEST_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["composite_file", "cell_number", "path", "split"])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"Built {composite_idx} composites for {len(rows)} images -> {OUT_DIR}")
    print(f"Manifest: {MANIFEST_OUT}")


if __name__ == "__main__":
    main()
