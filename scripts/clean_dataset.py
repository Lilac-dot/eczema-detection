"""
Clean the Eczema/Normal image dataset.

- Verifies every file is a genuinely openable image (catches corrupt/truncated files)
- Detects exact duplicates (MD5) and near-duplicates (perceptual hash)
- Flags unusually tiny images (likely icons/thumbnails, not real skin photos)
- Flags non-image files
- Does NOT delete anything: moves flagged files into dataset/_quarantine/<class>/<reason>/
- Writes a clean manifest CSV (dataset/manifest_clean.csv) of surviving files + labels
- Writes a report CSV (dataset/clean_report.csv) listing every flagged file + reason
"""
import hashlib
import csv
import shutil
from pathlib import Path
from PIL import Image
import imagehash

ROOT = Path(r"C:\Users\tishy\Documents\Honors\dataset")
CLASSES = ["Eczema", "Normal"]
QUARANTINE = ROOT / "_quarantine"
MIN_DIM = 64  # px, below this we treat as a thumbnail/icon, not a usable photo

def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()

def quarantine(path: Path, cls: str, reason: str):
    dest_dir = QUARANTINE / cls / reason
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    if dest.exists():
        dest = dest_dir / f"{path.stem}_{md5_of(path)[:8]}{path.suffix}"
    shutil.move(str(path), str(dest))

def main():
    report_rows = []
    manifest_rows = []
    seen_md5 = {}       # md5 -> path (exact duplicates, across all classes)
    seen_phash = {}      # perceptual hash -> path (near duplicates, across all classes)

    for cls in CLASSES:
        cls_dir = ROOT / cls
        files = sorted(p for p in cls_dir.iterdir() if p.is_file())
        print(f"[{cls}] scanning {len(files)} files...")

        for path in files:
            # 1. Must be a genuinely openable image
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    ph = imagehash.phash(im)
            except Exception as e:
                report_rows.append([str(path), cls, "corrupt_or_unreadable", str(e)])
                quarantine(path, cls, "corrupt_or_unreadable")
                continue

            # 2. Too small to be a real photo
            if w < MIN_DIM or h < MIN_DIM:
                report_rows.append([str(path), cls, "too_small", f"{w}x{h}"])
                quarantine(path, cls, "too_small")
                continue

            # 3. Exact duplicate (byte-identical)
            digest = md5_of(path)
            if digest in seen_md5:
                report_rows.append([str(path), cls, "exact_duplicate", f"dup_of={seen_md5[digest]}"])
                quarantine(path, cls, "exact_duplicate")
                continue
            seen_md5[digest] = str(path)

            # 4. Near-duplicate (perceptual hash, hamming distance <= 4)
            is_near_dup = False
            for existing_hash, existing_path in seen_phash.items():
                if ph - existing_hash <= 4:
                    report_rows.append([str(path), cls, "near_duplicate", f"similar_to={existing_path}"])
                    quarantine(path, cls, "near_duplicate")
                    is_near_dup = True
                    break
            if is_near_dup:
                continue
            seen_phash[ph] = str(path)

            manifest_rows.append([str(path), cls, w, h])

    # Write outputs
    with open(ROOT / "clean_report.csv", "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow(["path", "class", "reason", "detail"])
        w_.writerows(report_rows)

    with open(ROOT / "manifest_clean.csv", "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow(["path", "class", "width", "height"])
        w_.writerows(manifest_rows)

    # Summary
    from collections import Counter
    reasons = Counter(r[2] for r in report_rows)
    kept = Counter(r[1] for r in manifest_rows)
    print("\n=== Cleaning summary ===")
    print("Flagged/quarantined:")
    for reason, n in reasons.items():
        print(f"  {reason}: {n}")
    print("Kept (clean manifest):")
    for cls, n in kept.items():
        print(f"  {cls}: {n}")
    print(f"\nTotal quarantined: {len(report_rows)}")
    print(f"Total kept: {len(manifest_rows)}")

if __name__ == "__main__":
    main()
