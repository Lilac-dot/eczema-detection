"""
Clean the new "Human Skin Diseases (Image)" Kaggle dataset (SkinDisease/{train,valid,test}/<class>/),
same methodology as scripts/clean_dataset.py applied to the original Eczema/Normal set:

- Verifies every file is a genuinely openable image (catches corrupt/truncated files)
- Detects exact duplicates (MD5) and near-duplicates (perceptual hash), WITHIN each of
  train/valid/test separately -- a duplicate across train and test would be a real
  train/test leak, so that is checked and reported separately and more seriously.
- Flags unusually tiny images (likely icons/thumbnails, not real skin photos)
- Flags non-image files
- Does NOT delete anything: moves flagged files into SkinDisease/_quarantine/<split>/<class>/<reason>/
- Writes a clean manifest CSV (SkinDisease/manifest_clean.csv) of surviving files + labels + split
- Writes a report CSV (SkinDisease/clean_report.csv) listing every flagged file + reason
- Writes SkinDisease/cross_split_duplicates.csv for any exact/near duplicate found between
  different splits (train vs valid vs test) -- this is a leakage risk, not just a redundancy
  one, and is called out separately from within-split duplicates.
"""
import hashlib
import csv
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image
import imagehash

ROOT = Path(r"C:\Users\tishy\Documents\Honors\SkinDisease")
SPLITS = ["train", "valid", "test"]
QUARANTINE = ROOT / "_quarantine"
MIN_DIM = 64


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def quarantine(path: Path, split: str, cls: str, reason: str):
    dest_dir = QUARANTINE / split / cls / reason
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    if dest.exists():
        dest = dest_dir / f"{path.stem}_{md5_of(path)[:8]}{path.suffix}"
    shutil.move(str(path), str(dest))


def main():
    report_rows = []
    manifest_rows = []

    # global (cross-split) registries, to catch train/test leakage specifically
    global_md5 = {}    # md5 -> (split, class, path)
    global_phash = {}  # phash -> (split, class, path)
    cross_split_dupes = []

    for split in SPLITS:
        split_dir = ROOT / split
        classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        seen_md5 = {}     # within this split+class scope isn't needed -- dedupe within whole split
        seen_phash = {}

        for cls in classes:
            cls_dir = split_dir / cls
            files = sorted(p for p in cls_dir.iterdir() if p.is_file())
            print(f"[{split}/{cls}] scanning {len(files)} files...")

            for path in files:
                try:
                    with Image.open(path) as im:
                        im.verify()
                    with Image.open(path) as im:
                        im = im.convert("RGB")
                        w, h = im.size
                        ph = imagehash.phash(im)
                except Exception as e:
                    report_rows.append([str(path), split, cls, "corrupt_or_unreadable", str(e)])
                    quarantine(path, split, cls, "corrupt_or_unreadable")
                    continue

                if w < MIN_DIM or h < MIN_DIM:
                    report_rows.append([str(path), split, cls, "too_small", f"{w}x{h}"])
                    quarantine(path, split, cls, "too_small")
                    continue

                digest = md5_of(path)

                # cross-split check first (more serious)
                if digest in global_md5:
                    other_split, other_cls, other_path = global_md5[digest]
                    if other_split != split:
                        cross_split_dupes.append(["exact", str(path), split, cls, other_path, other_split, other_cls])
                    report_rows.append([str(path), split, cls, "exact_duplicate", f"dup_of={other_path}"])
                    quarantine(path, split, cls, "exact_duplicate")
                    continue
                global_md5[digest] = (split, cls, str(path))

                is_near_dup = False
                for existing_hash, (o_split, o_cls, o_path) in list(global_phash.items()):
                    if ph - existing_hash <= 4:
                        if o_split != split:
                            cross_split_dupes.append(["near", str(path), split, cls, o_path, o_split, o_cls])
                        report_rows.append([str(path), split, cls, "near_duplicate", f"similar_to={o_path}"])
                        quarantine(path, split, cls, "near_duplicate")
                        is_near_dup = True
                        break
                if is_near_dup:
                    continue
                global_phash[ph] = (split, cls, str(path))

                manifest_rows.append([str(path), split, cls, w, h])

    with open(ROOT / "clean_report.csv", "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow(["path", "split", "class", "reason", "detail"])
        w_.writerows(report_rows)

    with open(ROOT / "manifest_clean.csv", "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow(["path", "split", "class", "width", "height"])
        w_.writerows(manifest_rows)

    with open(ROOT / "cross_split_duplicates.csv", "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow(["dup_type", "path", "split", "class", "duplicate_of_path", "other_split", "other_class"])
        w_.writerows(cross_split_dupes)

    from collections import Counter
    reasons = Counter(r[3] for r in report_rows)
    kept = Counter((r[1], r[2]) for r in manifest_rows)
    kept_by_split = Counter(r[1] for r in manifest_rows)

    print("\n=== Cleaning summary ===")
    print("Flagged/quarantined:")
    for reason, n in reasons.items():
        print(f"  {reason}: {n}")
    print(f"\nTotal quarantined: {len(report_rows)}")
    print(f"Total kept: {len(manifest_rows)}")
    print("\nKept by split:")
    for split, n in kept_by_split.items():
        print(f"  {split}: {n}")
    print(f"\nCross-split duplicates (train/valid/test leakage risk): {len(cross_split_dupes)}")


if __name__ == "__main__":
    main()
