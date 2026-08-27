"""
Same as build_curated_subset.py, but the Eczema class is now the UNION of:
  - the original dataset's cleaned Eczema images (dataset/manifest_clean.csv, class==Eczema)
  - SkinDisease's cleaned Eczema images that are NOT duplicates of the above
    (scripts/check_eczema_overlap.py found 802 of 1034 are exact/near duplicates --
    those are dropped, keeping only the 232 genuinely new ones)

"Other" classes are unchanged: Psoriasis, Tinea, Candidiasis, Infestations_Bites, Lichen,
DrugEruption, Rosacea, all from SkinDisease (these were not checked against the original
dataset since they're unrelated diseases -- no plausible overlap).

Writes:
  SkinDisease/manifest_curated_v2.csv (path, disease_class, label)
  SkinDisease/manifest_curated_v2_{train,val,test}.csv
"""
import csv
import hashlib
import random
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import imagehash

ORIG_DATASET_ROOT = Path(r"C:\Users\tishy\Documents\Honors\dataset")
NEW_DATASET_ROOT = Path(r"C:\Users\tishy\Documents\Honors\SkinDisease")
random.seed(42)

OTHER_CLASSES = ["Psoriasis", "Tinea", "Candidiasis", "Infestations_Bites", "Lichen",
                 "DrugEruption", "Rosacea"]


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_class(manifest, cls):
    with open(manifest, newline="", encoding="utf-8") as f:
        return [row["path"] for row in csv.DictReader(f) if row["class"] == cls]


def main():
    orig_eczema = load_class(ORIG_DATASET_ROOT / "manifest_clean.csv", "Eczema")
    new_eczema = load_class(NEW_DATASET_ROOT / "manifest_clean.csv", "Eczema")
    print(f"Original dataset Eczema: {len(orig_eczema)}")
    print(f"SkinDisease Eczema (before dedup): {len(new_eczema)}")

    orig_md5, orig_phash = {}, {}
    for p in orig_eczema:
        with Image.open(p) as im:
            orig_md5[md5_of(p)] = p
            orig_phash[imagehash.phash(im.convert("RGB"))] = p

    new_unique = []
    for p in new_eczema:
        digest = md5_of(p)
        if digest in orig_md5:
            continue
        with Image.open(p) as im:
            ph = imagehash.phash(im.convert("RGB"))
        if any(ph - h <= 4 for h in orig_phash):
            continue
        new_unique.append(p)

    eczema_paths = orig_eczema + new_unique
    print(f"SkinDisease Eczema kept as genuinely new: {len(new_unique)}")
    print(f"Combined unique Eczema pool: {len(eczema_paths)}")

    by_class = defaultdict(list)
    by_class["Eczema"] = eczema_paths
    for cls in OTHER_CLASSES:
        by_class[cls] = load_class(NEW_DATASET_ROOT / "manifest_clean.csv", cls)

    print("\nFinal pooled counts:")
    for cls in sorted(by_class):
        print(f"  {cls}: {len(by_class[cls])}")

    all_rows = []
    for cls, paths in by_class.items():
        label = 1 if cls == "Eczema" else 0
        for p in paths:
            all_rows.append({"path": p, "disease_class": cls, "label": label})

    with open(NEW_DATASET_ROOT / "manifest_curated_v2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "disease_class", "label"])
        w.writeheader()
        w.writerows(all_rows)

    splits = {"train": [], "val": [], "test": []}
    for cls, paths in by_class.items():
        rows = [r for r in all_rows if r["disease_class"] == cls]
        random.shuffle(rows)
        n = len(rows)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        splits["train"].extend(rows[:n_train])
        splits["val"].extend(rows[n_train:n_train + n_val])
        splits["test"].extend(rows[n_train + n_val:])

    for name, rows in splits.items():
        with open(NEW_DATASET_ROOT / f"manifest_curated_v2_{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "disease_class", "label"])
            w.writeheader()
            w.writerows(rows)
        label_counts = Counter(r["label"] for r in rows)
        print(f"\n{name}: {len(rows)} total, label counts (1=Eczema)={dict(label_counts)}")


if __name__ == "__main__":
    main()
