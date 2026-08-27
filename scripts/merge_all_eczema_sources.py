"""
Final merge: combine all three Eczema sources into one deduplicated pool, then downsample
the "Other" disease classes so the final dataset is close to 50/50, as requested.

Eczema sources:
  1. Original dataset/manifest_clean.csv, class == Eczema (1,428 images)
  2. SkinDisease/manifest_clean.csv, class == Eczema, minus duplicates of #1 (232 new)
  3. Eczema/manifest_clean.csv, is_eczema == 1, minus duplicates of #1+#2 (up to 719 new)

"Other" classes (Psoriasis, Tinea, Candidiasis, Infestations_Bites, Lichen, DrugEruption,
Rosacea) come only from SkinDisease, unchanged in content -- but downsampled in COUNT
(proportionally across the 7 classes, random sample) to roughly match the final Eczema
total, since growing Eczema alone can't reach 50/50 (there just isn't enough real Eczema
data to match Other's ~4,460 images) -- undersampling the abundant side is the only way
to hit parity without duplicating data.

Writes:
  SkinDisease/manifest_curated_v3.csv (path, disease_class, label)
  SkinDisease/manifest_curated_v3_{train,val,test}.csv
"""
import csv
import hashlib
import random
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import imagehash

ORIG_ROOT = Path(r"C:\Users\tishy\Documents\Honors\dataset")
SKINDISEASE_ROOT = Path(r"C:\Users\tishy\Documents\Honors\SkinDisease")
NEW_ECZEMA_ROOT = Path(r"C:\Users\tishy\Documents\Honors\Eczema")
random.seed(42)

OTHER_CLASSES = ["Psoriasis", "Tinea", "Candidiasis", "Infestations_Bites", "Lichen",
                 "DrugEruption", "Rosacea"]


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_of(path):
    with Image.open(path) as im:
        return imagehash.phash(im.convert("RGB"))


def dedup_against(candidates, known_md5, known_phash):
    """Return the subset of candidate paths that are NOT duplicates of anything in
    known_md5/known_phash, and add the survivors into those pools too."""
    survivors = []
    for p in candidates:
        digest = md5_of(p)
        if digest in known_md5:
            continue
        ph = phash_of(p)
        if any(ph - h <= 4 for h in known_phash):
            continue
        known_md5[digest] = p
        known_phash[ph] = p
        survivors.append(p)
    return survivors


def load_class(manifest, class_col, class_val):
    with open(manifest, newline="", encoding="utf-8") as f:
        return [row["path"] for row in csv.DictReader(f) if row[class_col] == class_val]


def main():
    # Source 1: original dataset's Eczema
    orig_eczema = load_class(ORIG_ROOT / "manifest_clean.csv", "class", "Eczema")
    print(f"Source 1 (original dataset): {len(orig_eczema)}")
    known_md5, known_phash = {}, {}
    for p in orig_eczema:
        known_md5[md5_of(p)] = p
        known_phash[phash_of(p)] = p

    # Source 2: SkinDisease's Eczema, deduped against source 1
    skindisease_eczema_all = load_class(SKINDISEASE_ROOT / "manifest_clean.csv", "class", "Eczema")
    skindisease_new = dedup_against(skindisease_eczema_all, known_md5, known_phash)
    print(f"Source 2 (SkinDisease): {len(skindisease_eczema_all)} total, {len(skindisease_new)} genuinely new")

    # Source 3: new Eczema/ dataset's eczema subtypes, deduped against sources 1+2
    with open(NEW_ECZEMA_ROOT / "manifest_clean.csv", newline="", encoding="utf-8") as f:
        new_eczema_all = [row["path"] for row in csv.DictReader(f) if row["is_eczema"] == "1"]
    new_eczema_unique = dedup_against(new_eczema_all, known_md5, known_phash)
    print(f"Source 3 (new Eczema/ dataset): {len(new_eczema_all)} total, {len(new_eczema_unique)} genuinely new")

    all_eczema = orig_eczema + skindisease_new + new_eczema_unique
    print(f"\nFinal combined unique Eczema pool: {len(all_eczema)}")

    # Other classes, unchanged content, from SkinDisease
    other_by_class = {cls: load_class(SKINDISEASE_ROOT / "manifest_clean.csv", "class", cls)
                       for cls in OTHER_CLASSES}
    other_total = sum(len(v) for v in other_by_class.values())
    print(f"Other classes pool (before downsampling): {other_total}")

    # Downsample Other proportionally to match the Eczema count for ~50/50
    target_other_total = len(all_eczema)
    if other_total > target_other_total:
        scale = target_other_total / other_total
        rng = random.Random(42)
        downsampled_other = {}
        for cls, paths in other_by_class.items():
            n_keep = max(1, round(len(paths) * scale))
            downsampled_other[cls] = rng.sample(paths, n_keep)
        print(f"Downsampling Other to ~{target_other_total} total (scale={scale:.3f}):")
        for cls in OTHER_CLASSES:
            print(f"  {cls}: {len(other_by_class[cls])} -> {len(downsampled_other[cls])}")
    else:
        downsampled_other = other_by_class
        print("Other pool already <= Eczema count, no downsampling needed")

    by_class = {"Eczema": all_eczema, **downsampled_other}
    total_eczema = len(by_class["Eczema"])
    total_other = sum(len(by_class[c]) for c in OTHER_CLASSES)
    print(f"\nFinal balance: Eczema={total_eczema} ({100*total_eczema/(total_eczema+total_other):.1f}%), "
          f"Other={total_other} ({100*total_other/(total_eczema+total_other):.1f}%)")

    all_rows = []
    for cls, paths in by_class.items():
        label = 1 if cls == "Eczema" else 0
        for p in paths:
            all_rows.append({"path": p, "disease_class": cls, "label": label})

    with open(SKINDISEASE_ROOT / "manifest_curated_v3.csv", "w", newline="", encoding="utf-8") as f:
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
        with open(SKINDISEASE_ROOT / f"manifest_curated_v3_{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "disease_class", "label"])
            w.writeheader()
            w.writerows(rows)
        label_counts = Counter(r["label"] for r in rows)
        print(f"\n{name}: {len(rows)} total, label counts (1=Eczema)={dict(label_counts)}")


if __name__ == "__main__":
    main()
