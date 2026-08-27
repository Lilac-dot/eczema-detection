"""
Build the Eczema-vs-other-skin-condition subset from the cleaned SkinDisease dataset
(SkinDisease/manifest_clean.csv).

Curated "other" classes (picked for looking clinically similar to eczema -- itchy,
red/scaly rashes -- rather than dumping in every disease, which would make "other" too
easy to tell apart from Eczema just by being visually unrelated):
  Psoriasis, Tinea, Candidiasis, Infestations_Bites, Lichen, DrugEruption, Rosacea

The dataset's own train/valid/test split turned out to be unreliable (see
docs/skindisease_dataset_check_2026-08-26.md addendum -- Candidiasis and DrugEruption's
entire "valid" folders were exact duplicates of their "train" folders, and there were 173
test-vs-train/valid duplicates across other classes too). So this ignores the provided
split entirely: pools every surviving (cleaned, deduplicated) image for the 8 classes
across train+valid+test, then re-splits 70/15/15 ourselves, stratified by the original
8-way class (not just the binary label), the same method already used for the original
Eczema/Normal dataset (scripts/split_dataset.py).

Writes:
  SkinDisease/manifest_curated.csv (path, disease_class, label) -- label 1=Eczema, 0=Other
  SkinDisease/manifest_curated_{train,val,test}.csv
"""
import csv
import random
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(r"C:\Users\tishy\Documents\Honors\SkinDisease")
random.seed(42)

OTHER_CLASSES = ["Psoriasis", "Tinea", "Candidiasis", "Infestations_Bites", "Lichen",
                 "DrugEruption", "Rosacea"]
KEEP_CLASSES = set(OTHER_CLASSES) | {"Eczema"}


def main():
    by_class = defaultdict(list)
    with open(ROOT / "manifest_clean.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["class"] in KEEP_CLASSES:
                by_class[row["class"]].append(row["path"])

    print("Pooled counts (train+valid+test combined, post-cleaning):")
    for cls in sorted(by_class):
        print(f"  {cls}: {len(by_class[cls])}")

    all_rows = []
    for cls, paths in by_class.items():
        label = 1 if cls == "Eczema" else 0
        for p in paths:
            all_rows.append({"path": p, "disease_class": cls, "label": label})

    with open(ROOT / "manifest_curated.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "disease_class", "label"])
        w.writeheader()
        w.writerows(all_rows)

    # Stratified 70/15/15 split by the 8-way disease_class (not just binary label)
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
        with open(ROOT / f"manifest_curated_{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "disease_class", "label"])
            w.writeheader()
            w.writerows(rows)
        label_counts = Counter(r["label"] for r in rows)
        class_counts = Counter(r["disease_class"] for r in rows)
        print(f"\n{name}: {len(rows)} total, label counts (1=Eczema)={dict(label_counts)}")
        print(f"  by disease_class: {dict(class_counts)}")


if __name__ == "__main__":
    main()
