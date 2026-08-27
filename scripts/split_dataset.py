"""
Stratified train/val/test split of the cleaned manifest.
Writes dataset/manifest_train.csv, manifest_val.csv, manifest_test.csv (70/15/15 per class).
"""
import csv
import random
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"C:\Users\tishy\Documents\Honors\dataset")
random.seed(42)

by_class = defaultdict(list)
with open(ROOT / "manifest_clean.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        by_class[row["class"]].append(row)

splits = {"train": [], "val": [], "test": []}
for cls, rows in by_class.items():
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    splits["train"].extend(rows[:n_train])
    splits["val"].extend(rows[n_train:n_train + n_val])
    splits["test"].extend(rows[n_train + n_val:])

for name, rows in splits.items():
    with open(ROOT / f"manifest_{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "class", "width", "height"])
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    counts = Counter(r["class"] for r in rows)
    print(f"{name}: {len(rows)} total -> {dict(counts)}")
