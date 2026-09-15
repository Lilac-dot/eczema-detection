"""
Experiment 2 (external-generalization improvement attempt) step 1: split SCIN's and
SkinDisNet's external manifests into train/val/test (stratified by label, 50/20/30) so a
slice of each can be used for fine-tuning while a held-out slice stays untouched for a fair
"before vs after" comparison. Run once for each dataset before train_curated_cnn_
multisource.py or eval_external_common.py's held-out-test path are used.

Writes <manifest>_{train,val,test}.csv next to the original manifest, same schema.
"""
import csv
import random
import sys
from pathlib import Path

SEED = 42
SPLIT = (0.50, 0.20, 0.30)  # train, val, test


def split_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    by_label = {"0": [], "1": []}
    for r in rows:
        by_label[r["label"]].append(r)

    rng = random.Random(SEED)
    splits = {"train": [], "val": [], "test": []}
    for label, group in by_label.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * SPLIT[0])
        n_val = int(n * SPLIT[1])
        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train:n_train + n_val])
        splits["test"].extend(group[n_train + n_val:])

    stem = manifest_path.stem
    for name, split_rows in splits.items():
        rng.shuffle(split_rows)
        out_path = manifest_path.parent / f"{stem}_{name}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(split_rows)
        n1 = sum(1 for r in split_rows if r["label"] == "1")
        n0 = sum(1 for r in split_rows if r["label"] == "0")
        print(f"{out_path.name}: {len(split_rows)} total (Eczema={n1}, Other={n0})")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(f"\nSplitting {path}...")
        split_manifest(path)
