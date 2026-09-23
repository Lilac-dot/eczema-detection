"""Patient-level train/val/test split for SkinDisNet, replacing the image-level split
built by split_external_manifest.py (found in the 2026-09-17 report revision to leak 81%
of test-set patients into the training set -- docs/... patient-leakage audit).

Joins SkinDisNet_Metadata.csv's Patient_id field (never previously used by this project's
SkinDisNet manifest code) to manifest_skindisnet.csv by filename stem, confirms every
patient falls under exactly one binary label (checked, not assumed), then splits PATIENTS
(not images) 50/20/30 stratified by label, seed 42 -- matching split_external_manifest.py's
same proportions and seed so this is a fair, single-variable comparison against the
existing (leaky) split.

Writes SkinDisNet/manifest_skindisnet_{train,val,test}_patientsplit.csv (same path,
disease_class,label schema as the original manifest).
"""
import csv
import random
from pathlib import Path

from paths import SKINDISNET_DIR

SEED = 42
SPLIT = (0.50, 0.20, 0.30)


def main():
    import pandas as pd

    meta = pd.read_csv(SKINDISNET_DIR / "SkinDisNet_Metadata.csv")
    meta["image_id_norm"] = meta["Image_id"].str.strip()
    img_to_patient = dict(zip(meta["image_id_norm"], meta["Patient_id"]))

    with open(SKINDISNET_DIR / "manifest_skindisnet.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        stem = Path(r["path"]).stem
        r["patient_id"] = img_to_patient[stem]

    patient_labels = {}
    for r in rows:
        pid, label = r["patient_id"], r["label"]
        patient_labels.setdefault(pid, set()).add(label)
    multi_label_patients = {p: labs for p, labs in patient_labels.items() if len(labs) > 1}
    if multi_label_patients:
        raise RuntimeError(f"Patients with more than one label found (unexpected): {multi_label_patients}")
    print(f"Confirmed: every one of {len(patient_labels)} patients has exactly one label.")

    patients_by_label = {"0": [], "1": []}
    for pid, labs in patient_labels.items():
        patients_by_label[next(iter(labs))].append(pid)

    rng = random.Random(SEED)
    split_patients = {"train": set(), "val": set(), "test": set()}
    for label, plist in patients_by_label.items():
        rng.shuffle(plist)
        n = len(plist)
        n_train = int(n * SPLIT[0])
        n_val = int(n * SPLIT[1])
        split_patients["train"].update(plist[:n_train])
        split_patients["val"].update(plist[n_train:n_train + n_val])
        split_patients["test"].update(plist[n_train + n_val:])
        print(f"label={label}: {n} patients -> train {n_train}, val {n_val}, test {n - n_train - n_val}")

    for name, pset in split_patients.items():
        split_rows = [r for r in rows if r["patient_id"] in pset]
        rng.shuffle(split_rows)
        out_path = SKINDISNET_DIR / f"manifest_skindisnet_{name}_patientsplit.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "disease_class", "label"])
            w.writeheader()
            for r in split_rows:
                w.writerow({"path": r["path"], "disease_class": r["disease_class"], "label": r["label"]})
        n1 = sum(1 for r in split_rows if r["label"] == "1")
        n0 = sum(1 for r in split_rows if r["label"] == "0")
        print(f"{out_path.name}: {len(split_rows)} images ({len(pset)} patients), Eczema={n1} Other={n0}")

    # Sanity check: zero patient overlap across splits by construction
    assert not (split_patients["train"] & split_patients["val"])
    assert not (split_patients["train"] & split_patients["test"])
    assert not (split_patients["val"] & split_patients["test"])
    print("\nConfirmed: zero patient overlap across train/val/test (by construction).")


if __name__ == "__main__":
    main()
