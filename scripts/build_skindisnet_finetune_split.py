# -*- coding: utf-8 -*-
"""Patient-level train/val/test split for SkinDisNet fine-tuning (2026-09-21), built over
the DEDUPLICATED, mac-path-correct manifest (manifest_skindisnet_clean.csv) rather than the
raw manifest_skindisnet.csv that manifest_skindisnet_{train,val,test}_patientsplit.csv were
built from.

Two problems with reusing the existing _clean or _patientsplit manifests directly for a new
fine-tuning experiment on this machine:
  1. manifest_skindisnet_{train,val,test}_clean.csv and manifest_skindisnet_{train,val,test}
     _patientsplit.csv still carry Windows-style absolute paths (C:\\Users\\tishy\\...) from
     the original x86 development machine -- unusable as-is on this Apple Silicon Mac.
  2. The _patientsplit manifests were built (build_skindisnet_patient_split.py) from the
     RAW manifest_skindisnet.csv (1710 rows, pre-dedup), not the deduplicated/skin-content-
     filtered manifest_skindisnet_clean.csv (1558 rows) that this project's own zero-shot
     external-validation check (skindisnet_all9_external_validation.py) actually uses -- so
     their membership doesn't match the "clean" image set this paper reports against.

This script reuses build_skindisnet_patient_split.py's exact methodology (join Patient_id
from SkinDisNet_Metadata.csv by filename stem, verify every patient maps to exactly one
label, split PATIENTS not images, 50/20/30, stratified by label, seed 42) but runs it over
manifest_skindisnet_clean.csv's image list, so the output has both correct relative paths
AND the same deduplicated/filtered image set as the rest of this paper's SkinDisNet work.

Writes SkinDisNet/manifest_skindisnet_finetune_{train,val,test}.csv (same path,disease_class,
label schema as the source manifest). Does not touch any existing manifest file.
"""
import csv
import random
from pathlib import Path

from paths import SKINDISNET_DIR

SEED = 42
SPLIT = (0.50, 0.20, 0.30)
SOURCE_MANIFEST = SKINDISNET_DIR / "manifest_skindisnet_clean.csv"
METADATA = SKINDISNET_DIR / "SkinDisNet_Metadata.csv"


def main():
    meta_rows = list(csv.DictReader(open(METADATA, newline="", encoding="utf-8")))
    img_to_patient = {r["Image_id"].strip(): r["Patient_id"] for r in meta_rows}

    rows = list(csv.DictReader(open(SOURCE_MANIFEST, newline="", encoding="utf-8")))
    print(f"Source (deduplicated) manifest: {len(rows)} images")

    missing = []
    for r in rows:
        stem = Path(r["path"]).stem
        pid = img_to_patient.get(stem)
        if pid is None:
            missing.append(stem)
        r["patient_id"] = pid
    if missing:
        raise RuntimeError(f"{len(missing)} images in the clean manifest have no "
                            f"Patient_id match in {METADATA.name}: {missing[:10]}")
    print(f"Confirmed: every one of {len(rows)} clean images has a Patient_id match.")

    patient_labels = {}
    for r in rows:
        patient_labels.setdefault(r["patient_id"], set()).add(r["label"])
    multi_label_patients = {p: labs for p, labs in patient_labels.items() if len(labs) > 1}
    if multi_label_patients:
        raise RuntimeError(f"Patients with more than one label found (unexpected): "
                            f"{multi_label_patients}")
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
        print(f"label={label}: {n} patients -> train {n_train}, val {n_val}, "
              f"test {n - n_train - n_val}")

    for name, pset in split_patients.items():
        split_rows = [r for r in rows if r["patient_id"] in pset]
        rng.shuffle(split_rows)
        out_path = SKINDISNET_DIR / f"manifest_skindisnet_finetune_{name}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "disease_class", "label"])
            w.writeheader()
            for r in split_rows:
                w.writerow({"path": r["path"], "disease_class": r["disease_class"],
                            "label": r["label"]})
        n1 = sum(1 for r in split_rows if r["label"] == "1")
        n0 = sum(1 for r in split_rows if r["label"] == "0")
        print(f"{out_path.name}: {len(split_rows)} images ({len(pset)} patients), "
              f"positive(Eczema/AD)={n1} other={n0} "
              f"(pos_fraction={n1 / len(split_rows):.3f})")

    assert not (split_patients["train"] & split_patients["val"])
    assert not (split_patients["train"] & split_patients["test"])
    assert not (split_patients["val"] & split_patients["test"])
    print("\nConfirmed: zero patient overlap across finetune train/val/test (by construction).")


if __name__ == "__main__":
    main()
