"""
Builds a clean, patient-wise-split manifest for the Widiawaty et al. 2025 Figshare dataset
(DOI 10.6084/m9.figshare.29925533.v4, "Clinical data of AD and non-AD research samples from
phases 1 and 2") -- the first dataset in this project with genuinely paired image + clinical-
text data per patient, usable to train a real jointly-trained image+text diagnosis model
(unlike the image+stress fusion, which stays decision-level because no such pairing exists
for wearable data anywhere).

Patient identity problem: the source spreadsheet has no real patient ID column -- "Sample
number" is per-photo, not per-patient (verified: hundreds of rows per sheet share an
identical clinical profile, almost certainly the same patient photographed more than once).

First attempt at a grouping key used only 4 coarse fields (Gender, Age, Past/Family Medical
History) and was WRONG -- checked and rejected before trusting it: those 4 fields have very
low cardinality (2 genders, common repeated ages, and Past/Family Medical History are
checkbox-style answers with only a handful of distinct combinations, e.g. "NaN" alone
covers 336 of 926 rows in one sheet), so that key collapsed 2809 rows into only 406 groups
-- mostly FALSE merges of unrelated patients who happen to share demographics, not real
repeat-patient detection. Over-merging still can't cause test-set leakage, but it would
have thrown away most of the dataset's real diversity for no genuine leakage protection.

Fixed key: every available clinical free-text field (Gender, Age, chief complaint, allergen
history, source of infection, trigger factors, duration, lesion location, Hanifin-Rajka
major/minor criteria where present, past/family medical history) -- much higher cardinality,
so an exact match across ALL of these is a far more specific same-patient signal, not a
demographic-bucket collision. This matches the exploratory check run before writing this
script (767/926, 445/697, 442/525, 605/663 unique combos per sheet -- about 80% of rows
unique, 20% plausible repeat-patient photos), not the first attempt's 86% merge rate.
Still a heuristic, not a verified patient ID -- flagged honestly, not treated as ground
truth.

Diagnosis label: sheet-derived (AD sheets -> 1, Non-AD sheets -> 0), cross-checked against
the Non-AD sheets' own "Diagnosis" column (Psoriasis vulgaris / Contact dermatitis /
Nummular dermatitis / Lichen simplex chronicus -- no missing or ambiguous values, confirms
sheet-level labeling is safe to trust).

Clinical text: chief complaint, allergen/irritant contact history, trigger factors, lesion
location, and (AD sheets only) the Hanifin-Rajka major/minor criteria strings, joined into
one free-text field per row -- this is the "anamnesis" input a text encoder would consume,
mirroring what the original paper's MPNet branch was built on.

Writes: dataset/widiawaty_figshare_29925533/manifest.csv
  columns: row_id, phase, label (1=AD/0=Non-AD), diagnosis, drive_file_id, clinical_text,
           patient_group_id, split (train/val/test)
"""
import re
import csv
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\tishy\Documents\Honors\dataset\widiawaty_figshare_29925533")
XLSX = ROOT / "Clinical_data_AD_nonAD_phases1_2.xlsx"
OUT_CSV = ROOT / "manifest.csv"

SEED = 42
SPLIT_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}

SHEET_LABELS = {
    "AD PHASE 1": (1, "phase1"),
    "AD PHASE 2": (1, "phase2"),
    "NON-AD PHASE 1": (0, "phase1"),
    "NON-AD PHASE 2": (0, "phase2"),
}

GROUP_KEY_COLS = [
    "Gender", "Patient Age", "Chief Complaint and Onset",
    "History of Contact with Allergens or Irritants", "Source of Infection",
    "Current Disease Trigger Factors", "Duration of Illness",
    "Lesion Location", "Location of lesion",
    "Major Criteria", "Minor Criteria",
    "Past Medical History", "Family Medical History",
]
TEXT_COLS = [
    "Chief Complaint and Onset",
    "History of Contact with Allergens or Irritants",
    "Source of Infection",
    "Current Disease Trigger Factors",
    "Lesion Location",
    "Location of lesion",
    "Major Criteria",
    "Minor Criteria",
]


def drive_id_from_link(link):
    if not isinstance(link, str):
        return None
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", link)
    return m.group(1) if m else None


def build_clinical_text(row):
    parts = []
    for col in TEXT_COLS:
        if col in row and pd.notna(row[col]):
            val = str(row[col]).strip()
            if val:
                parts.append(f"{col}: {val}")
    return " | ".join(parts)


def main():
    xl = pd.ExcelFile(XLSX)
    all_rows = []

    for sheet, (label, phase) in SHEET_LABELS.items():
        df = pd.read_excel(xl, sheet_name=sheet)
        link_col = [c for c in df.columns if "g-drive" in c.lower() or "foto" in c.lower()][0]

        for _, row in df.iterrows():
            group_key = tuple(
                str(row[c]).strip().lower() if c in row and pd.notna(row[c]) else ""
                for c in GROUP_KEY_COLS
            )
            all_rows.append({
                "phase": phase,
                "sheet": sheet,
                "label": label,
                "diagnosis": row.get("Diagnosis", "Atopic Dermatitis" if label == 1 else None),
                "drive_file_id": drive_id_from_link(row[link_col]),
                "clinical_text": build_clinical_text(row),
                "group_key": group_key,
            })

    df_all = pd.DataFrame(all_rows)
    before = len(df_all)
    df_all = df_all[df_all["drive_file_id"].notna()].reset_index(drop=True)
    dropped = before - len(df_all)
    if dropped:
        print(f"Dropped {dropped} rows with no parseable drive file id")

    # assign patient_group_id: same conservative key -> same id, stable across the whole df
    unique_keys = {k: i for i, k in enumerate(sorted(set(df_all["group_key"])))}
    df_all["patient_group_id"] = df_all["group_key"].map(unique_keys)
    df_all = df_all.drop(columns=["group_key"])

    n_groups = df_all["patient_group_id"].nunique()
    print(f"{len(df_all)} rows -> {n_groups} patient groups "
          f"({len(df_all) - n_groups} rows merged into an existing group's id)")

    # patient-wise, label-stratified-ish split: assign each GROUP (not row) to a split,
    # in an order that keeps AD/non-AD roughly balanced across splits despite group sizes
    # varying (some groups have 1 row, some have 6+).
    rng = np.random.RandomState(SEED)
    group_label = df_all.groupby("patient_group_id")["label"].agg(lambda s: s.mode()[0])
    group_size = df_all.groupby("patient_group_id").size()

    splits = {}
    for label_val in (0, 1):
        groups = group_label[group_label == label_val].index.tolist()
        rng.shuffle(groups)
        sizes = group_size.loc[groups]
        total = sizes.sum()
        cum = 0
        for g, sz in zip(groups, sizes):
            frac_so_far = cum / total
            if frac_so_far < SPLIT_FRACS["train"]:
                splits[g] = "train"
            elif frac_so_far < SPLIT_FRACS["train"] + SPLIT_FRACS["val"]:
                splits[g] = "val"
            else:
                splits[g] = "test"
            cum += sz

    df_all["split"] = df_all["patient_group_id"].map(splits)

    # sanity check: no patient group spans more than one split
    spans = df_all.groupby("patient_group_id")["split"].nunique()
    assert (spans == 1).all(), "BUG: a patient group spans multiple splits"

    df_all["row_id"] = range(len(df_all))
    out_cols = ["row_id", "phase", "label", "diagnosis", "drive_file_id", "clinical_text",
                "patient_group_id", "split"]
    df_all[out_cols].to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"\nSaved {OUT_CSV}")
    print("\nSplit sizes (rows):")
    print(df_all.groupby(["split", "label"]).size().unstack(fill_value=0))
    print("\nSplit sizes (patient groups):")
    print(df_all.drop_duplicates("patient_group_id").groupby(["split", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
