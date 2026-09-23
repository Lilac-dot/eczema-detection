"""
Feature extraction for the second wearable-stress dataset (Hongn et al. 2025, "Wearable
Device Dataset from Induced Stress and Structured Exercise Sessions", PhysioNet,
DOI 10.13026/he0v-tf17), mirroring build_wesad_features.py EXACTLY so the output feature
table has the same columns as dataset/WESAD/wesad_wrist_features.csv and can be fed
straight into the existing WESAD stress model or a domain-shift eval against it.

NOT YET RUN AGAINST REAL DATA -- physionet.org is unreachable from this dev environment
(confirmed: DNS resolves, TCP connect times out -- a network-level block on this machine,
not a dataset access/credentialing issue; the dataset itself is Open Data Commons
Attribution licensed, no login needed). Written from the companion Scientific Data paper's
documented file format (Hongn et al. 2025, PMC11953403), NOT verified against the actual
files. Column/file names below are the paper's own documented names -- check them against
the real downloaded files before trusting this script's output, and fix anything that
doesn't match (this docstring will be wrong about something; the paper's methods section
is not the same as reading the actual CSV header).

Expected layout once downloaded, per the paper (STRESS/<subject_id>/):
  BVP.csv, EDA.csv, TEMP.csv, ACC.csv, HR.csv, IBI.csv, tags.csv
  -- Empatica E4 export convention: row 1 = UTC start timestamp, row 2 = sample rate (Hz),
     rows 3+ = data (ACC.csv has 3 columns, x/y/z; the rest are single-column).
  -- tags.csv: one UTC timestamp per row, one per button press marking a protocol-phase
     boundary (start/end of each rest/task segment).
Stress_level_v1.csv / Stress_level_v2.csv (one file, at the STRESS/ root, not per-subject
  -- TODO verify): subject's self-reported stress level (1-10) before/after each phase.

Two known protocol variants share this dataset (different subject ID prefixes):
  v1 ("Sxx", S01-S18): baseline -> Stroop -> rest -> TMCT -> rest -> debate -> backward-count
  v2 ("fxx", f01-f18): same idea, Stroop removed, rest periods extended/relocated, remote.
The exact phase count/order differs between them -- this script derives phases from
tags.csv boundaries directly rather than hardcoding either timeline, so it should handle
both, but this is unverified.

Known exclusions (per the paper's own ML validation): f07 (incorrect wearable placement),
f13 (bad wristband fit) -- excluded here too, for the same reason the paper excluded them.

Labeling choice (mine, not the paper's): binarizes self-reported stress level (SL) into
stress/not-stress the same way this project already treats WESAD (baseline/amusement=0,
TSST=1) -- SL empirically has no fixed WESAD-style condition code here, so the split has
to be a threshold choice on SL. Default SL_STRESS_THRESHOLD=5 (SL>=5 -> stress) is a
starting point, not a validated choice -- revisit once real SL value distributions are
visible (e.g. plot the histogram before committing to a cutoff).

Writes dataset/stress2_physionet/stress2_wrist_features.csv with the SAME columns as
wesad_wrist_features.csv (subject_id, label, condition, <48 feature columns: 24 raw +
24 personal-baseline-relative _rel>), so eval scripts can concatenate or cross-test
against WESAD directly.
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from paths import ROOT
from wesad_features import WINDOW_SEC, FS, extract_window_features
from build_wesad_features import add_personal_baseline_features

# EDIT to wherever the downloaded dataset actually lands.
DATASET_DIR = ROOT / "dataset" / "stress2_physionet" / "STRESS"
OUT_CSV = ROOT / "dataset" / "stress2_physionet" / "stress2_wrist_features.csv"

EXCLUDED_SUBJECTS = {"f07", "f13"}
SL_STRESS_THRESHOLD = 5  # TODO: revisit once real SL distribution is visible
BASELINE_PHASE_KEYWORDS = ("baseline", "rest")  # TODO: match against real tags/phase labels


def load_e4_csv(path):
    """Empatica E4 export: row1=start UTC timestamp, row2=sample rate (Hz), rows3+=data."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        start_ts = float(next(reader)[0])
        fs = float(next(reader)[0])
        data = np.array([[float(x) for x in row] for row in reader if row])
    return start_ts, fs, data


def load_tags(path):
    with open(path, newline="") as f:
        return [float(row[0]) for row in csv.reader(f) if row]


def process_subject(subject_dir, sl_lookup):
    """Returns a list of per-window feature dicts, matching build_wesad_features.py's
    process_subject() output shape (feats + subject_id + label + condition)."""
    subject_id = subject_dir.name

    signals = {}
    starts = {}
    for mod, fname in (("EDA", "EDA.csv"), ("TEMP", "TEMP.csv"), ("BVP", "BVP.csv"), ("ACC", "ACC.csv")):
        start_ts, fs, data = load_e4_csv(subject_dir / fname)
        signals[mod] = data
        starts[mod] = start_ts
        assert abs(fs - FS[mod]) < 0.5, (
            f"{subject_id} {mod}: file says {fs} Hz, wesad_features.py expects {FS[mod]} Hz -- "
            f"extract_window_features() will silently mis-featurize this. Fix FS in "
            f"wesad_features.py or resample before proceeding, don't just ignore this assert."
        )

    tag_times = load_tags(subject_dir / "tags.csv")
    if len(tag_times) < 2:
        print(f"  {subject_id}: fewer than 2 tags, cannot form any phase -- skipping")
        return []

    # TODO: this assumes consecutive tag pairs bound alternating rest/task phases in
    # protocol order, and that SL ratings in sl_lookup are ordered to match -- VERIFY
    # against the real tags.csv/Stress_level file structure before trusting output.
    phases = list(zip(tag_times[:-1], tag_times[1:]))
    sl_values = sl_lookup.get(subject_id, [])
    if len(sl_values) < len(phases):
        print(f"  {subject_id}: {len(phases)} phases but only {len(sl_values)} SL ratings -- "
              f"labeling only the first {len(sl_values)} phases, dropping the rest")
        phases = phases[:len(sl_values)]

    rows = []
    for phase_idx, (t0, t1) in enumerate(phases):
        sl = sl_values[phase_idx] if phase_idx < len(sl_values) else None
        if sl is None:
            continue
        label = int(sl >= SL_STRESS_THRESHOLD)
        condition = 1 if label == 0 else 2  # mirrors WESAD's baseline=1/stress=2 convention

        n_windows = int((t1 - t0) // WINDOW_SEC)
        for w in range(n_windows):
            w_start = t0 + w * WINDOW_SEC
            w_end = w_start + WINDOW_SEC

            sig_slices = {}
            ok = True
            for mod in FS:
                start_ts, fs = starts[mod], FS[mod]
                s = int((w_start - start_ts) * fs)
                e = int((w_end - start_ts) * fs)
                if s < 0 or e > len(signals[mod]) or e - s < fs * 2:
                    ok = False
                    break
                sig_slices[mod] = signals[mod][s:e]
            if not ok:
                continue

            feats = extract_window_features(sig_slices)
            feats["subject_id"] = subject_id
            feats["label"] = label
            feats["condition"] = condition
            feats["stress_level_raw"] = sl
            rows.append(feats)

    return rows


def load_sl_lookup():
    """TODO: verify exact column layout of Stress_level_v1.csv / Stress_level_v2.csv once
    downloaded -- this assumes one row per subject, one column per phase, which is a guess
    based on the paper's description, not a confirmed file read."""
    sl_lookup = {}
    for fname in ("Stress_level_v1.csv", "Stress_level_v2.csv"):
        fpath = DATASET_DIR.parent / fname
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found -- skipping")
            continue
        df = pd.read_csv(fpath)
        id_col = df.columns[0]
        for _, row in df.iterrows():
            sid = str(row[id_col])
            sl_lookup[sid] = [float(v) for v in row[1:] if pd.notna(v)]
    return sl_lookup


def main():
    if not DATASET_DIR.exists():
        print(f"REFUSING TO RUN: {DATASET_DIR} does not exist yet.")
        print("Download the STRESS/ folder from "
              "https://physionet.org/content/wearable-device-dataset/1.0.1/ "
              f"and place it at {DATASET_DIR} (or edit DATASET_DIR above).")
        return

    sl_lookup = load_sl_lookup()
    subject_dirs = sorted(
        p for p in DATASET_DIR.iterdir()
        if p.is_dir() and p.name not in EXCLUDED_SUBJECTS
    )
    print(f"Found {len(subject_dirs)} subject directories (excluded: {sorted(EXCLUDED_SUBJECTS)})")

    all_rows = []
    for sd in subject_dirs:
        rows = process_subject(sd, sl_lookup)
        n_stress = sum(r["label"] for r in rows)
        print(f"  {sd.name}: {len(rows)} windows, {n_stress} stress "
              f"({100 * n_stress / len(rows) if rows else 0:.1f}%)")
        all_rows.extend(rows)

    if not all_rows:
        print("\nNo windows extracted -- something in the assumed file format is wrong. "
              "Do not proceed to training on an empty/garbage table.")
        return

    df = pd.DataFrame(all_rows)
    base_feat_cols = [c for c in df.columns
                       if c not in ("subject_id", "label", "condition", "stress_level_raw")]
    df = add_personal_baseline_features(df, base_feat_cols)

    cols = ["subject_id", "label", "condition", "stress_level_raw"] + \
           [c for c in df.columns if c not in ("subject_id", "label", "condition", "stress_level_raw")]
    df = df[cols]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nTotal windows: {len(df)}")
    print(f"Stress windows: {df['label'].sum()} ({100 * df['label'].mean():.1f}%)")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print(f"Saved: {OUT_CSV}")
    print("\nSANITY-CHECK THIS OUTPUT before using it for anything -- this script was "
          "written from the paper's description, not tested against real files.")


if __name__ == "__main__":
    main()
