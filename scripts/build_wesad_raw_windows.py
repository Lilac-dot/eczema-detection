"""
Build RAW (not hand-summarized) windowed signal tensors from the WESAD wrist data, for
the CNN+attention model in train_wesad_cnn_attention.py. Unlike build_wesad_features.py
(which collapses each window to ~6 stats per modality for LightGBM), this keeps the full
per-sample signal so a 1D-CNN can learn its own patterns.

Windowing: 30-second windows, 15-second stride (50% overlap) -- shorter and overlapping
compared to the LightGBM version's 60s/no-overlap, specifically to give a neural net more
training examples (a CNN needs far more examples per subject than LightGBM does). Overlap
only ever happens within one subject's own recording, so it can't leak across the
subject-level split used for evaluation.

Same label rule as build_wesad_features.py: majority-vote from the 700Hz label channel,
>=90% agreement required, only baseline(1)/stress(2)/amusement(3) kept, stress vs.
not-stress binary target.

Unlike the stats version's fraction-of-total-length window boundaries, this computes each
modality's sample range directly from elapsed time (start_sample = round(t * fs)), so
every window has an EXACT fixed sample count per modality -- required for a CNN's fixed
input shape.

Writes dataset/WESAD/wesad_raw_windows.npz with arrays:
  EDA (N,120)  TEMP (N,120)  BVP (N,1920)  ACC (N,960,3)  label (N,)  subject (N,) [strings]
"""
import pickle
from pathlib import Path

import numpy as np

WESAD_DIR = Path(r"C:\Users\tishy\Documents\Honors\dataset\WESAD")
OUT_NPZ = Path(r"C:\Users\tishy\Documents\Honors\dataset\WESAD\wesad_raw_windows.npz")

WINDOW_SEC = 30
STRIDE_SEC = 15
LABEL_HZ = 700
FS = {"EDA": 4, "TEMP": 4, "BVP": 64, "ACC": 32}
KEEP_LABELS = {1, 2, 3}
STRESS_LABEL = 2
MAJORITY_THRESHOLD = 0.90


def load_subject(path):
    with open(path, "rb") as f:
        try:
            data = pickle.load(f, encoding="latin1")
        except TypeError:
            data = pickle.load(f)
    return data


def window_label(label_slice):
    vals, counts = np.unique(label_slice, return_counts=True)
    top_idx = np.argmax(counts)
    top_label, top_frac = vals[top_idx], counts[top_idx] / len(label_slice)
    if top_frac < MAJORITY_THRESHOLD or top_label not in KEEP_LABELS:
        return None
    return int(top_label)


def process_subject(subject_dir):
    subject_id = subject_dir.name
    data = load_subject(subject_dir / f"{subject_id}.pkl")
    labels = np.asarray(data["label"]).ravel()
    wrist = data["signal"]["wrist"]
    signals = {mod: np.asarray(wrist[mod]) for mod in FS}

    n_label = len(labels)
    total_sec = n_label / LABEL_HZ
    window_lbl_samples = WINDOW_SEC * LABEL_HZ

    rows = {"EDA": [], "TEMP": [], "BVP": [], "ACC": [], "label": [], "subject": []}

    t = 0.0
    while t + WINDOW_SEC <= total_sec:
        lbl_start = int(round(t * LABEL_HZ))
        lbl_end = lbl_start + window_lbl_samples
        lbl = window_label(labels[lbl_start:lbl_end])
        if lbl is None:
            t += STRIDE_SEC
            continue

        ok = True
        window_data = {}
        for mod, fs in FS.items():
            start = int(round(t * fs))
            n_needed = WINDOW_SEC * fs
            end = start + n_needed
            arr = signals[mod]
            if end > len(arr):
                ok = False
                break
            window_data[mod] = arr[start:end]
        if not ok:
            t += STRIDE_SEC
            continue

        for mod in FS:
            rows[mod].append(window_data[mod].astype(np.float32))
        rows["label"].append(int(lbl == STRESS_LABEL))
        rows["subject"].append(subject_id)
        t += STRIDE_SEC

    return rows


def main():
    subject_dirs = sorted(
        [p for p in WESAD_DIR.iterdir() if p.is_dir() and p.name.startswith("S")],
        key=lambda p: int(p.name[1:]),
    )
    print(f"Found {len(subject_dirs)} subject directories")

    all_rows = {"EDA": [], "TEMP": [], "BVP": [], "ACC": [], "label": [], "subject": []}
    for sd in subject_dirs:
        rows = process_subject(sd)
        n = len(rows["label"])
        n_stress = sum(rows["label"])
        print(f"  {sd.name}: {n} windows, {n_stress} stress ({100*n_stress/n if n else 0:.1f}%)")
        for k in all_rows:
            all_rows[k].extend(rows[k])

    EDA = np.stack(all_rows["EDA"])
    TEMP = np.stack(all_rows["TEMP"])
    BVP = np.stack(all_rows["BVP"])
    ACC = np.stack(all_rows["ACC"])
    label = np.array(all_rows["label"], dtype=np.int64)
    subject = np.array(all_rows["subject"])

    print(f"\nShapes: EDA={EDA.shape} TEMP={TEMP.shape} BVP={BVP.shape} ACC={ACC.shape}")
    print(f"Total windows: {len(label)}  Stress: {label.sum()} ({100*label.mean():.1f}%)")

    np.savez_compressed(OUT_NPZ, EDA=EDA, TEMP=TEMP, BVP=BVP, ACC=ACC, label=label, subject=subject)
    print(f"Saved: {OUT_NPZ}")


if __name__ == "__main__":
    main()
