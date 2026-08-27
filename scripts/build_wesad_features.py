"""
Build the WESAD stress-classification feature table from the raw per-subject pickles
(dataset/WESAD/WESAD/S<n>/S<n>.pkl).

Scope: WRIST modalities only (Empatica E4: EDA 4Hz, TEMP 4Hz, BVP 64Hz, ACC 32Hz), not
chest -- this project's planned hardware replica is a wrist-worn device (see the fusion/
hardware plan), so the model should be trained on the same sensor set the hardware will
actually produce, not the higher-fidelity RespiBAN chest data no wearable here will have.

Task: binary "stress vs not-stress". WESAD's label channel (700Hz, chest-aligned) marks
1=baseline, 2=stress (TSST), 3=amusement, 4=meditation, 0/5/6/7=transient/undefined/other
protocol segments. Meditation (4) is dropped -- not every subject completed it, and it's
not a "not-stress" condition in the same sense as baseline/amusement are. Transient/
undefined segments (0, 5, 6, 7) are dropped as label noise between conditions. Positive
class = stress (2); negative class = baseline or amusement (1 or 3).

Windowing: non-overlapping 60-second windows (the window length used in the original
WESAD paper, Schmidt et al. 2018), majority-vote label per window from the 700Hz label
channel resampled to each window's wrist-clock timespan. A window is dropped if less than
90% of its labels agree (straddles a condition boundary) -- avoids mislabeling transition
windows as cleanly one class or the other.

Split unit: SUBJECT (15 subjects, S2-S11 + S13-S17 -- S1 and S12 don't exist in the
released set). Writes one feature table; the training script does leave-one-subject-out
CV itself rather than a fixed train/val/test split, since 15 subjects is too few for a
single held-out split to be reliable (same LOSO-CV reasoning validated for this exact
task by Chun et al. 2021 -- see docs/paper_review_adam_sensor_2026-08-26.md).

Writes dataset/WESAD/wesad_wrist_features.csv:
  subject_id, label (0=not-stress, 1=stress), <feature columns>
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

WESAD_DIR = Path(r"C:\Users\tishy\Documents\Honors\dataset\WESAD")
OUT_CSV = Path(r"C:\Users\tishy\Documents\Honors\dataset\WESAD\wesad_wrist_features.csv")

WINDOW_SEC = 60
LABEL_HZ = 700  # chest label channel sampling rate; all WESAD signals share this clock origin
FS = {"EDA": 4, "TEMP": 4, "BVP": 64, "ACC": 32}
KEEP_LABELS = {1, 2, 3}  # baseline, stress, amusement
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
    """Majority label for one window, or None if it straddles a condition boundary
    or is dominated by a dropped/transient label."""
    vals, counts = np.unique(label_slice, return_counts=True)
    top_idx = np.argmax(counts)
    top_label, top_frac = vals[top_idx], counts[top_idx] / len(label_slice)
    if top_frac < MAJORITY_THRESHOLD:
        return None
    if top_label not in KEEP_LABELS:
        return None
    return int(top_label)


def extract_window_features(sig_slices):
    """sig_slices: dict modality -> 1D (or Nx3 for ACC) array covering one window."""
    feats = {}

    for mod in ("EDA", "TEMP"):
        x = sig_slices[mod].astype(np.float64).ravel()
        feats[f"{mod}_mean"] = x.mean()
        feats[f"{mod}_std"] = x.std()
        feats[f"{mod}_min"] = x.min()
        feats[f"{mod}_max"] = x.max()
        feats[f"{mod}_range"] = x.max() - x.min()
        # linear trend across the window -- e.g. EDA tends to rise under stress,
        # peripheral TEMP tends to fall (vasoconstriction)
        t = np.arange(len(x))
        feats[f"{mod}_slope"] = np.polyfit(t, x, 1)[0] if len(x) > 1 else 0.0

    bvp = sig_slices["BVP"].astype(np.float64).ravel()
    feats["BVP_mean"] = bvp.mean()
    feats["BVP_std"] = bvp.std()
    feats["BVP_min"] = bvp.min()
    feats["BVP_max"] = bvp.max()
    # crude heart-rate proxy: dominant frequency in the plausible cardiac band (42-210 bpm)
    freqs = np.fft.rfftfreq(len(bvp), d=1.0 / FS["BVP"])
    spec = np.abs(np.fft.rfft(bvp - bvp.mean()))
    band = (freqs >= 0.7) & (freqs <= 3.5)
    if band.any() and spec[band].sum() > 0:
        feats["BVP_dominant_hz"] = freqs[band][np.argmax(spec[band])]
        feats["BVP_band_power"] = spec[band].sum()
    else:
        feats["BVP_dominant_hz"] = 0.0
        feats["BVP_band_power"] = 0.0

    acc = sig_slices["ACC"].astype(np.float64)
    mag = np.linalg.norm(acc, axis=1)
    feats["ACC_mag_mean"] = mag.mean()
    feats["ACC_mag_std"] = mag.std()
    for i, axis in enumerate("xyz"):
        feats[f"ACC_{axis}_mean"] = acc[:, i].mean()
        feats[f"ACC_{axis}_std"] = acc[:, i].std()

    return feats


def process_subject(subject_dir):
    subject_id = subject_dir.name
    pkl_path = subject_dir / f"{subject_id}.pkl"
    data = load_subject(pkl_path)

    labels = np.asarray(data["label"]).ravel()
    wrist = data["signal"]["wrist"]

    n_label_samples = len(labels)
    window_label_samples = WINDOW_SEC * LABEL_HZ
    n_windows = n_label_samples // window_label_samples

    rows = []
    for w in range(n_windows):
        lbl_start = w * window_label_samples
        lbl_end = lbl_start + window_label_samples
        lbl = window_label(labels[lbl_start:lbl_end])
        if lbl is None:
            continue

        frac_start = lbl_start / n_label_samples
        frac_end = lbl_end / n_label_samples

        sig_slices = {}
        ok = True
        for mod, fs in FS.items():
            arr = np.asarray(wrist[mod])
            n = arr.shape[0]
            s, e = int(frac_start * n), int(frac_end * n)
            if e - s < fs * 2:  # degenerate window, too few samples to featurize
                ok = False
                break
            sig_slices[mod] = arr[s:e]
        if not ok:
            continue

        feats = extract_window_features(sig_slices)
        feats["subject_id"] = subject_id
        feats["label"] = int(lbl == STRESS_LABEL)
        rows.append(feats)

    return rows


def main():
    subject_dirs = sorted(
        [p for p in WESAD_DIR.iterdir() if p.is_dir() and p.name.startswith("S")],
        key=lambda p: int(p.name[1:]),
    )
    print(f"Found {len(subject_dirs)} subject directories")

    all_rows = []
    for sd in subject_dirs:
        rows = process_subject(sd)
        n_stress = sum(r["label"] for r in rows)
        print(f"  {sd.name}: {len(rows)} windows, {n_stress} stress ({100*n_stress/len(rows) if rows else 0:.1f}%)")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    cols = ["subject_id", "label"] + [c for c in df.columns if c not in ("subject_id", "label")]
    df = df[cols]
    df.to_csv(OUT_CSV, index=False)

    print(f"\nTotal windows: {len(df)}")
    print(f"Stress windows: {df['label'].sum()} ({100*df['label'].mean():.1f}%)")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print(f"Feature columns: {len(cols) - 2}")
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
