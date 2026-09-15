"""
Build the AAUWSS sleep-disruption feature table from the aligned raw window tensors
(dataset/AAUWSS/aauwss_raw_windows.npz, produced by build_aauwss_raw_windows.py).

Reuses wesad_features.extract_window_features() unmodified -- AAUWSS's Empatica E4
export uses the identical sampling rates as WESAD's wrist data (EDA/TEMP 4Hz, BVP 64Hz,
ACC 32Hz), so the same window-statistics feature set applies directly.

Personal-baseline calibration mirrors build_wesad_features.py's
add_personal_baseline_features() exactly: for every feature, a `_rel` column expresses
that window's value relative to this subject's own baseline-condition (condition==1,
i.e. asleep -- see build_aauwss_raw_windows.py's docstring for why "asleep" plays the
role WESAD's calm pre-stress period plays) average of the same feature.

Writes dataset/AAUWSS/aauwss_features.csv:
  subject_id, label (0=asleep, 1=Wake/disrupted), condition, stage, <feature columns>
"""
import numpy as np
import pandas as pd

from paths import AAUWSS_DIR
from wesad_features import extract_window_features

IN_NPZ = AAUWSS_DIR / "aauwss_raw_windows.npz"
OUT_CSV = AAUWSS_DIR / "aauwss_features.csv"


def add_personal_baseline_features(df, feat_cols):
    """Same logic as build_wesad_features.py's function of the same name -- see there
    for the full rationale (between-subject nuisance removal via baseline-relative
    deltas)."""
    baseline_means = df[df["condition"] == 1].groupby("subject_id")[feat_cols].mean()
    rel = df[feat_cols].copy()
    for subj in df["subject_id"].unique():
        mask = df["subject_id"] == subj
        if subj in baseline_means.index:
            rel.loc[mask, feat_cols] = df.loc[mask, feat_cols] - baseline_means.loc[subj]
        else:
            print(f"  WARNING: no baseline (asleep) windows for {subj}, leaving raw "
                  f"features unadjusted")
    rel.columns = [f"{c}_rel" for c in feat_cols]
    return pd.concat([df, rel], axis=1)


def main():
    npz = np.load(IN_NPZ, allow_pickle=True)
    # Materialize each array ONCE -- npz["X"] re-decompresses the whole array from the
    # .npz zip archive on every access (NpzFile.__getitem__ doesn't cache), so indexing
    # npz["EDA"][i] inside the loop would silently re-decompress all of EDA on every one
    # of the 9700 iterations. This one-time load avoids that.
    EDA, TEMP, BVP, ACC = npz["EDA"], npz["TEMP"], npz["BVP"], npz["ACC"]
    label_arr, condition_arr = npz["label"], npz["condition"]
    subject_arr, stage_arr = npz["subject"], npz["stage"]
    n = len(label_arr)
    print(f"Loaded {n} windows from {IN_NPZ}")

    rows = []
    for i in range(n):
        sig_slices = {
            "EDA": EDA[i], "TEMP": TEMP[i],
            "BVP": BVP[i], "ACC": ACC[i],
        }
        feats = extract_window_features(sig_slices)
        feats["subject_id"] = str(npz["subject"][i])
        feats["label"] = int(npz["label"][i])
        feats["condition"] = int(npz["condition"][i])
        feats["stage"] = str(npz["stage"][i])
        rows.append(feats)

    df = pd.DataFrame(rows)
    base_feat_cols = [c for c in df.columns if c not in
                       ("subject_id", "label", "condition", "stage")]
    df = add_personal_baseline_features(df, base_feat_cols)

    cols = ["subject_id", "label", "condition", "stage"] + \
           [c for c in df.columns if c not in ("subject_id", "label", "condition", "stage")]
    df = df[cols]
    df.to_csv(OUT_CSV, index=False)

    print(f"\nTotal windows: {len(df)}")
    print(f"Wake (disrupted) windows: {df['label'].sum()} ({100 * df['label'].mean():.1f}%)")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print(f"Feature columns: {len(cols) - 4} (incl. {len(base_feat_cols)} "
          f"personal-baseline-relative)")
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
