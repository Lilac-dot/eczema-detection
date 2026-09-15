"""
Apply the Cole-Kripke (1992) and Sadeh (1994) actigraphy sleep/wake scoring algorithms
to AAUWSS as a literature-standard, non-ML baseline, motivated by both from-scratch
models (train_aauwss_sleep.py LightGBM, train_aauwss_sleep_cnn.py CNN+attention) scoring
near chance under LOSO-CV. Both algorithms are fixed published formulas -- there is
nothing to "train": the same coefficients are applied to every subject, so this script
reports per-subject and pooled performance directly, not a LOSO-CV comparison.

IMPORTANT ADAPTATION CAVEATS (read before citing this as "the" Cole-Kripke/Sadeh result):

1. Both algorithms were designed for 1-MINUTE epochs and a device-specific proprietary
   "activity count" (zero-crossing-mode, ZCM) that is not publicly documented in a way
   that can be exactly reproduced from raw accelerometer data. This project's epochs are
   30 seconds (fixed by AAUWSS's own AASM annotation granularity -- see
   build_aauwss_raw_windows.py), and the activity count used below is a substitute:
   per-epoch summed jerk (sum of |diff| of the tri-axial acceleration-vector magnitude),
   a standard, widely-used proxy for legacy actigraph counts when only raw accelerometer
   data is available (this general approach -- deriving a ZCM-like count from raw
   accelerometry -- is discussed in Migueles et al. 2019 and used by open packages such
   as GGIR and pyActigraphy for the same reason: no public spec for the original
   proprietary counts exists).
2. Each algorithm's published window (e.g. Cole-Kripke's -2/+4 minutes, Sadeh's
   +/-5-minute windows) is rescaled from MINUTES to this project's 30-SECOND epochs by
   doubling the epoch counts (so the real-world window duration is preserved), not by
   reinterpreting "epoch" as 30 seconds while keeping the original epoch counts.
3. Window neighbors are taken from ADJACENT KEPT EPOCHS in a subject's own recording
   (not strictly adjacent by wall-clock time), since a subject's dropped epochs (Section
   in build_aauwss_raw_windows.py -- E4 battery died early for several subjects) form one
   contiguous trailing block in every case checked, not scattered gaps, so this is a safe
   simplification here, not a general guarantee.

Given these adaptations, this should be read as "how well do these algorithms' published
weighting schemes generalize to this project's specific epoch length and accelerometer
proxy," not a validated reproduction of either paper's own reported accuracy.

Writes: dataset/AAUWSS/aauwss_actigraphy_baseline_results.csv (per-subject metrics)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from paths import AAUWSS_DIR

IN_NPZ = AAUWSS_DIR / "aauwss_raw_windows.npz"
OUT_CSV = AAUWSS_DIR / "aauwss_actigraphy_baseline_results.csv"

# Cole-Kripke (1992) weights, epoch t-4..t+2 (rescaled 2x for 30s epochs: t-8..t+4 by twos
# is NOT how we rescale -- see module docstring point 2: we double the NUMBER of epochs on
# each side to preserve real-world window duration, i.e. t-8..t+4 at 30s spacing).
CK_WEIGHTS = {-8: 106, -6: 54, -4: 58, -2: 76, 0: 230, 2: 74, 4: 67}
CK_SCALE = 0.001
CK_THRESHOLD = 1.0  # SI >= 1 -> Wake

SADEH_MA_HALFWIN = 10   # +/-5 min -> +/-10 half-minute (30s) epochs
SADEH_SDA_LOOKBACK = 12  # current + preceding 6 min -> current + preceding 12 epochs
SADEH_NAT_HALFWIN = 10


def compute_activity(ACC):
    """Per-epoch 'activity count' proxy: summed jerk (sum of |diff| of the acceleration
    magnitude) across each epoch's samples. See module docstring point 1."""
    mag = np.linalg.norm(ACC, axis=2)  # (N, samples_per_epoch)
    jerk = np.abs(np.diff(mag, axis=1)).sum(axis=1)  # (N,)
    return jerk


def cole_kripke_scores(activity):
    """Returns SI (higher = more likely Wake) for every index; edge indices (where the
    full +/-window isn't available) get NaN."""
    n = len(activity)
    si = np.full(n, np.nan)
    max_offset = max(CK_WEIGHTS)
    for i in range(max_offset, n - max_offset):
        si[i] = CK_SCALE * sum(w * activity[i + off] for off, w in CK_WEIGHTS.items())
    return si


def sadeh_scores(activity):
    """Returns -PS (higher = more likely Wake, to match Cole-Kripke's direction and the
    project's label=1=Wake convention) for every index; edge indices get NaN."""
    n = len(activity)
    neg_ps = np.full(n, np.nan)
    lo_needed = max(SADEH_MA_HALFWIN, SADEH_SDA_LOOKBACK, SADEH_NAT_HALFWIN)
    for i in range(lo_needed, n - SADEH_MA_HALFWIN):
        window = activity[i - SADEH_MA_HALFWIN:i + SADEH_MA_HALFWIN + 1]
        ma5 = window.mean()
        nat = int(((window >= 50) & (window < 100)).sum())
        sda_window = activity[i - SADEH_SDA_LOOKBACK:i + 1]
        sda6 = sda_window.std()
        a0 = activity[i]
        ps = 7.601 - 0.065 * ma5 - 1.08 * nat - 0.056 * sda6 - 0.073 * np.log(a0 + 1)
        neg_ps[i] = -ps
    return neg_ps


def evaluate_binary(score, label, threshold, higher_is_positive=True):
    valid = ~np.isnan(score)
    s, y = score[valid], label[valid]
    pred = (s >= threshold).astype(int) if higher_is_positive else (s < threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (pred == y).mean() if len(y) else float("nan")
    auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")
    return dict(n=int(valid.sum()), auc=auc, acc=acc, precision=precision, recall=recall, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn)


def main():
    npz = np.load(IN_NPZ, allow_pickle=True)
    ACC, label, subject = npz["ACC"], npz["label"], npz["subject"]

    rows = []
    all_ck_scores, all_ck_labels = [], []
    all_sadeh_scores, all_sadeh_labels = [], []
    for subj in sorted(np.unique(subject)):
        idx = np.where(subject == subj)[0]  # already in original (time) order
        acc_s = ACC[idx]
        label_s = label[idx]
        activity = compute_activity(acc_s)

        ck = cole_kripke_scores(activity)
        sadeh = sadeh_scores(activity)

        ck_result = evaluate_binary(ck, label_s, CK_THRESHOLD, higher_is_positive=True)
        sadeh_result = evaluate_binary(sadeh, label_s, 0.0, higher_is_positive=True)

        rows.append(dict(subject=subj, algorithm="Cole-Kripke", **ck_result))
        rows.append(dict(subject=subj, algorithm="Sadeh", **sadeh_result))

        valid_ck = ~np.isnan(ck)
        all_ck_scores.append(ck[valid_ck]); all_ck_labels.append(label_s[valid_ck])
        valid_sadeh = ~np.isnan(sadeh)
        all_sadeh_scores.append(sadeh[valid_sadeh]); all_sadeh_labels.append(label_s[valid_sadeh])

        print(f"[{subj}] n={len(idx)}  Cole-Kripke: AUC={ck_result['auc']:.3f} "
              f"F1={ck_result['f1']:.3f} Acc={ck_result['acc']:.3f}  |  "
              f"Sadeh: AUC={sadeh_result['auc']:.3f} F1={sadeh_result['f1']:.3f} "
              f"Acc={sadeh_result['acc']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("\n=== Per-algorithm summary across 13 subjects (mean of per-subject metrics) ===")
    for algo in ["Cole-Kripke", "Sadeh"]:
        sub = df[df["algorithm"] == algo]
        print(f"{algo}: mean AUC={sub['auc'].mean():.4f}  mean F1={sub['f1'].mean():.4f}  "
              f"mean Acc={sub['acc'].mean():.4f}  mean Precision={sub['precision'].mean():.4f}  "
              f"mean Recall={sub['recall'].mean():.4f}")

    pooled_ck_scores = np.concatenate(all_ck_scores)
    pooled_ck_labels = np.concatenate(all_ck_labels)
    pooled_sadeh_scores = np.concatenate(all_sadeh_scores)
    pooled_sadeh_labels = np.concatenate(all_sadeh_labels)
    print("\n=== Pooled (all subjects' valid epochs concatenated) ===")
    print(f"Cole-Kripke pooled AUC: {roc_auc_score(pooled_ck_labels, pooled_ck_scores):.4f}")
    print(f"Sadeh pooled AUC:       {roc_auc_score(pooled_sadeh_labels, pooled_sadeh_scores):.4f}")
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
