"""
Subject-level deep-dive on the personal-baseline calibration effect: combines the
before/after LOSO fold CSVs into one table, computes physiological baseline-mismatch
measures per subject, tests the mismatch-vs-improvement correlation, and compares the
three largest-gain subjects (S5, S14, S17) against non-improving subjects on the
underlying signal characteristics.

Reads (does not modify): dataset/WESAD/wesad_loso_fold_results.csv (after),
dataset/WESAD/wesad_loso_fold_results_before_calibration.csv (before),
dataset/WESAD/wesad_wrist_features.csv (raw per-window features).

Writes: dataset/WESAD/wesad_subject_level_table.csv, prints all statistics used in
docs/wesad_calibration_deep_dive_2026-09-16.md Sections 2, 4, 5, 8.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from paths import WESAD_DIR

before = pd.read_csv(WESAD_DIR / "wesad_loso_fold_results_before_calibration.csv").set_index("subject")
after = pd.read_csv(WESAD_DIR / "wesad_loso_fold_results.csv").set_index("subject")
feats = pd.read_csv(WESAD_DIR / "wesad_wrist_features.csv")

subjects = sorted(before.index, key=lambda s: int(s[1:]))

# ---- Section 2: complete subject-level table ----
rows = []
for s in subjects:
    b, a = before.loc[s], after.loc[s]
    rows.append(dict(
        subject=s,
        pop_auc=b["test_auc"], pers_auc=a["test_auc"], delta_auc=a["test_auc"] - b["test_auc"],
        pop_f1=b["f1"], pers_f1=a["f1"], delta_f1=a["f1"] - b["f1"],
        pop_precision=b["precision"], pers_precision=a["precision"],
        pop_recall=b["recall"], pers_recall=a["recall"],
        pop_threshold=b["threshold"], pers_threshold=a["threshold"],
        pop_acc=b["acc"], pers_acc=a["acc"],
    ))
table = pd.DataFrame(rows).set_index("subject")

NEGLIGIBLE = 0.02
table["category"] = np.where(table["delta_auc"] > NEGLIGIBLE, "improved",
                       np.where(table["delta_auc"] < -NEGLIGIBLE, "worsened", "negligible"))

print("=== Subject-level table (AUC) ===")
print(table[["pop_auc", "pers_auc", "delta_auc", "category"]].sort_values("delta_auc", ascending=False))
print()
print("Category counts:", table["category"].value_counts().to_dict())
print(f"Largest positive: {table['delta_auc'].idxmax()} ({table['delta_auc'].max():+.4f})")
print(f"Largest negative: {table['delta_auc'].idxmin()} ({table['delta_auc'].min():+.4f})")

# ---- Section 8: outlier check on delta_auc (1.5x IQR rule) ----
q1, q3 = table["delta_auc"].quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = table[(table["delta_auc"] < lo) | (table["delta_auc"] > hi)]
print(f"\n=== Outlier check (1.5xIQR on delta_auc): bounds [{lo:.4f}, {hi:.4f}] ===")
print("Outliers:", list(outliers.index) if len(outliers) else "none")

# ---- Section 4: baseline-mismatch hypothesis ----
baseline_windows = feats[feats["condition"] == 1]
pop_eda_mean = baseline_windows["EDA_mean"].mean()
pop_eda_std = baseline_windows["EDA_mean"].std()
subj_baseline_eda = baseline_windows.groupby("subject_id")["EDA_mean"].mean()

table["subj_calm_eda_mean"] = subj_baseline_eda.reindex(table.index)
table["eda_mismatch_raw"] = table["subj_calm_eda_mean"] - pop_eda_mean
table["eda_mismatch_z"] = table["eda_mismatch_raw"] / pop_eda_std

rho, p = spearmanr(table["eda_mismatch_z"].abs(), table["delta_auc"])
# bootstrap CI on rho, resampling subjects
rng = np.random.RandomState(42)
n = len(table)
boot_rhos = []
x = table["eda_mismatch_z"].abs().values
y = table["delta_auc"].values
for _ in range(10000):
    idx = rng.randint(0, n, n)
    if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
        continue
    r, _ = spearmanr(x[idx], y[idx])
    boot_rhos.append(r)
boot_rhos = np.array(boot_rhos)
ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

print(f"\n=== Baseline-mismatch (|EDA z-mismatch|) vs delta_auc ===")
print(table[["eda_mismatch_raw", "eda_mismatch_z", "delta_auc"]].sort_values("delta_auc", ascending=False))
print(f"Spearman rho={rho:.4f} p={p:.4f}  bootstrap 95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]")

# ---- Section 5: S5/S14/S17 vs comparison subjects (S2, S6, S4 -- non-improvers) ----
focus = ["S5", "S14", "S17"]
compare = ["S2", "S6", "S4"]
detail_rows = []
for s in focus + compare:
    calm = feats[(feats["subject_id"] == s) & (feats["condition"] == 1)]
    stress = feats[(feats["subject_id"] == s) & (feats["condition"] == 2)]
    detail_rows.append(dict(
        subject=s, group="focus" if s in focus else "comparison",
        delta_auc=table.loc[s, "delta_auc"],
        eda_mismatch_z=table.loc[s, "eda_mismatch_z"],
        calm_eda_mean=calm["EDA_mean"].mean(), calm_eda_std=calm["EDA_mean"].std(),
        stress_eda_mean=stress["EDA_mean"].mean(),
        calm_temp_mean=calm["TEMP_mean"].mean(), calm_temp_std=calm["TEMP_mean"].std(),
        stress_temp_mean=stress["TEMP_mean"].mean(),
        n_calm=len(calm), n_stress=len(stress),
        n_total=len(feats[feats["subject_id"] == s]),
        stress_frac=len(stress) / len(feats[feats["subject_id"] == s]),
    ))
detail = pd.DataFrame(detail_rows).set_index("subject")
print("\n=== S5/S14/S17 (focus) vs S2/S6/S4 (comparison, non-improvers) ===")
print(detail.T)

table.to_csv(WESAD_DIR / "wesad_subject_level_table.csv")
detail.to_csv(WESAD_DIR / "wesad_focus_subject_detail.csv")
print(f"\nSaved: {WESAD_DIR / 'wesad_subject_level_table.csv'}")
print(f"Saved: {WESAD_DIR / 'wesad_focus_subject_detail.csv'}")
