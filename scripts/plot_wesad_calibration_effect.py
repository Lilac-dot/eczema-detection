"""Publication-quality figures for the WESAD personal-baseline calibration deep-dive.
Reads the subject-level table (wesad_subject_level_analysis.py output) and the raw
before/after fold CSVs. Writes 3 PNGs to docs/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from paths import WESAD_DIR, ROOT

DOCS = ROOT / "docs"
table = pd.read_csv(WESAD_DIR / "wesad_subject_level_table.csv").set_index("subject")
table = table.sort_values("delta_auc", ascending=False)

# --- Figure 1: paired before/after AUC, one line per subject ---
fig, ax = plt.subplots(figsize=(7, 6))
for i, (s, row) in enumerate(table.iterrows()):
    color = "#2a9d8f" if row["delta_auc"] > 0.02 else ("#e76f51" if row["delta_auc"] < -0.02 else "#8d99ae")
    ax.plot([0, 1], [row["pop_auc"], row["pers_auc"]], marker="o", color=color, alpha=0.85, linewidth=1.5)
    ax.annotate(s, (1.02, row["pers_auc"]), fontsize=8, va="center")
ax.set_xlim(-0.15, 1.35)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Population\nbaseline", "Personalized\nbaseline"])
ax.set_ylabel("Test AUC (LOSO-CV)")
ax.set_title("WESAD stress detection: paired AUC per subject,\npopulation vs. personalized baseline calibration")
ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(DOCS / "wesad_calibration_paired_auc_2026-09-16.png", dpi=150)
plt.close(fig)

# --- Figure 2: sorted delta-AUC bar chart ---
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#2a9d8f" if d > 0.02 else ("#e76f51" if d < -0.02 else "#8d99ae") for d in table["delta_auc"]]
ax.bar(table.index, table["delta_auc"], color=colors)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("ΔAUC (personalized − population)")
ax.set_title("Per-subject AUC change from personal-baseline calibration\n(sorted, largest gain to largest loss)")
for i, (s, d) in enumerate(table["delta_auc"].items()):
    ax.annotate(f"{d:+.3f}", (i, d + (0.01 if d >= 0 else -0.02)), ha="center", fontsize=7)
fig.tight_layout()
fig.savefig(DOCS / "wesad_calibration_delta_auc_2026-09-16.png", dpi=150)
plt.close(fig)

# --- Figure 3: F1 comparison (showing the flatness itself) ---
fig, ax = plt.subplots(figsize=(8, 5))
tf1 = table.sort_values("delta_f1", ascending=False)
colors_f1 = ["#2a9d8f" if d > 0.02 else ("#e76f51" if d < -0.02 else "#8d99ae") for d in tf1["delta_f1"]]
ax.bar(tf1.index, tf1["delta_f1"], color=colors_f1)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("ΔF1 (personalized − population)")
ax.set_title("Per-subject F1 change from personal-baseline calibration\n(no significant net shift, Wilcoxon p=0.85 -- shown for comparison with ΔAUC)")
fig.tight_layout()
fig.savefig(DOCS / "wesad_calibration_delta_f1_2026-09-16.png", dpi=150)
plt.close(fig)

print("Saved 3 figures to docs/:")
print(" - wesad_calibration_paired_auc_2026-09-16.png")
print(" - wesad_calibration_delta_auc_2026-09-16.png")
print(" - wesad_calibration_delta_f1_2026-09-16.png")
