import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import ROOT

# Final test-set numbers, all 9 models (8 candidates + ResNet18 baseline), from
# transfer_cnn_final_comparison_2026-09-18.json -- the single official test-set read.
# (name, size_mb, test_acc_pct, color, is_selected)
points = [
    ("ResNet18 (baseline)", 42.72, 81.07, "#4C72B0", False),
    ("EfficientNet-B0", 16.20, 77.51, "#DD8452", False),
    ("EfficientNet-Lite0", 13.74, 78.90, "#8172B3", False),
    ("MobileNetV2-1.0x", 9.34, 78.90, "#937860", False),
    ("ShuffleNetV2-1.0x (selected)", 5.45, 79.68, "#C44E52", True),
    ("RepGhostNet-0.5x", 4.85, 76.53, "#64B5CD", False),
    ("MobileNetV3-Small", 3.94, 76.73, "#55A868", False),
    ("SqueezeNet1.1", 3.03, 75.54, "#CCB974", False),
    ("ShuffleNetV2-0.5x", 1.94, 74.95, "#8C8C8C", False),
]

fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=200)
for name, size, acc, color, selected in points:
    if selected:
        ax.scatter(size, acc, s=220, color=color, zorder=4, edgecolor="black",
                   linewidth=1.6, marker="*")
    else:
        ax.scatter(size, acc, s=90, color=color, zorder=3, edgecolor="white", linewidth=0.8)
    offset = (8, 6)
    if "ResNet18" in name:
        offset = (8, -16)
    elif "ShuffleNetV2-1.0x" in name:
        offset = (10, 8)
    ax.annotate(name, (size, acc), textcoords="offset points", xytext=offset,
                fontsize=8.5, color="#333333", fontweight="bold" if selected else "normal")

ax.set_xscale("log")
ax.set_xlabel("Checkpoint size, MB (log scale)", fontsize=10)
ax.set_ylabel("Held-out test accuracy (%)", fontsize=10)
ax.set_title("Model size vs. test accuracy -- final comparison\n"
              "(all 8 lightweight candidates + ResNet18 baseline; "
              "none statistically distinguishable, Holm-corrected)",
              fontsize=10.5)
ax.set_ylim(73, 83)
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()
OUT_PATH = ROOT / "docs" / "architecture_comparison_size_vs_accuracy_2026-09-18.png"
fig.savefig(OUT_PATH, dpi=200)
print(f"saved {OUT_PATH}")
