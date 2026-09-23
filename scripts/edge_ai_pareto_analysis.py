"""
Pareto (accuracy vs. cost) analysis for the edge-AI compression paper revision, using
data already on disk from edge_simulation_all_architectures_2026-09-19.json (size,
latency, pruning sweep) and edge_ai_extended_analysis_2026-09-19.json (bootstrap CIs,
memory). No new model runs -- this is purely re-plotting/re-tabulating existing numbers,
so it needs no meaningful memory and is safe to run alongside anything else.

Two separate Pareto views, kept separate deliberately rather than combined into one
"cost" axis, because they measure genuinely different things and conflating them would
overstate what was actually measured:

1. Accuracy vs. on-disk size (FP32 vs. static INT8 only, across all 9 models) -- this is
   the one axis where compression produces a REAL, measured size reduction. Unstructured
   pruning's dense checkpoint does NOT shrink on disk (already established in
   papers/edge-ai-lightweight-deployment/edge_deployment_simulation_2026-09-19.md's
   pruning section) -- so pruned variants are deliberately excluded from this plot rather
   than plotted at a misleading "same size as FP32" point.
2. Accuracy vs. pruning sparsity, all 9 models on one plot -- shows where each
   architecture's own accuracy-vs-sparsity knee sits, since edge_ai_extended_analysis.py
   already established this varies by architecture (20%-60%).

A model/variant is marked "Pareto-optimal" on a given plot if no other point has both a
smaller cost (size or higher sparsity) AND equal-or-higher accuracy -- the standard
non-dominated-point definition.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import ROOT as PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "papers" / "edge-ai-lightweight-deployment"
SIM_JSON = OUT_DIR / "edge_simulation_all_architectures_2026-09-19.json"
EXT_JSON = OUT_DIR / "edge_ai_extended_analysis_2026-09-19.json"
DOCS_DIR = PROJECT_ROOT / "docs"

ALL_MODELS = ["resnet18", "efficientnet_b0", "efficientnet_lite0", "mobilenetv3_small",
              "mobilenetv2_100", "shufflenet_v2_x0_5", "shufflenet_v2_x1_0",
              "squeezenet1_1", "repghostnet_050"]


def pareto_front(points, minimize_x=True, maximize_y=True):
    """points: list of (label, x, y). Returns the subset that is non-dominated."""
    front = []
    for label, x, y in points:
        dominated = False
        for label2, x2, y2 in points:
            if label2 == label:
                continue
            better_or_equal_x = (x2 <= x) if minimize_x else (x2 >= x)
            better_or_equal_y = (y2 >= y) if maximize_y else (y2 <= y)
            strictly_better = (x2 < x if minimize_x else x2 > x) or (y2 > y if maximize_y else y2 < y)
            if better_or_equal_x and better_or_equal_y and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append((label, x, y))
    return front


def main():
    with open(SIM_JSON) as f:
        sim = json.load(f)
    with open(EXT_JSON) as f:
        ext = json.load(f)

    # --- Plot 1: accuracy vs on-disk size, FP32 vs static INT8, all 9 models ---
    points = []
    for model in ALL_MODELS:
        q = sim[model]["quantization"]
        fp32_acc = q["fp32"]["val_metrics"]["accuracy"]
        fp32_size = q["fp32"]["state_dict_disk_mb"]
        int8_acc = q["static_int8_fx"]["val_metrics"]["accuracy"]
        int8_size = q["static_int8_fx"]["state_dict_disk_mb"]
        points.append((f"{model} (FP32)", fp32_size, fp32_acc))
        points.append((f"{model} (INT8)", int8_size, int8_acc))

    front = pareto_front(points, minimize_x=True, maximize_y=True)
    front_labels = {p[0] for p in front}

    fig, ax = plt.subplots(figsize=(10, 7))
    for label, x, y in points:
        is_int8 = "INT8" in label
        marker = "s" if is_int8 else "o"
        color = "tab:orange" if label in front_labels else ("tab:blue" if is_int8 else "tab:gray")
        ax.scatter(x, y, marker=marker, s=90 if label in front_labels else 50,
                    color=color, edgecolors="black" if label in front_labels else "none",
                    linewidths=1.5, zorder=3 if label in front_labels else 2)
        if label in front_labels:
            ax.annotate(label.replace(" (", "\n("), (x, y), fontsize=7, xytext=(4, 4),
                        textcoords="offset points")
    ax.set_xlabel("On-disk checkpoint size (MB)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Accuracy vs. size Pareto front -- FP32 vs. static INT8, all 9 architectures\n"
                 "(circle=FP32, square=INT8; orange/black-edged = Pareto-optimal)")
    fig.tight_layout()
    out1 = DOCS_DIR / "edge_ai_pareto_accuracy_vs_size_2026-09-19.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Saved {out1}")
    print("\nPareto-optimal points (accuracy vs. size):")
    for label, x, y in sorted(front, key=lambda p: p[1]):
        print(f"  {label}: size={x:.2f}MB acc={y:.4f}")

    # --- Plot 2: accuracy vs pruning sparsity, all 9 models ---
    fig, ax = plt.subplots(figsize=(10, 7))
    knees = {}
    for model in ALL_MODELS:
        sweep = sim[model]["pruning_sweep"]
        sparsities = [r["sparsity_target"] for r in sweep]
        accs = [r["val_metrics"]["accuracy"] for r in sweep]
        ax.plot(sparsities, accs, marker="o", markersize=4, label=model)
        base_acc = accs[0]
        knee = 0.9
        for r in sweep:
            if base_acc - r["val_metrics"]["accuracy"] > 0.05 or r["val_metrics"]["f1"] < 0.05:
                knee = r["sparsity_target"]
                break
        knees[model] = knee
    ax.axhline(0.5, color="gray", linestyle=":", label="~chance (this dataset)")
    ax.set_xlabel("Pruning sparsity")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Accuracy vs. pruning sparsity, all 9 architectures")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out2 = DOCS_DIR / "edge_ai_pareto_accuracy_vs_sparsity_2026-09-19.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out2}")
    print("\nAccuracy-vs-sparsity knee per architecture (first sparsity with >5pt drop or F1<0.05):")
    for model, knee in sorted(knees.items(), key=lambda kv: kv[1]):
        print(f"  {model}: {knee}")

    out_json = OUT_DIR / "edge_ai_pareto_analysis_2026-09-19.json"
    with open(out_json, "w") as f:
        json.dump({
            "accuracy_vs_size_points": [{"label": l, "size_mb": x, "accuracy": y} for l, x, y in points],
            "accuracy_vs_size_pareto_front": [{"label": l, "size_mb": x, "accuracy": y} for l, x, y in front],
            "pruning_knees": knees,
        }, f, indent=2)
    print(f"\nSaved {out_json}")


if __name__ == "__main__":
    main()
