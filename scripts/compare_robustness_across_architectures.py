"""
Aggregates results/robustness_corruption_results_<model>.json across every architecture
that has one, to check whether the blur/JPEG-dominant degradation pattern found for
ShuffleNetV2-1.0x (docs/robustness_corruption_benchmark_2026-09-19.md) is specific to that
model or general across architectures.

Usage:
    python compare_robustness_across_architectures.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import ROOT as PROJECT_ROOT
from train_transfer_cnn import MODEL_CONFIGS

RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"
CORRUPTIONS = ["gaussian_blur", "brightness_down", "brightness_up", "contrast_down",
               "gaussian_noise", "jpeg_compression", "color_shift_warm"]


def load_all():
    out = {}
    for model in MODEL_CONFIGS:
        path = RESULTS_DIR / f"robustness_corruption_results_{model}.json"
        if path.exists():
            with open(path) as f:
                out[model] = json.load(f)
    return out


def main():
    all_results = load_all()
    print(f"Found results for {len(all_results)}/{len(MODEL_CONFIGS)} architectures: "
          f"{list(all_results.keys())}")
    if len(all_results) < 2:
        print("Fewer than 2 architectures done -- nothing to compare yet.")
        return

    # Table: accuracy drop (sev1 -> sev5) per corruption per model.
    rows = []
    for model, data in all_results.items():
        clean_acc = data["conditions"]["clean"]["metrics"]["accuracy"]
        row = {"model": model, "clean_acc": clean_acc}
        for c in CORRUPTIONS:
            sev1 = data["conditions"][f"{c}_sev1"]["metrics"]["accuracy"]
            sev5 = data["conditions"][f"{c}_sev5"]["metrics"]["accuracy"]
            row[f"{c}_drop"] = sev1 - sev5
        rows.append(row)

    print("\nAccuracy drop (severity 1 -> 5) per corruption type, per architecture:")
    header = "model".ljust(22) + "clean".rjust(8) + "".join(c[:10].rjust(12) for c in CORRUPTIONS)
    print(header)
    for row in rows:
        line = row["model"].ljust(22) + f"{row['clean_acc']:.3f}".rjust(8)
        for c in CORRUPTIONS:
            line += f"{row[f'{c}_drop']:+.3f}".rjust(12)
        print(line)

    # Rank corruptions by mean drop across all available architectures.
    mean_drop = {}
    for c in CORRUPTIONS:
        vals = [row[f"{c}_drop"] for row in rows]
        mean_drop[c] = sum(vals) / len(vals)
    ranked = sorted(mean_drop.items(), key=lambda kv: -kv[1])
    print("\nCorruption types ranked by mean accuracy drop across all tested architectures:")
    for name, drop in ranked:
        print(f"  {name}: {drop:+.4f}")

    # Plot: accuracy-vs-severity for the worst corruption (whatever ranks #1 above),
    # one line per architecture, to visually check whether the pattern is consistent.
    worst_corruption = ranked[0][0]
    fig, ax = plt.subplots(figsize=(8, 6))
    for model, data in all_results.items():
        clean_acc = data["conditions"]["clean"]["metrics"]["accuracy"]
        accs = [data["conditions"][f"{worst_corruption}_sev{s}"]["metrics"]["accuracy"]
                for s in [1, 2, 3, 4, 5]]
        ax.plot([0, 1, 2, 3, 4, 5], [clean_acc] + accs, marker="o", label=model)
    ax.set_xlabel("Severity (0 = clean)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Cross-architecture accuracy under {worst_corruption}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_plot = DOCS_DIR / "robustness_cross_architecture_comparison_2026-09-19.png"
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)
    print(f"\nSaved comparison plot to {out_plot}")

    out_json = RESULTS_DIR / "robustness_cross_architecture_summary.json"
    with open(out_json, "w") as f:
        json.dump({"n_architectures": len(all_results), "rows": rows,
                    "mean_drop_ranked": ranked}, f, indent=2)
    print(f"Saved summary table to {out_json}")


if __name__ == "__main__":
    main()
