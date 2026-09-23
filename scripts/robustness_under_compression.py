"""
The actual combined experiment: does compressing the deployed model (pruning/quantizing
it) change how badly it degrades under corrupted input images (blur, JPEG compression,
etc.)? This is the piece that was missing after running the corruption-robustness
benchmark (docs/robustness_corruption_benchmark_2026-09-19.md) and the compression
analysis (papers/edge-ai-lightweight-deployment/) as two SEPARATE threads -- this script
runs the corruption sweep ON TOP OF the compressed model variants, not just the
full-precision one.

Literature check (2026-09-19) found this exact combination -- compression x corruption
interaction -- already studied in general computer vision (mixed/contested results: some
studies find compression hurts corruption robustness, some find it helps, it's
architecture-dependent), but never checked for dermatology/skin-disease models. So this
is "established methodology, first domain application" -- not a brand-new phenomenon.

Originally scoped to ShuffleNetV2-1.0x only (the deployed candidate). Extended
2026-09-19 (generalized via --model) to check whether the contrast-sensitivity
interaction found for ShuffleNetV2-1.0x is specific to that model or general: added
EfficientNet-B0 (SE/swish-containing, already flagged as quantization-fragile --
directly tests whether the interaction is an SE/swish phenomenon) and MobileNetV2-1.0x
(no SE/swish, a different architecture family entirely -- tests whether it's specific to
grouped-conv/channel-shuffle architectures like ShuffleNet). ResNet18 excluded, same as
the rest of this project's corruption-robustness work, since run_inference() below
assumes a sigmoid single-logit head, not ResNet18's softmax 2-class head. Each
architecture's own accuracy-vs-sparsity knee is looked up automatically -- not the same
sparsity value is reused across models. The full-precision (FP32) results are NOT
re-run for any model -- reused directly from the already-completed
results/robustness_corruption_results_<model>.json, since corruption + eval on a fixed
checkpoint is deterministic.

Three new variants tested per model, using the exact same pruning/quantization functions
already validated in scripts/edge_ai_extended_analysis.py (same knee-sparsity source,
same static INT8 calibration procedure) -- not reimplemented, imported directly, so
results are apples-to-apples comparable with that paper's numbers:
  - pruned_at_knee: unstructured L1 pruning at this model's own accuracy-vs-sparsity knee
  - static_int8: post-training static quantization (FX graph mode)
  - pruned30_plus_int8: 30% pruning (safely inside every model's knee) + static INT8

Usage:
    python robustness_under_compression.py --model shufflenet_v2_x1_0
    python robustness_under_compression.py --model efficientnet_b0
    python robustness_under_compression.py --model mobilenetv2_100
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT as PROJECT_ROOT, SKINDISEASE_DIR
from mem_guard import require_free_mb
from train_transfer_cnn import DEVICE
from robustness_corruption_benchmark import (
    CorruptedValDataset, CORRUPTIONS, SEVERITIES, VAL_MANIFEST, BATCH_SIZE,
    run_inference, metrics_from_probs, risk_coverage_curve,
)
from edge_ai_extended_analysis import (
    load_model, prune_to_sparsity, quantize_static, CuratedDataset, make_eval_transform,
    TRAIN_MANIFEST, SOURCE_JSON,
)

RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"
MIN_FREE_MB = 400


def get_knee_sparsity(model_name):
    """Same lookup edge_ai_extended_analysis.py uses: first sparsity level in the
    original pruning sweep with a >5pt accuracy drop or F1 collapsing to ~0."""
    with open(SOURCE_JSON) as f:
        source = json.load(f)
    sweep = source[model_name]["pruning_sweep"]
    baseline_acc = sweep[0]["val_metrics"]["accuracy"]
    knee = 0.9
    for row in sweep:
        if (baseline_acc - row["val_metrics"]["accuracy"] > 0.05) or row["val_metrics"].get("f1", 1.0) < 0.05:
            knee = row["sparsity_target"]
            break
    return knee


def make_train_loader():
    train_ds = CuratedDataset(TRAIN_MANIFEST, make_eval_transform())
    return DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)


def run_corruption_sweep(model, label):
    """Runs the full 36-condition sweep (clean + 7 corruptions x 5 severities) for one
    model variant, mirroring robustness_corruption_benchmark.py's main() loop exactly so
    results are directly comparable."""
    conditions = [("clean", None, 0)]
    for name, fn in CORRUPTIONS.items():
        for sev in SEVERITIES:
            conditions.append((name, fn, sev))

    results = {}
    t_start = time.time()
    for name, fn, sev in conditions:
        key = "clean" if fn is None else f"{name}_sev{sev}"
        ds = CorruptedValDataset(VAL_MANIFEST, corruption_fn=fn, severity=sev)
        y_true, y_prob = run_inference(model, ds)
        m = metrics_from_probs(y_true, y_prob)
        print(f"  [{label}] {key:28s} acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
              f"ece={m['ece']:.4f}", flush=True)
        results[key] = {"metrics": m, "y_true": y_true.tolist(), "y_prob": y_prob.tolist()}
    print(f"  [{label}] sweep done in {time.time() - t_start:.1f}s")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="shufflenet_v2_x1_0")
    args = parser.parse_args()
    model_name = args.model
    fp32_results_path = RESULTS_DIR / f"robustness_corruption_results_{model_name}.json"

    if not require_free_mb(MIN_FREE_MB, context="robustness-under-compression startup"):
        raise SystemExit("Not enough free RAM to safely start.")

    print(f"Loading existing FP32 corruption-sweep results for {model_name} "
          f"(not re-running -- deterministic, already on disk)")
    with open(fp32_results_path) as f:
        fp32_full = json.load(f)
    fp32_results = {k: v["metrics"] for k, v in fp32_full["conditions"].items()}

    knee = get_knee_sparsity(model_name)
    print(f"Knee sparsity for {model_name}: {knee}")

    train_loader = make_train_loader()

    all_variants = {"fp32": fp32_results}

    print(f"\n=== Variant: pruned_at_knee ({knee}) ===")
    m = load_model(model_name)
    m = prune_to_sparsity(m, knee)
    variant_results = run_corruption_sweep(m, f"pruned@{knee}")
    all_variants["pruned_at_knee"] = {k: v["metrics"] for k, v in variant_results.items()}
    del m

    print(f"\n=== Variant: static_int8 ===")
    m = load_model(model_name)
    m_q = quantize_static(m, train_loader)
    variant_results = run_corruption_sweep(m_q, "int8")
    all_variants["static_int8"] = {k: v["metrics"] for k, v in variant_results.items()}
    del m, m_q

    print(f"\n=== Variant: pruned30_plus_int8 ===")
    m = load_model(model_name)
    m = prune_to_sparsity(m, 0.3)
    m_q = quantize_static(m, train_loader)
    variant_results = run_corruption_sweep(m_q, "pruned30+int8")
    all_variants["pruned30_plus_int8"] = {k: v["metrics"] for k, v in variant_results.items()}
    del m, m_q

    out_path = RESULTS_DIR / f"robustness_under_compression_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump({"model": model_name, "knee_sparsity": knee, "variants": all_variants}, f, indent=2)
    print(f"\nSaved to {out_path}")

    # --- Comparison table: accuracy drop (sev1->sev5) per corruption, per variant ---
    print("\nAccuracy drop (severity 1 -> 5) per corruption type, per variant:")
    header = "variant".ljust(20) + "clean".rjust(8) + "".join(c[:10].rjust(12) for c in CORRUPTIONS)
    print(header)
    for variant, data in all_variants.items():
        clean_acc = data["clean"]["accuracy"]
        line = variant.ljust(20) + f"{clean_acc:.3f}".rjust(8)
        for c in CORRUPTIONS:
            drop = data[f"{c}_sev1"]["accuracy"] - data[f"{c}_sev5"]["accuracy"]
            line += f"{drop:+.3f}".rjust(12)
        print(line)

    # --- Plot: accuracy vs severity for the two dominant corruptions, all 4 variants ---
    for corruption in ["gaussian_blur", "jpeg_compression"]:
        fig, ax = plt.subplots(figsize=(8, 6))
        for variant, data in all_variants.items():
            clean_acc = data["clean"]["accuracy"]
            accs = [data[f"{corruption}_sev{s}"]["accuracy"] for s in SEVERITIES]
            ax.plot([0] + SEVERITIES, [clean_acc] + accs, marker="o", label=variant)
        ax.set_xlabel("Severity (0 = clean)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{model_name} -- {corruption} robustness by compression variant")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_plot = DOCS_DIR / f"robustness_under_compression_{model_name}_{corruption}_2026-09-19.png"
        fig.savefig(out_plot, dpi=150)
        plt.close(fig)
        print(f"Saved {out_plot}")

    print("\nDone. This never opened manifest_curated_v3_test.csv.")


if __name__ == "__main__":
    main()
