# -*- coding: utf-8 -*-
"""Backend-consistency check, 2026-09-19: does static INT8 quantization behave
differently on qnnpack (this Mac / ARM -- the realistic backend for an eventual
edge/wearable device) than it did on x86/fbgemm (the original Windows dev machine)?

Triggered by a real, surprising finding: re-running the compression x corruption
experiment for MobileNetV2-1.0x and EfficientNet-B0 on this Mac showed static INT8
collapsing them to near-chance / fully degenerate classifiers, even though the SAME
quantization procedure on the SAME checkpoints only cost ~1-7 points of accuracy on the
original x86 machine (already on record in edge_ai_extended_analysis_2026-09-19.json).
This script checks whether that's specific to those two architectures or general, across
all 9.

Only the static-INT8 step is redone here (on qnnpack, this machine's only supported
quantized engine) -- FP32 baselines, pruning results, and the SE/swish ablation are
backend-independent (no quantization involved) and are NOT recomputed; they're pulled
directly from the existing edge_ai_extended_analysis_2026-09-19.json for comparison.
Validation set only, same as every other script in this project.
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mem_guard import require_free_mb  # noqa: E402
from paths import ROOT  # noqa: E402
from edge_ai_extended_analysis import (  # noqa: E402
    CuratedDataset, make_eval_transform, get_predictions, point_metrics,
    bootstrap_ci, load_model, quantize_static, quantized_op_coverage,
    VAL_MANIFEST, TRAIN_MANIFEST, ALL_MODELS, QUANT_BACKEND,
)

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)

OUT_DIR = ROOT / "papers" / "edge-ai-lightweight-deployment"
OUT_JSON = OUT_DIR / "edge_ai_qnnpack_backend_check_2026-09-19.json"
EXT_JSON = OUT_DIR / "edge_ai_extended_analysis_2026-09-19.json"
# All 9 recomputed fresh here (rather than reusing the 3 already spot-checked via
# robustness_under_compression.py) so every model's qnnpack number comes from the same
# code path with the same CI/coverage treatment -- one consistent source of truth.


def main():
    if not require_free_mb(400, context="startup"):
        print("Not enough free RAM to start. Aborting.")
        return

    print(f"Quantized backend in use: {QUANT_BACKEND}")
    with open(EXT_JSON) as f:
        ext = json.load(f)

    eval_tf = make_eval_transform()
    val_ds = CuratedDataset(VAL_MANIFEST, eval_tf)
    train_ds = CuratedDataset(TRAIN_MANIFEST, eval_tf)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)

    out = {}
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            out = json.load(f)
        print(f"Resuming -- already done: {list(out.keys())}", flush=True)

    for model_name in ALL_MODELS:
        if model_name in out:
            print(f"\n=== {model_name}: already done, skipping ===", flush=True)
            continue
        if not require_free_mb(400, context=f"before {model_name}"):
            print(f"Stopping before {model_name} (low RAM). Re-run to resume.")
            break

        print(f"\n=== qnnpack static INT8: {model_name} ===", flush=True)
        t0 = time.time()
        is_resnet18 = model_name == "resnet18"

        x86_entry = ext["per_model"][model_name]["static_int8"]
        x86_point = x86_entry["point"]

        m = load_model(model_name)
        m_q = quantize_static(m, train_loader)
        labels, probs = get_predictions(m_q, val_loader, is_resnet18)
        qnnpack_point = point_metrics(labels, probs)
        qnnpack_ci = bootstrap_ci(labels, probs)
        coverage = quantized_op_coverage(m_q)

        out[model_name] = {
            "x86_fbgemm_backend_original": x86_point,
            "qnnpack_backend_this_mac": {
                "point": qnnpack_point,
                "bootstrap_ci": qnnpack_ci,
                "op_coverage": coverage,
            },
            "accuracy_delta_qnnpack_minus_x86": qnnpack_point["accuracy"] - x86_point["accuracy"],
            "collapsed_on_qnnpack": qnnpack_point["f1"] < 0.05,
            "total_time_s": time.time() - t0,
        }
        print(f"  x86: acc={x86_point['accuracy']:.4f} sens={x86_point['sensitivity_recall']:.4f} "
              f"spec={x86_point['specificity']:.4f}", flush=True)
        print(f"  qnnpack: acc={qnnpack_point['accuracy']:.4f} sens={qnnpack_point['sensitivity_recall']:.4f} "
              f"spec={qnnpack_point['specificity']:.4f} "
              f"{'*** COLLAPSED ***' if out[model_name]['collapsed_on_qnnpack'] else ''}", flush=True)

        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=2)
        del m, m_q
        gc.collect()

    print(f"\nAll done. Results at {OUT_JSON}", flush=True)
    print("\nSummary (accuracy: x86 vs qnnpack, delta):")
    for model_name, entry in out.items():
        x86_acc = entry["x86_fbgemm_backend_original"]["accuracy"]
        qn_acc = entry["qnnpack_backend_this_mac"]["point"]["accuracy"]
        flag = " *** COLLAPSED ON QNNPACK ***" if entry["collapsed_on_qnnpack"] else ""
        print(f"  {model_name:22s} x86={x86_acc:.4f}  qnnpack={qn_acc:.4f}  "
              f"delta={qn_acc - x86_acc:+.4f}{flag}")


if __name__ == "__main__":
    main()
