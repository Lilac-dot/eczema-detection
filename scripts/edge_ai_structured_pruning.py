# -*- coding: utf-8 -*-
"""Structured (channel) pruning for the edge-AI compression paper revision (2026-09-19),
addressing task 2d and directly testing task 1g's claim in this project's own existing
docs: that unstructured L1 pruning ("prune.global_unstructured" in
edge_ai_extended_analysis.py) does NOT actually shrink the on-disk checkpoint or reduce
parameter count, because it only zeroes weights without removing them. This script uses
torch-pruning (dependency installed earlier, never previously exercised) to physically
remove channels, and directly measures the contrast: does structured pruning actually
shrink the model on disk, unlike unstructured pruning?

Validation set only, no fine-tuning after pruning (that is a separate, harder question --
see edge_ai_pruning_finetune.py for pruning WITH fine-tuning). This script answers a
narrower question: at a given "pruning_ratio" (torch-pruning's per-layer channel
fraction), how much does accuracy drop with NO recovery step, and how much does the
checkpoint actually shrink?

KNOWN LIMITATION, discovered and verified during development, not glossed over: both
ShuffleNetV2 variants (0.5x and 1.0x -- the latter being this project's actual deployed
candidate) fail with a ZeroDivisionError inside torch-pruning's dependency-graph handling
of ShuffleNet's channel-shuffle / grouped-convolution structure. This is a real library
limitation with this specific architecture family, not a bug in this script -- confirmed
by isolating the failure to the pruner.step() call alone, on an unmodified architecture.
Both ShuffleNetV2 variants are therefore excluded from this experiment; the other 7 of 9
trained architectures (including the deployed candidate's own un-prunable status) are
reported honestly below.
"""
import copy
import gc
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch_pruning as tp
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mem_guard import require_free_mb  # noqa: E402
from paths import ROOT  # noqa: E402
from edge_ai_extended_analysis import (  # noqa: E402
    CuratedDataset, make_eval_transform, get_predictions, point_metrics,
    bootstrap_ci, build_candidate, build_resnet18_baseline, VAL_MANIFEST,
)
import train_transfer_cnn as tcnn  # noqa: E402

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)

IMG_SIZE = 224
# ShuffleNetV2-0.5x/1.0x excluded: torch-pruning raises ZeroDivisionError on their
# channel-shuffle/grouped-conv structure (verified, not attempted to work around here).
STRUCT_MODELS = [
    "resnet18", "mobilenetv2_100", "mobilenetv3_small", "squeezenet1_1",
    "efficientnet_lite0", "efficientnet_b0", "repghostnet_050",
]
EXCLUDED_MODELS = {
    "shufflenet_v2_x0_5": "torch-pruning ZeroDivisionError on channel-shuffle/grouped-conv structure",
    "shufflenet_v2_x1_0": "torch-pruning ZeroDivisionError on channel-shuffle/grouped-conv structure (this project's deployed candidate)",
}
PRUNING_RATIOS = [0.2, 0.3, 0.5]
OUT_DIR = ROOT / "papers" / "edge-ai-lightweight-deployment"
OUT_JSON = OUT_DIR / "edge_ai_structured_pruning_2026-09-19.json"


def load_fresh(model_name):
    return build_resnet18_baseline() if model_name == "resnet18" else build_candidate(model_name)


def final_head(model, model_name):
    if model_name == "resnet18":
        return model.fc
    cfg = tcnn.MODEL_CONFIGS[model_name]
    return getattr(model, cfg["final_attr"])


def checkpoint_size_mb(model):
    """Actual serialized state_dict size -- the real-world contrast against unstructured
    pruning, which leaves this number unchanged (already established elsewhere in this
    project's docs)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / (1024 * 1024)


def structured_prune(model, model_name, ratio):
    head = final_head(model, model_name)
    example_inputs = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    imp = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MagnitudePruner(
        model, example_inputs, importance=imp, pruning_ratio=ratio, ignored_layers=[head],
    )
    pruner.step()
    return model


def main():
    if not require_free_mb(400, context="startup"):
        print("Not enough free RAM to start. Aborting.")
        return

    eval_tf = make_eval_transform()
    val_ds = CuratedDataset(VAL_MANIFEST, eval_tf)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    out = {}
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            out = json.load(f)
        print(f"Resuming -- already done: {list(out.get('per_model', {}).keys())}", flush=True)
    out["excluded_models"] = EXCLUDED_MODELS
    out.setdefault("per_model", {})

    for model_name in STRUCT_MODELS:
        if model_name in out["per_model"]:
            print(f"\n=== {model_name}: already done, skipping ===", flush=True)
            continue
        if not require_free_mb(400, context=f"before {model_name}"):
            print(f"Stopping before {model_name} (low RAM). Re-run to resume.")
            break

        print(f"\n=== structured pruning: {model_name} ===", flush=True)
        t0 = time.time()
        is_resnet18 = model_name == "resnet18"

        # FP32 baseline, recomputed here (cheap) so this file is self-contained
        m0 = load_fresh(model_name)
        base_params = sum(p.numel() for p in m0.parameters())
        base_size_mb = checkpoint_size_mb(m0)
        labels, probs = get_predictions(m0, val_loader, is_resnet18)
        fp32_metrics = point_metrics(labels, probs)
        del m0
        gc.collect()
        print(f"  fp32: acc={fp32_metrics['accuracy']:.4f} params={base_params} "
              f"size={base_size_mb:.2f}MB", flush=True)

        model_out = {
            "fp32": {"point": fp32_metrics, "params": base_params, "checkpoint_size_mb": base_size_mb},
            "structured_pruning": {},
        }

        for ratio in PRUNING_RATIOS:
            try:
                m = load_fresh(model_name)
                m = structured_prune(m, model_name, ratio)
                new_params = sum(p.numel() for p in m.parameters())
                new_size_mb = checkpoint_size_mb(m)
                labels, probs = get_predictions(m, val_loader, is_resnet18)
                metrics = point_metrics(labels, probs)
                ci = bootstrap_ci(labels, probs)
                model_out["structured_pruning"][str(ratio)] = {
                    "point": metrics,
                    "bootstrap_ci": ci,
                    "params": new_params,
                    "checkpoint_size_mb": new_size_mb,
                    "param_reduction_fraction": 1 - new_params / base_params,
                    "size_reduction_fraction": 1 - new_size_mb / base_size_mb,
                }
                print(f"  ratio={ratio}: acc={metrics['accuracy']:.4f} "
                      f"params {base_params}->{new_params} "
                      f"({1 - new_params/base_params:.1%} smaller) "
                      f"size {base_size_mb:.2f}->{new_size_mb:.2f}MB", flush=True)
                del m
                gc.collect()
            except Exception as e:
                print(f"  ratio={ratio}: FAILED -- {type(e).__name__}: {e}", flush=True)
                model_out["structured_pruning"][str(ratio)] = {"error": f"{type(e).__name__}: {e}"}

        model_out["total_time_s"] = time.time() - t0
        out["per_model"][model_name] = model_out
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved progress ({len(out['per_model'])}/{len(STRUCT_MODELS)}).", flush=True)

    print(f"\nAll done. Results at {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
