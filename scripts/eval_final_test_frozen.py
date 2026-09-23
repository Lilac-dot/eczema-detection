# -*- coding: utf-8 -*-
"""Final held-out TEST evaluation, 2026-09-20. Implements EXACTLY the protocol committed
in papers/edge-ai-lightweight-deployment/FROZEN_TEST_PROTOCOL_2026-09-20.md, written and
saved to disk BEFORE this script was written or run. Do not add conditions, models, or
metrics beyond what that document specifies -- if something needs to change, the protocol
document must be edited first, with the change visible in its own history, not silently
expanded here.

manifest_curated_v3_test.csv (507 images) has never been opened for any experiment in
this project's compression/deployment work before this script. This is that first (and
only) use.

Two conditions per architecture, all 9 architectures, no exceptions and no
post-hoc selection:
  1. FP32 (existing checkpoint, no retraining)
  2. ARM/qnnpack static INT8 (calibrated on the TRAINING split only, never on
     validation, test, or SkinDisNet data)

x86/fbgemm INT8 on test is NOT attempted -- this machine's PyTorch build only supports
the qnnpack quantized engine, so there is no way to execute an x86/fbgemm-quantized model
here. This is reported as an infeasible experiment in the output JSON, not skipped
silently.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT, SKINDISEASE_DIR  # noqa: E402
from edge_ai_extended_analysis import (  # noqa: E402
    CuratedDataset, make_eval_transform, get_predictions, point_metrics,
    bootstrap_ci, load_model, quantize_static, TRAIN_MANIFEST, ALL_MODELS, QUANT_BACKEND,
)

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)

TEST_MANIFEST = SKINDISEASE_DIR / "manifest_curated_v3_test.csv"
assert TEST_MANIFEST.name == "manifest_curated_v3_test.csv"
OUT_JSON = ROOT / "papers" / "edge-ai-lightweight-deployment" / "final_test_frozen_results_2026-09-20.json"


def confusion(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def main():
    print(f"Opening TEST manifest for the first time in this compression study: {TEST_MANIFEST}", flush=True)
    eval_tf = make_eval_transform()
    test_ds = CuratedDataset(TEST_MANIFEST, eval_tf)
    train_ds = CuratedDataset(TRAIN_MANIFEST, eval_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    print(f"n_test = {len(test_ds)}", flush=True)
    print(f"Quantized backend in use: {QUANT_BACKEND}", flush=True)

    out = {"protocol_file": "FROZEN_TEST_PROTOCOL_2026-09-20.md", "n_test": len(test_ds),
           "x86_fbgemm_int8_on_test": "INFEASIBLE -- this machine only supports the qnnpack quantized engine",
           "per_model": {}}
    if OUT_JSON.exists():
        prev = json.load(open(OUT_JSON))
        out["per_model"] = prev.get("per_model", {})
        print(f"Resuming -- already done: {list(out['per_model'].keys())}", flush=True)

    for model_name in ALL_MODELS:
        if model_name in out["per_model"]:
            print(f"\n=== {model_name}: already done, skipping ===", flush=True)
            continue
        print(f"\n=== {model_name} ===", flush=True)
        is_resnet18 = model_name == "resnet18"

        m = load_model(model_name)
        labels, probs = get_predictions(m, test_loader, is_resnet18)
        fp32_metrics = point_metrics(labels, probs)
        fp32_ci = bootstrap_ci(labels, probs)
        fp32_cm = confusion(labels, probs)
        print(f"  FP32: acc={fp32_metrics['accuracy']:.4f} sens={fp32_metrics['sensitivity_recall']:.4f} "
              f"spec={fp32_metrics['specificity']:.4f} f1={fp32_metrics['f1']:.4f} "
              f"auroc={fp32_metrics['auroc']:.4f}", flush=True)

        m_q = quantize_static(load_model(model_name), train_loader)
        labels_q, probs_q = get_predictions(m_q, test_loader, is_resnet18)
        int8_metrics = point_metrics(labels_q, probs_q)
        int8_ci = bootstrap_ci(labels_q, probs_q)
        int8_cm = confusion(labels_q, probs_q)
        print(f"  qnnpack INT8: acc={int8_metrics['accuracy']:.4f} sens={int8_metrics['sensitivity_recall']:.4f} "
              f"spec={int8_metrics['specificity']:.4f} f1={int8_metrics['f1']:.4f} "
              f"auroc={int8_metrics['auroc']:.4f}", flush=True)

        out["per_model"][model_name] = {
            "fp32": {"point": fp32_metrics, "bootstrap_ci": fp32_ci, "confusion_matrix": fp32_cm},
            "qnnpack_int8": {"point": int8_metrics, "bootstrap_ci": int8_ci, "confusion_matrix": int8_cm},
        }
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=2)

    print(f"\nAll done. Results at {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
