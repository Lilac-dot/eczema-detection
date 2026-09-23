# -*- coding: utf-8 -*-
"""Methodological diagnostic, 2026-09-20 (requested during publication-readiness review):
the fbgemm-vs-qnnpack comparison in this project (edge_ai_qnnpack_backend_check.py) uses
each backend's own DEFAULT qconfig, which differ in more than kernel numerics --
verified directly on this machine's installed PyTorch:

  x86/fbgemm default: weight = PerChannelMinMaxObserver (per-CHANNEL symmetric qint8),
                       activation = HistogramObserver(reduce_range=True)
  qnnpack default:     weight = MinMaxObserver (per-TENSOR symmetric qint8),
                       activation = HistogramObserver(reduce_range=False)

So the original comparison conflates two variables: the backend's execution kernels, and
each backend's differing default quantization granularity (per-channel vs per-tensor
weights). This script isolates the granularity variable: it re-quantizes each affected
architecture on the qnnpack ENGINE (so it still executes with qnnpack's kernels) but with
weight quantization FORCED to per-channel (confirmed to convert and run without error on
qnnpack in this PyTorch build, i.e. not a hard kernel limitation), holding activation
quantization at qnnpack's own default (HistogramObserver, reduce_range=False). If this
rescues an architecture, the granularity default is implicated as the dominant cause; if
collapse persists, something else backend-specific (e.g. the FP32-vs-limited-float
requantization path difference documented in pytorch/pytorch#44939) is implicated instead.

Scope: the 3 architectures that collapse under qnnpack's default config
(EfficientNet-Lite0, MobileNetV2-1.0x, RepGhostNet-0.5x) plus the 2 that are severely
degraded but not fully collapsed (EfficientNet-B0, MobileNetV3-Small) -- not all 9, since
the other 4 already behave the same on both backends and this diagnostic only matters for
the ones that differ. Validation set only, calibration on the training split only (same
discipline as every other quantization result in this project).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.observer import PerChannelMinMaxObserver, HistogramObserver
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
from edge_ai_extended_analysis import (  # noqa: E402
    CuratedDataset, make_eval_transform, get_predictions, point_metrics,
    load_model, VAL_MANIFEST, TRAIN_MANIFEST,
)

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)
torch.backends.quantized.engine = "qnnpack"

IMG_SIZE = 224
TARGET_MODELS = ["efficientnet_lite0", "mobilenetv2_100", "repghostnet_050",
                  "efficientnet_b0", "mobilenetv3_small"]
OUT_JSON = ROOT / "papers" / "edge-ai-lightweight-deployment" / "qnnpack_perchannel_diagnostic_2026-09-20.json"


def quantize_qnnpack_perchannel(model, train_loader, n_calib_batches=8):
    """qnnpack engine, but weight observer forced to per-channel (qnnpack's own default
    is per-tensor) -- isolates the granularity variable from the rest of qnnpack's
    quantization pipeline."""
    qconfig = QConfig(
        activation=HistogramObserver.with_args(reduce_range=False),
        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
    )
    qm = QConfigMapping().set_global(qconfig)
    example_inputs = (torch.randn(1, 3, IMG_SIZE, IMG_SIZE),)
    prepared = prepare_fx(model, qm, example_inputs)
    with torch.no_grad():
        for i, (imgs, _) in enumerate(train_loader):
            if i >= n_calib_batches:
                break
            prepared(imgs)
    return convert_fx(prepared)


def main():
    eval_tf = make_eval_transform()
    val_ds = CuratedDataset(VAL_MANIFEST, eval_tf)
    train_ds = CuratedDataset(TRAIN_MANIFEST, eval_tf)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)

    out = {}
    if OUT_JSON.exists():
        out = json.load(open(OUT_JSON))

    for model_name in TARGET_MODELS:
        if model_name in out:
            print(f"{model_name}: already done, skipping")
            continue
        print(f"\n=== {model_name}: qnnpack engine, weights forced per-channel ===", flush=True)
        try:
            m = load_model(model_name)
            m_q = quantize_qnnpack_perchannel(m, train_loader)
            labels, probs = get_predictions(m_q, val_loader, model_name == "resnet18")
            metrics = point_metrics(labels, probs)
            out[model_name] = {"status": "ok", "point": metrics}
            print(f"  acc={metrics['accuracy']:.4f} sens={metrics['sensitivity_recall']:.4f} "
                  f"spec={metrics['specificity']:.4f} f1={metrics['f1']:.4f}", flush=True)
        except Exception as e:
            out[model_name] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=2)

    print(f"\nAll done. Results at {OUT_JSON}")


if __name__ == "__main__":
    main()
