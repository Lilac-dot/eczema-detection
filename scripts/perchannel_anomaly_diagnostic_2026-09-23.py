# -*- coding: utf-8 -*-
"""Why doesn't forced per-channel weight quantization rescue MobileNetV2 (and EfficientNet-Lite0,
RepGhostNet) on qnnpack, when the literature (Krishnamoorthi 2018; Nagel et al. 2019) shows per-channel
restores MobileNetV2 to near-FP32? (qnnpack_perchannel_diagnostic_2026-09-20.json: F1 = 0, AUROC ~0.53.)

Controlled comparisons, one architecture at a time, validation split, calibration on training split:
  A. weight-only fake quantization after BN folding (activations stay FP32): per-tensor vs per-channel
  B. full static quantization with three qconfigs, each converted TWO ways from the same calibrated model:
       - convert_to_reference_fx: simulated quantization in float math (backend-kernel independent)
       - convert_fx on the qnnpack engine: real qnnpack int8 kernels
     qconfigs: qnnpack default (per-tensor, reduce_range=False); per-channel + reduce_range=False
     (= the 09-20 diagnostic); fbgemm default (per-channel, reduce_range=True).
Diagnosis logic: A isolates weight rounding; reference-vs-qnnpack isolates the kernels.
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.ao.quantization import QConfigMapping, get_default_qconfig_mapping
from torch.ao.quantization.observer import HistogramObserver, PerChannelMinMaxObserver, MinMaxObserver
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, convert_to_reference_fx, fuse_fx
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
from edge_ai_extended_analysis import (  # noqa: E402
    CuratedDataset, make_eval_transform, get_predictions, point_metrics, load_model,
    VAL_MANIFEST, TRAIN_MANIFEST,
)

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(8)
torch.backends.quantized.engine = "qnnpack"
MODELS = sys.argv[1:] or ["mobilenetv2_100"]
OUT = ROOT / "papers" / "edge-ai-lightweight-deployment" / "perchannel_anomaly_diagnostic_2026-09-23.json"
EX = (torch.randn(1, 3, 224, 224),)


def summarize(labels, probs):
    m = point_metrics(labels, probs)
    return {k: round(float(m[k]), 4) for k in ("accuracy", "sensitivity_recall", "specificity", "f1", "auroc")}


def fake_quant_weights(model, per_channel):
    """Symmetric int8 fake-quantization of every Conv/Linear weight (after BN folding)."""
    m = fuse_fx(copy.deepcopy(model).eval())
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                w = mod.weight
                if per_channel:
                    amax = w.detach().abs().flatten(1).max(dim=1).values.clamp_min(1e-12)
                    scale = (amax / 127.0).view(-1, *([1] * (w.dim() - 1)))
                else:
                    scale = w.detach().abs().max().clamp_min(1e-12) / 127.0
                w.copy_(torch.clamp(torch.round(w / scale), -127, 127) * scale)
    return m


QCONFIGS = {
    "qnnpack_default_per_tensor": None,  # get_default_qconfig_mapping("qnnpack")
    "per_channel_rr_false_(09-20 diag)": QConfig(
        activation=HistogramObserver.with_args(reduce_range=False),
        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)),
    "fbgemm_default_per_channel_rr_true": QConfig(
        activation=HistogramObserver.with_args(reduce_range=True),
        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)),
}


def main():
    tf = make_eval_transform()
    val_loader = DataLoader(CuratedDataset(VAL_MANIFEST, tf), batch_size=32, shuffle=False)
    train_ds = CuratedDataset(TRAIN_MANIFEST, tf)
    calib_idx = np.random.RandomState(42).choice(len(train_ds), 256, replace=False)
    calib_loader = DataLoader(Subset(train_ds, calib_idx), batch_size=32, shuffle=False)
    out = json.load(open(OUT)) if OUT.exists() else {}
    for name in MODELS:
        res = {}
        model = load_model(name).eval()
        labels, p = get_predictions(model, val_loader, False)
        res["fp32"] = summarize(labels, p)
        print(f"\n=== {name} ===\n fp32: {res['fp32']}", flush=True)
        for pc in (False, True):
            key = f"weight_only_{'per_channel' if pc else 'per_tensor'}"
            labels, p = get_predictions(fake_quant_weights(model, pc), val_loader, False)
            res[key] = summarize(labels, p)
            print(f" {key}: {res[key]}", flush=True)
        for qname, qc in QCONFIGS.items():
            qm = get_default_qconfig_mapping("qnnpack") if qc is None else get_default_qconfig_mapping("qnnpack").set_global(qc)
            prepared = prepare_fx(copy.deepcopy(model), qm, EX)
            with torch.no_grad():
                for imgs, _ in calib_loader:
                    prepared(imgs)
            for conv_name, conv in (("reference_sim", convert_to_reference_fx), ("qnnpack_kernels", convert_fx)):
                t0 = time.time()
                qmodel = conv(copy.deepcopy(prepared))
                labels, p = get_predictions(qmodel, val_loader, False)
                res[f"{qname}|{conv_name}"] = summarize(labels, p)
                print(f" {qname} | {conv_name}: {res[f'{qname}|{conv_name}']} ({time.time() - t0:.0f}s)", flush=True)
        out[name] = res
        json.dump(out, open(OUT, "w"), indent=1)
    print("Saved:", OUT)


if __name__ == "__main__":
    import os
    os.chdir(ROOT)
    main()
