# -*- coding: utf-8 -*-
"""Root-cause check for the qnnpack INT8 "collapse" (2026-09-23).

Finding (perchannel_anomaly_diagnostic_2026-09-23.py + micro-tests): on this machine's PyTorch 2.8.0
qnnpack build, quantized conv outputs are channels_last (NHWC), and the quantized relu6 / hardtanh /
clamp kernels return WRONG values for channels_last quantized input (max error 6.0 on a 0-6 range,
exact on NCHW input). Plain quantized ReLU is unaffected (it is fused into the conv). Architectures that
apply ReLU6 (or timm's hard_sigmoid = relu6(x + 3) / 6) after a quantized conv therefore get scrambled
activations.

This script re-runs the ORIGINAL qnnpack evaluation (edge_ai_qnnpack_backend_check.py: default qnnpack
qconfig, 8 calibration batches from the training split) and evaluates each calibrated model twice:
  as_is  -- identical to the original pipeline (should reproduce the 2026-09-19 numbers)
  fixed  -- same calibrated model, with x.contiguous() inserted before every relu6/hardtanh/clamp node
Validation split only; the frozen test split is NOT re-opened.
"""
import copy
import json
import operator  # noqa: F401
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
import edge_ai_extended_analysis as ext  # noqa: E402

torch.backends.quantized.engine = "qnnpack"
OUT = ROOT / "papers" / "edge-ai-lightweight-deployment" / "qnnpack_relu6_layout_fix_2026-09-23.json"
ORIG = ROOT / "papers" / "edge-ai-lightweight-deployment" / "edge_ai_qnnpack_backend_check_2026-09-19.json"
ALL = ["resnet18", "efficientnet_b0", "efficientnet_lite0", "mobilenetv3_small", "mobilenetv2_100",
       "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "squeezenet1_1", "repghostnet_050"]
CLAMP_FUNCS = {F.relu6, F.hardtanh, torch.clamp, torch.clamp_, F.hardtanh_}
CLAMP_METHODS = {"clamp", "clamp_", "clip", "clip_"}


def is_clamp_node(gm, n):
    if n.op == "call_module":
        return isinstance(gm.get_submodule(n.target), (nn.ReLU6, nn.Hardtanh))
    if n.op == "call_function":
        return n.target in CLAMP_FUNCS
    if n.op == "call_method":
        return n.target in CLAMP_METHODS
    return False


def fix_layout(gm):
    gm = copy.deepcopy(gm)
    count = 0
    for n in list(gm.graph.nodes):
        if is_clamp_node(gm, n):
            with gm.graph.inserting_before(n):
                c = gm.graph.call_method("contiguous", (n.args[0],))
            n.replace_input_with(n.args[0], c)
            count += 1
    gm.graph.lint()
    gm.recompile()
    return gm, count


def main():
    import os
    os.chdir(ROOT)
    orig = json.load(open(ORIG))
    tf = ext.make_eval_transform()
    val_loader = DataLoader(ext.CuratedDataset(ext.VAL_MANIFEST, tf), batch_size=32, shuffle=False)
    train_loader = DataLoader(ext.CuratedDataset(ext.TRAIN_MANIFEST, tf), batch_size=32, shuffle=True)
    out = {}
    for name in ALL:
        torch.manual_seed(42)
        np.random.seed(42)
        is_r = name == "resnet18"
        q = ext.quantize_static(ext.load_model(name), train_loader)
        qf, n_fixed = fix_layout(q)
        res = {"n_clamp_nodes_fixed": n_fixed}
        for tag, model in (("as_is", q), ("fixed", qf)):
            labels, probs = ext.get_predictions(model, val_loader, is_r)
            res[tag] = {k: float(v) for k, v in ext.point_metrics(labels, probs).items()}
            res[tag + "_labels_probs"] = {"labels": [int(v) for v in labels], "probs": [float(v) for v in probs]}
        res["orig_2026_09_19_accuracy"] = orig[name]["qnnpack_backend_this_mac"]["point"]["accuracy"]
        out[name] = res
        a, f_ = res["as_is"], res["fixed"]
        print(f"{name:20s} clamp-nodes={n_fixed:3d} | as_is acc={a['accuracy']:.3f} se={a['sensitivity_recall']:.3f} "
              f"sp={a['specificity']:.3f} (orig {res['orig_2026_09_19_accuracy']:.3f}) | fixed acc={f_['accuracy']:.3f} "
              f"se={f_['sensitivity_recall']:.3f} sp={f_['specificity']:.3f} auroc={f_['auroc']:.3f}", flush=True)
        json.dump(out, open(OUT, "w"))
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
