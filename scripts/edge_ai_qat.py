# -*- coding: utf-8 -*-
"""Quantization-aware training (QAT) for the 3 SE/swish-containing models identified as
fragile under static post-training quantization (PTQ) in the original edge-AI sweep
(EfficientNet-B0, MobileNetV3-Small, RepGhostNet-0.5x) -- addresses the paper-revision
request to compare static PTQ against QAT specifically for these three.

Starts from each model's already fine-tuned FP32 checkpoint (the same one used
everywhere else in this project), inserts fake-quantization ops (FX graph mode,
prepare_qat_fx), fine-tunes for a few epochs on the TRAIN split at a low learning rate
(this is fine-tuning an already-converged model to become quantization-robust, not
training from scratch -- standard QAT practice), then converts to a real INT8 model and
evaluates on the VALIDATION split. Test manifest never opened.

CPU-only, this project's 8GB dev machine -- QAT is real gradient-based training, several
times more expensive per step than plain PTQ calibration, so this is deliberately bounded
(3 epochs, small LR) rather than a full from-scratch QAT run. Results are reported next to
the corresponding static-PTQ numbers already on record, not as a replacement for them.
"""
import copy
import csv
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mem_guard import require_free_mb  # noqa: E402
from paths import ROOT, SKINDISEASE_DIR  # noqa: E402
import train_transfer_cnn as tcnn  # noqa: E402
from edge_ai_extended_analysis import (  # noqa: E402
    CuratedDataset, make_eval_transform, get_predictions, point_metrics,
    bootstrap_ci, build_candidate, VAL_MANIFEST, TRAIN_MANIFEST, QUANT_BACKEND,
)

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)

IMG_SIZE = 224
QAT_MODELS = ["efficientnet_b0", "mobilenetv3_small", "repghostnet_050"]
QAT_EPOCHS = 3
QAT_LR = 1e-5
OUT_DIR = ROOT / "papers" / "edge-ai-lightweight-deployment"
OUT_JSON = OUT_DIR / "edge_ai_qat_2026-09-19.json"


def run_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, n = 0.0, 0
    for imgs, labels in loader:
        labels_f = labels.float().unsqueeze(1)
        logits = model(imgs)
        loss = criterion(logits, labels_f)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n


def main():
    if not require_free_mb(500, context="startup"):
        print("Not enough free RAM to start QAT. Aborting.")
        return

    eval_tf = make_eval_transform()
    val_ds = CuratedDataset(VAL_MANIFEST, eval_tf)
    train_ds = CuratedDataset(TRAIN_MANIFEST, eval_tf)  # deterministic tf; QAT fine-tune
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    train_loader_eval = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)
    train_loader_train = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)

    out = {}
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            out = json.load(f)
        print(f"Resuming -- already done: {list(out.keys())}", flush=True)

    for model_name in QAT_MODELS:
        if model_name in out:
            print(f"\n=== {model_name}: already done, skipping ===", flush=True)
            continue
        if not require_free_mb(500, context=f"before {model_name}"):
            print(f"Stopping before {model_name} (low RAM). Re-run to resume.")
            break

        print(f"\n=== QAT: {model_name} ===", flush=True)
        t0 = time.time()

        fp32_model = build_candidate(model_name)
        qconfig_mapping = get_default_qat_qconfig_mapping(QUANT_BACKEND)
        example_inputs = (torch.randn(1, 3, IMG_SIZE, IMG_SIZE),)
        fp32_model.train()
        prepared = prepare_qat_fx(fp32_model, qconfig_mapping, example_inputs)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(prepared.parameters(), lr=QAT_LR, weight_decay=1e-4)

        history = []
        for epoch in range(1, QAT_EPOCHS + 1):
            if not require_free_mb(400, context=f"{model_name} QAT epoch {epoch}"):
                print(f"  Stopping {model_name} QAT early at epoch {epoch} (low RAM).")
                break
            t_ep = time.time()
            train_loss = run_epoch(prepared, train_loader_train, criterion, optimizer)
            dt = time.time() - t_ep
            history.append({"epoch": epoch, "train_loss": train_loss, "time_s": dt})
            print(f"  epoch {epoch}/{QAT_EPOCHS} loss={train_loss:.4f} ({dt:.1f}s)", flush=True)

        prepared.eval()
        quantized = convert_fx(prepared)
        labels, probs = get_predictions(quantized, val_loader, is_resnet18=False)
        qat_metrics = point_metrics(labels, probs)
        qat_ci = bootstrap_ci(labels, probs)

        total_time = time.time() - t0
        out[model_name] = {
            "epochs_completed": len(history),
            "history": history,
            "total_time_s": total_time,
            "val_metrics": qat_metrics,
            "bootstrap_ci": qat_ci,
        }
        print(f"  QAT result: acc={qat_metrics['accuracy']:.4f} f1={qat_metrics['f1']:.4f} "
              f"auroc={qat_metrics['auroc']:.4f} (total {total_time:.0f}s)", flush=True)

        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=2)

        del fp32_model, prepared, quantized
        gc.collect()

    print(f"\nAll done. Results at {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
