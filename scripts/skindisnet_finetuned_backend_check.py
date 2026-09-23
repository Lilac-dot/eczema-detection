# -*- coding: utf-8 -*-
"""SkinDisNet backend/collapse check AFTER fine-tuning on SkinDisNet's own data (2026-09-21) --
extends the zero-shot check in skindisnet_all9_external_validation.py.

Motivation: in the zero-shot check, per-architecture FP32 F1 on SkinDisNet is already very
low (0.013-0.100) before any quantization is applied, because none of the 9 architectures
had ever seen SkinDisNet data. That means the zero-shot check cannot cleanly separate "the
qnnpack backend/quantization hurts this model" from "this model never generalized to
SkinDisNet's distribution in the first place" -- for at least one architecture
(ShuffleNetV2-0.5x: FP32 F1 0.036 vs INT8 F1 0.032, barely different), the zero-shot
"collapse" looks like it's almost entirely the latter.

This script fine-tunes each of the 9 architectures on SkinDisNet's own training data first
(scripts/build_skindisnet_finetune_split.py's patient-level, deduplicated split), giving
each architecture a real FP32 baseline ON SkinDisNet, THEN re-runs the same FP32-vs-qnnpack-
INT8 comparison on top of that fine-tuned model. This isolates the backend/quantization
effect from the pre-existing distribution-shift confound, the same way the paper's primary
finding does for its own dataset.

Fine-tuning recipe reused EXACTLY from edge_ai_pruning_finetune.py (this project's existing
fine-tuning recipe for these same 9 architectures, used for the paper's Table 7): whole-model
AdamW, lr=5e-5, weight_decay=1e-4, 3 epochs, batch_size=32, CrossEntropyLoss for resnet18
(2-class softmax head) / BCEWithLogitsLoss for the 8 candidates (1-logit sigmoid head). Kept
identical on purpose so results differ only in the training data, not the recipe.

Evaluation split: manifest_skindisnet_finetune_val.csv (298 images / 80 patients), matching
this project's own discipline of using the validation split for every exploratory result and
never opening a test split until a single final, frozen evaluation (paper Section 4.10, 9.1).
INT8 calibration uses manifest_skindisnet_finetune_train.csv only (the same split used for
fine-tuning), matching Section 4.2's rule that calibration data should match the training
distribution being deployed.

This machine's PyTorch build supports only the qnnpack quantized backend
(torch.backends.quantized.supported_engines == ['qnnpack', 'none']) -- no x86/fbgemm number
is produced or claimed here.

Saves each fine-tuned checkpoint to models/<model_name>_skindisnet_finetuned.pt (new files,
existing checkpoints under models/ and experiments/ are never touched -- fine-tuning always
starts from a FRESH copy of the original eczema-trained weights via load_fresh()).
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mem_guard import require_free_mb  # noqa: E402
from paths import ROOT, SKINDISNET_DIR, MODELS_DIR  # noqa: E402
from edge_ai_extended_analysis import (  # noqa: E402
    CuratedDataset, make_eval_transform, get_predictions, point_metrics, bootstrap_ci,
    build_candidate, build_resnet18_baseline, quantize_static, quantized_op_coverage,
    ALL_MODELS, QUANT_BACKEND,
)

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)

# This machine has a usable MPS GPU (torch.backends.mps.is_available()); the rest of this
# project's scripts only check torch.cuda.is_available() and never exercise MPS. Used HERE
# only for the FP32 fine-tuning loop -- torch's quantized (INT8) ops have no MPS kernels at
# all, so quantize_static() and every INT8 evaluation stay on CPU with the qnnpack backend,
# unchanged from the original plan. FP32 evaluation (get_predictions) also stays on CPU for
# simplicity/consistency with the rest of this project's CPU-only evaluation code -- only
# training-loop tensors move to MPS. If a given architecture throws an MPS-unsupported-op
# error during fine-tuning, that architecture falls back to CPU for its fine-tune only; which
# device actually trained each architecture is recorded per-model in the output JSON.
FT_DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

FT_EPOCHS = 3
FT_LR = 5e-5
FT_WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32

FINETUNE_TRAIN = SKINDISNET_DIR / "manifest_skindisnet_finetune_train.csv"
FINETUNE_VAL = SKINDISNET_DIR / "manifest_skindisnet_finetune_val.csv"
OUT_DIR = ROOT / "papers" / "edge-ai-lightweight-deployment"
OUT_JSON = OUT_DIR / "skindisnet_finetuned_backend_check_2026-09-21.json"

assert FINETUNE_TRAIN.exists(), "Run build_skindisnet_finetune_split.py first."
assert FINETUNE_VAL.exists(), "Run build_skindisnet_finetune_split.py first."


def load_fresh(model_name):
    return build_resnet18_baseline() if model_name == "resnet18" else build_candidate(model_name)


def run_epoch(model, loader, criterion, optimizer, is_resnet18, device):
    model.train()
    total_loss, n = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        if is_resnet18:
            logits = model(imgs)
            loss = criterion(logits, labels)
        else:
            logits = model(imgs)
            loss = criterion(logits, labels.float().unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n


def classify_status(f1, sensitivity, specificity):
    """Paper Section 4.8's operational thresholds, reused verbatim."""
    if f1 < 0.05:
        return "collapsed"
    if sensitivity < 0.20 or specificity < 0.20:
        return "severely degraded"
    return "backend-stable"


def main():
    if not require_free_mb(500, context="startup"):
        print("Not enough free RAM to start. Aborting.")
        return

    eval_tf = make_eval_transform()
    ft_train_ds = CuratedDataset(FINETUNE_TRAIN, eval_tf)
    ft_val_ds = CuratedDataset(FINETUNE_VAL, eval_tf)
    # shuffle=True loader for fine-tuning; a separate shuffle=False loader (same
    # underlying dataset/images) is used for INT8 calibration, matching how
    # edge_ai_extended_analysis.py's own train_loader is used for calibration elsewhere.
    ft_train_loader_shuffled = DataLoader(ft_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    calib_loader = DataLoader(ft_train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader = DataLoader(ft_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    out = {}
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            out = json.load(f)
        print(f"Resuming -- already done: {list(out.get('per_model', {}).keys())}", flush=True)

    out.setdefault("metadata", {
        "quant_backend": QUANT_BACKEND,
        "finetune_recipe": {
            "epochs": FT_EPOCHS, "lr": FT_LR, "weight_decay": FT_WEIGHT_DECAY,
            "optimizer": "AdamW", "batch_size": BATCH_SIZE,
            "source": "edge_ai_pruning_finetune.py FT_EPOCHS/FT_LR, reused verbatim",
        },
        "finetune_train_manifest": str(FINETUNE_TRAIN.relative_to(ROOT)),
        "finetune_train_n": len(ft_train_ds),
        "eval_manifest": str(FINETUNE_VAL.relative_to(ROOT)),
        "eval_n": len(ft_val_ds),
        "calibration_manifest": str(FINETUNE_TRAIN.relative_to(ROOT)),
        "calibration_n_batches": 8,
        "status_thresholds": "collapsed: F1<0.05; severely degraded: Sens or Spec <20%; else backend-stable (paper Section 4.8)",
        "note": "x86/fbgemm not attempted -- this machine only supports qnnpack.",
    })
    out.setdefault("per_model", {})

    for model_name in ALL_MODELS:
        if model_name in out["per_model"]:
            print(f"\n=== {model_name}: already done, skipping ===", flush=True)
            continue
        if not require_free_mb(500, context=f"before {model_name}"):
            print(f"Stopping before {model_name} (low RAM). Re-run to resume.")
            break

        print(f"\n=== {model_name} ===", flush=True)
        t0 = time.time()
        is_resnet18 = model_name == "resnet18"
        criterion = nn.CrossEntropyLoss() if is_resnet18 else nn.BCEWithLogitsLoss()

        m = load_fresh(model_name)

        # pre-fine-tune (zero-shot) FP32 accuracy on this SAME val split, for reference --
        # this is what the original zero-shot check is analogous to, just on this split.
        labels, probs = get_predictions(m, val_loader, is_resnet18)
        zero_shot_fp32 = point_metrics(labels, probs)
        print(f"  zero-shot FP32 (pre-fine-tune) on finetune_val: acc={zero_shot_fp32['accuracy']:.4f} "
              f"f1={zero_shot_fp32['f1']:.4f}", flush=True)

        train_device = FT_DEVICE
        try:
            m = m.to(train_device)
        except Exception as e:
            print(f"  Could not move {model_name} to {train_device} ({e}); using cpu.", flush=True)
            train_device = torch.device("cpu")
            m = m.to(train_device)

        optimizer = torch.optim.AdamW(m.parameters(), lr=FT_LR, weight_decay=FT_WEIGHT_DECAY)
        history = []
        mps_fallback = False
        for epoch in range(1, FT_EPOCHS + 1):
            if not require_free_mb(400, context=f"{model_name} epoch {epoch}"):
                print(f"  Stopping fine-tune early at epoch {epoch} (low RAM).")
                break
            t_ep = time.time()
            try:
                train_loss = run_epoch(m, ft_train_loader_shuffled, criterion, optimizer,
                                        is_resnet18, train_device)
            except Exception as e:
                if train_device.type == "mps":
                    print(f"  MPS error on {model_name} epoch {epoch} ({e}); "
                          f"falling back to CPU for the rest of this model's fine-tune.", flush=True)
                    train_device = torch.device("cpu")
                    m = m.to(train_device)
                    optimizer = torch.optim.AdamW(m.parameters(), lr=FT_LR, weight_decay=FT_WEIGHT_DECAY)
                    mps_fallback = True
                    t_ep = time.time()
                    train_loss = run_epoch(m, ft_train_loader_shuffled, criterion, optimizer,
                                            is_resnet18, train_device)
                else:
                    raise
            dt = time.time() - t_ep
            history.append({"epoch": epoch, "train_loss": train_loss, "time_s": dt,
                             "device": train_device.type})
            print(f"    epoch {epoch}/{FT_EPOCHS} loss={train_loss:.4f} ({dt:.1f}s) "
                  f"device={train_device.type}", flush=True)

        m = m.to("cpu")

        ckpt_path = MODELS_DIR / f"{model_name}_skindisnet_finetuned.pt"
        torch.save(m.state_dict(), ckpt_path)

        labels, probs = get_predictions(m, val_loader, is_resnet18)
        fp32_metrics = point_metrics(labels, probs)
        fp32_ci = bootstrap_ci(labels, probs)
        fp32_status = classify_status(fp32_metrics["f1"], fp32_metrics["sensitivity_recall"],
                                       fp32_metrics["specificity"])
        print(f"  fine-tuned FP32: acc={fp32_metrics['accuracy']:.4f} f1={fp32_metrics['f1']:.4f} "
              f"sens={fp32_metrics['sensitivity_recall']:.4f} spec={fp32_metrics['specificity']:.4f} "
              f"status={fp32_status}", flush=True)

        m_q = quantize_static(m, calib_loader, n_calib_batches=8)
        labels, probs = get_predictions(m_q, val_loader, is_resnet18)
        int8_metrics = point_metrics(labels, probs)
        int8_ci = bootstrap_ci(labels, probs)
        int8_status = classify_status(int8_metrics["f1"], int8_metrics["sensitivity_recall"],
                                       int8_metrics["specificity"])
        op_cov = quantized_op_coverage(m_q)
        print(f"  fine-tuned qnnpack INT8: acc={int8_metrics['accuracy']:.4f} f1={int8_metrics['f1']:.4f} "
              f"sens={int8_metrics['sensitivity_recall']:.4f} spec={int8_metrics['specificity']:.4f} "
              f"status={int8_status}", flush=True)

        total_time = time.time() - t0
        out["per_model"][model_name] = {
            "zero_shot_fp32_on_finetune_val": zero_shot_fp32,
            "finetune_device": train_device.type,
            "mps_fallback_to_cpu_mid_run": mps_fallback,
            "finetune_history": history,
            "checkpoint": str(ckpt_path.relative_to(ROOT)),
            "finetuned_fp32": {"point": fp32_metrics, "bootstrap_ci": fp32_ci, "status": fp32_status},
            "finetuned_qnnpack_int8": {
                "point": int8_metrics, "bootstrap_ci": int8_ci, "status": int8_status,
                "op_coverage": op_cov,
            },
            "accuracy_delta_fp32_minus_int8": fp32_metrics["accuracy"] - int8_metrics["accuracy"],
            "f1_delta_fp32_minus_int8": fp32_metrics["f1"] - int8_metrics["f1"],
            "total_time_s": total_time,
        }
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved progress ({len(out['per_model'])}/{len(ALL_MODELS)}). "
              f"total_time={total_time:.0f}s", flush=True)

        del m, m_q, optimizer
        gc.collect()

    print(f"\nAll done. Results at {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
