# -*- coding: utf-8 -*-
"""Unstructured pruning + fine-tuning (task 2a), for 3 representative architectures:
ResNet18 (most pruning-robust per the original sweep), ShuffleNetV2-1.0x (the deployed
candidate), MobileNetV3-Small (an SE/swish model, pruning-fragile region). Answers a
question the original one-shot pruning sweep (edge_ai_extended_analysis.py's
"pruned_at_knee", no fine-tuning) could not: how much of the accuracy collapse from
pruning is recoverable by fine-tuning the surviving weights, versus permanently lost?

Method: prune to a target sparsity via the SAME L1Unstructured global pruning as
elsewhere in this project, but do NOT call prune.remove() until after fine-tuning --
while the reparametrization is active, PyTorch computes weight = weight_orig * mask on
every forward pass, so pruned positions are exactly zero in every forward pass regardless
of what happens to weight_orig underneath (the chain rule makes their gradient exactly
zero too, so a plain optimizer step doesn't move them; only weight-decay technically
still touches weight_orig at those positions, but since mask stays 0, the EFFECTIVE
weight used in every forward pass never becomes nonzero until prune.remove() bakes in
the final values, which only happens after all fine-tuning is complete). This is what
"fine-tuning with the pruning mask held fixed" means concretely.

TRAIN split only for fine-tuning, VALIDATION split only for evaluation (test manifest
never opened, same discipline as every other script in this project).
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mem_guard import require_free_mb  # noqa: E402
from paths import ROOT  # noqa: E402
from edge_ai_extended_analysis import (  # noqa: E402
    CuratedDataset, make_eval_transform, get_predictions, point_metrics,
    bootstrap_ci, build_candidate, build_resnet18_baseline, prunable_modules,
    VAL_MANIFEST, TRAIN_MANIFEST,
)

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)

IMG_SIZE = 224
PF_MODELS = ["resnet18", "shufflenet_v2_x1_0", "mobilenetv3_small"]
SPARSITY_LEVELS = [0.3, 0.5, 0.7]
FT_EPOCHS = 3
FT_LR = 5e-5
OUT_DIR = ROOT / "papers" / "edge-ai-lightweight-deployment"
OUT_JSON = OUT_DIR / "edge_ai_pruning_finetune_2026-09-19.json"


def load_fresh(model_name):
    return build_resnet18_baseline() if model_name == "resnet18" else build_candidate(model_name)


def run_epoch(model, loader, criterion, optimizer, is_resnet18):
    model.train()
    total_loss, n = 0.0, 0
    for imgs, labels in loader:
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


def main():
    if not require_free_mb(500, context="startup"):
        print("Not enough free RAM to start. Aborting.")
        return

    eval_tf = make_eval_transform()
    val_ds = CuratedDataset(VAL_MANIFEST, eval_tf)
    train_ds = CuratedDataset(TRAIN_MANIFEST, eval_tf)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)

    out = {}
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            out = json.load(f)
        print(f"Resuming -- already done: {list(out.get('per_model', {}).keys())}", flush=True)
    out.setdefault("per_model", {})

    for model_name in PF_MODELS:
        is_resnet18 = model_name == "resnet18"
        criterion = nn.CrossEntropyLoss() if is_resnet18 else nn.BCEWithLogitsLoss()

        if model_name not in out["per_model"]:
            if not require_free_mb(500, context=f"before {model_name} fp32 baseline"):
                print(f"Stopping before {model_name} (low RAM). Re-run to resume.")
                break
            m0 = load_fresh(model_name)
            labels, probs = get_predictions(m0, val_loader, is_resnet18)
            out["per_model"][model_name] = {
                "fp32_baseline": {"point": point_metrics(labels, probs)},
                "sparsity_levels": {},
            }
            del m0
            gc.collect()
            with open(OUT_JSON, "w") as f:
                json.dump(out, f, indent=2)
            print(f"\n=== {model_name}: fp32 baseline "
                  f"acc={out['per_model'][model_name]['fp32_baseline']['point']['accuracy']:.4f} ===",
                  flush=True)

        model_entry = out["per_model"][model_name]

        for sparsity in SPARSITY_LEVELS:
            key = str(sparsity)
            if key in model_entry["sparsity_levels"]:
                print(f"  {model_name} @ sparsity={sparsity}: already done, skipping", flush=True)
                continue
            if not require_free_mb(500, context=f"{model_name} sparsity={sparsity}"):
                print(f"Stopping before {model_name}@{sparsity} (low RAM). Re-run to resume.")
                break

            print(f"\n--- {model_name}: prune to {sparsity} + fine-tune ---", flush=True)
            t0 = time.time()
            m = load_fresh(model_name)

            # prune, keep reparametrization active (mask enforced every forward)
            params = prunable_modules(m)
            prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=sparsity)

            labels, probs = get_predictions(m, val_loader, is_resnet18)
            pre_ft_metrics = point_metrics(labels, probs)
            print(f"  immediately after pruning (no fine-tune yet): "
                  f"acc={pre_ft_metrics['accuracy']:.4f}", flush=True)

            optimizer = torch.optim.AdamW(m.parameters(), lr=FT_LR, weight_decay=1e-4)
            history = []
            for epoch in range(1, FT_EPOCHS + 1):
                if not require_free_mb(400, context=f"{model_name}@{sparsity} epoch {epoch}"):
                    print(f"  Stopping fine-tune early at epoch {epoch} (low RAM).")
                    break
                t_ep = time.time()
                train_loss = run_epoch(m, train_loader, criterion, optimizer, is_resnet18)
                dt = time.time() - t_ep
                history.append({"epoch": epoch, "train_loss": train_loss, "time_s": dt})
                print(f"    epoch {epoch}/{FT_EPOCHS} loss={train_loss:.4f} ({dt:.1f}s)", flush=True)

            # bake in the final pruned+fine-tuned weights permanently
            for module, name in params:
                prune.remove(module, name)

            labels, probs = get_predictions(m, val_loader, is_resnet18)
            post_ft_metrics = point_metrics(labels, probs)
            post_ft_ci = bootstrap_ci(labels, probs)
            total_time = time.time() - t0

            model_entry["sparsity_levels"][key] = {
                "immediately_after_pruning_no_finetune": pre_ft_metrics,
                "after_finetune": {"point": post_ft_metrics, "bootstrap_ci": post_ft_ci},
                "finetune_history": history,
                "accuracy_recovered": post_ft_metrics["accuracy"] - pre_ft_metrics["accuracy"],
                "total_time_s": total_time,
            }
            print(f"  after fine-tune: acc={post_ft_metrics['accuracy']:.4f} "
                  f"(recovered {model_entry['sparsity_levels'][key]['accuracy_recovered']:+.4f} "
                  f"from pre-finetune, total {total_time:.0f}s)", flush=True)

            out["per_model"][model_name] = model_entry
            with open(OUT_JSON, "w") as f:
                json.dump(out, f, indent=2)
            del m, optimizer
            gc.collect()

    print(f"\nAll done. Results at {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
