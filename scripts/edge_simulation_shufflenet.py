# -*- coding: utf-8 -*-
"""Edge-deployment simulation for the selected image-channel architecture
(ShuffleNetV2-1.0x, docs/architecture_comparison_size_vs_accuracy_2026-09-18.png /
build_architecture_comparison_report_2026-09-18.py Section 5.6).

No Raspberry Pi / Android device is available in this project -- this simulates the
"lightweight/edge AI" comparison on the CPU-only development machine instead, which the
report's own Section 8/10 already flags as not representative of real phone-class
latency. Treat every latency number here the same way: an internal, CPU-only proxy, not
a claim about real edge hardware.

Two things happen here:
  1. A magnitude-pruning sweep (unstructured, global, L1) at sparsity levels
     0%-90%, one-shot (no fine-tuning after pruning), evaluated on the VALIDATION
     set only.
  2. A quantization comparison: FP32 baseline, dynamic INT8 (Linear layers only),
     and static post-training INT8 (FX graph mode, full network, x86/onednn
     backend), also evaluated on the VALIDATION set only.

Deliberately does not open manifest_curated_v3_test.csv. This project's own test-set
discipline (docs/transfer_cnn_test_set_access_2026-09-18.md, Section 4.3 of the
architecture report) treats the test set as a resource to open once, only for a
finalized candidate list -- pruning/quantization variants are exploratory sweeps, not
finalized candidates, so they belong on validation like every other sweep in this
project's history (e.g. Section 5.2 of the architecture report).
"""
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mem_guard import require_free_mb  # noqa: E402
from paths import ROOT, SKINDISEASE_DIR  # noqa: E402

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)

IMG_SIZE = 224
CHECKPOINT = ROOT / "experiments" / "shufflenet_v2_x1_0" / "checkpoints" / "best_overall.pt"
VAL_MANIFEST = SKINDISEASE_DIR / "manifest_curated_v3_val.csv"
TRAIN_MANIFEST = SKINDISEASE_DIR / "manifest_curated_v3_train.csv"  # calibration only
OUT_JSON = ROOT / "papers" / "edge-ai-lightweight-deployment" / "edge_simulation_shufflenet_2026-09-19.json"

assert "test" not in str(VAL_MANIFEST).lower() or "test" not in VAL_MANIFEST.name
assert VAL_MANIFEST.name == "manifest_curated_v3_val.csv"
assert TRAIN_MANIFEST.name == "manifest_curated_v3_train.csv"


class CuratedDataset(Dataset):
    def __init__(self, manifest_path, transform):
        self.rows = []
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.rows.append((row["path"], int(row["label"])))
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label = self.rows[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def make_eval_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])


def build_shufflenet():
    backbone = models.shufflenet_v2_x1_0(weights=None)
    backbone.fc = nn.Sequential(
        nn.Linear(1024, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
    )
    return backbone


def load_baseline():
    model = build_shufflenet()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_labels, all_probs = [], []
    for imgs, labels in loader:
        logits = model(imgs)
        probs = torch.sigmoid(logits).numpy().ravel()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    acc = float(np.mean(np.array(preds) == np.array(all_labels)))
    metrics = {
        "accuracy": acc,
        "precision": float(precision_score(all_labels, preds, zero_division=0)),
        "recall": float(recall_score(all_labels, preds, zero_division=0)),
        "f1": float(f1_score(all_labels, preds, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


@torch.no_grad()
def measure_latency_ms(model, n_runs=30, n_warmup=5):
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    for _ in range(n_warmup):
        model(dummy)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(dummy)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def state_dict_disk_size_mb(model, path):
    torch.save(model.state_dict(), path)
    size_mb = path.stat().st_size / (1024 * 1024)
    path.unlink()
    return size_mb


def sparse_effective_size_mb(model):
    """Estimate of what a true sparse format (values + int32 indices for nonzero
    entries only) would need for the pruned Conv2d/Linear weights, plus dense storage
    for everything else (biases, BatchNorm, unpruned layers). This is a size a real
    sparse-tensor runtime could approach, NOT what torch.save produces for a masked-
    but-still-dense tensor (that stays the same size as unpruned -- measured
    separately and reported honestly, not conflated with this estimate)."""
    total_bytes = 0
    pruned_types = (nn.Conv2d, nn.Linear)
    pruned_param_ids = set()
    for module in model.modules():
        if isinstance(module, pruned_types) and hasattr(module, "weight"):
            w = module.weight.detach()
            nnz = int((w != 0).sum().item())
            # 4 bytes value (fp32) + 4 bytes index per nonzero, COO-style estimate.
            total_bytes += nnz * 8
            pruned_param_ids.add(id(module.weight))
    for p in model.parameters():
        if id(p) not in pruned_param_ids:
            total_bytes += p.numel() * 4
    for b in model.buffers():
        total_bytes += b.numel() * 4
    return total_bytes / (1024 * 1024)


def prunable_modules(model):
    return [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def run_pruning_sweep(val_loader, tmp_path):
    print("\n=== Pruning sweep (unstructured global L1, one-shot, no fine-tune) ===", flush=True)
    sparsities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    for s in sparsities:
        if not require_free_mb(400, context=f"pruning sparsity={s}"):
            print(f"Aborting pruning sweep at sparsity={s} (low RAM).")
            break
        model = load_baseline()
        params = prunable_modules(model)
        if s > 0:
            prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=s)
            for module, name in params:
                prune.remove(module, name)
        metrics = evaluate(model, val_loader)
        lat_mean, lat_std = measure_latency_ms(model)
        actual_zero_frac = _actual_sparsity(model)
        row = {
            "sparsity_target": s,
            "sparsity_actual": actual_zero_frac,
            "val_metrics": metrics,
            "inference_ms_mean": lat_mean,
            "inference_ms_std": lat_std,
            "state_dict_disk_mb_dense": state_dict_disk_size_mb(model, tmp_path),
            "sparse_effective_mb_estimate": sparse_effective_size_mb(model),
        }
        results.append(row)
        print(f"sparsity={s:.1f} (actual {actual_zero_frac:.3f}) "
              f"val_acc={metrics['accuracy']:.4f} val_f1={metrics['f1']:.4f} "
              f"val_auc={metrics['roc_auc']:.4f} "
              f"latency={lat_mean:.2f}ms sparse_est_mb={row['sparse_effective_mb_estimate']:.2f}",
              flush=True)
    return results


def _actual_sparsity(model):
    zeros, total = 0, 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.detach()
            zeros += int((w == 0).sum().item())
            total += w.numel()
    return zeros / total if total else 0.0


def run_quantization_comparison(val_loader, train_loader, tmp_path):
    print("\n=== Quantization: FP32 vs dynamic INT8 vs static INT8 (FX, x86/onednn) ===",
          flush=True)
    results = {}

    fp32 = load_baseline()
    fp32_metrics = evaluate(fp32, val_loader)
    lat_mean, lat_std = measure_latency_ms(fp32)
    results["fp32"] = {
        "val_metrics": fp32_metrics,
        "inference_ms_mean": lat_mean,
        "inference_ms_std": lat_std,
        "state_dict_disk_mb": state_dict_disk_size_mb(fp32, tmp_path),
    }
    print(f"fp32: acc={fp32_metrics['accuracy']:.4f} f1={fp32_metrics['f1']:.4f} "
          f"latency={lat_mean:.2f}ms size={results['fp32']['state_dict_disk_mb']:.2f}MB",
          flush=True)

    dyn = load_baseline()
    dyn_q = torch.ao.quantization.quantize_dynamic(dyn, {nn.Linear}, dtype=torch.qint8)
    dyn_metrics = evaluate(dyn_q, val_loader)
    lat_mean, lat_std = measure_latency_ms(dyn_q)
    torch.save(dyn_q.state_dict(), tmp_path)
    dyn_size = tmp_path.stat().st_size / (1024 * 1024)
    tmp_path.unlink()
    results["dynamic_int8_linear_only"] = {
        "val_metrics": dyn_metrics,
        "inference_ms_mean": lat_mean,
        "inference_ms_std": lat_std,
        "state_dict_disk_mb": dyn_size,
        "note": "Only the 2 Linear layers in the classification head are quantized -- "
                "dynamic quantization does not touch Conv2d, so most of this "
                "conv-heavy backbone stays FP32. Included for completeness, not "
                "expected to be the main size/speed win.",
    }
    print(f"dynamic INT8 (Linear only): acc={dyn_metrics['accuracy']:.4f} "
          f"f1={dyn_metrics['f1']:.4f} latency={lat_mean:.2f}ms size={dyn_size:.2f}MB",
          flush=True)

    print("Calibrating static INT8 (FX graph mode, x86 backend)...", flush=True)
    fp32_for_static = load_baseline()
    qconfig_mapping = get_default_qconfig_mapping("x86")
    example_inputs = (torch.randn(1, 3, IMG_SIZE, IMG_SIZE),)
    prepared = prepare_fx(fp32_for_static, qconfig_mapping, example_inputs)
    n_calib_batches = 8
    with torch.no_grad():
        for i, (imgs, _) in enumerate(train_loader):
            if i >= n_calib_batches:
                break
            prepared(imgs)
    static_q = convert_fx(prepared)
    static_metrics = evaluate(static_q, val_loader)
    lat_mean, lat_std = measure_latency_ms(static_q)
    torch.save(static_q.state_dict(), tmp_path)
    static_size = tmp_path.stat().st_size / (1024 * 1024)
    tmp_path.unlink()
    results["static_int8_fx"] = {
        "val_metrics": static_metrics,
        "inference_ms_mean": lat_mean,
        "inference_ms_std": lat_std,
        "state_dict_disk_mb": static_size,
        "calibration_batches": n_calib_batches,
        "calibration_images": n_calib_batches * train_loader.batch_size,
        "note": "Full-network INT8 (weights + activations), calibrated on TRAIN "
                "images only (never validation or test), x86/onednn backend "
                "(the only backend torch.backends.quantized.supported_engines "
                "reports on this machine -- fbgemm/qnnpack are not available "
                "in this torch build).",
    }
    print(f"static INT8 (FX): acc={static_metrics['accuracy']:.4f} "
          f"f1={static_metrics['f1']:.4f} latency={lat_mean:.2f}ms size={static_size:.2f}MB",
          flush=True)

    return results


def main():
    if not require_free_mb(400, context="startup"):
        print("Not enough free RAM to start. Aborting.")
        return

    eval_tf = make_eval_transform()
    val_ds = CuratedDataset(VAL_MANIFEST, eval_tf)
    train_ds = CuratedDataset(TRAIN_MANIFEST, eval_tf)  # eval_tf: deterministic, for calibration
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    print(f"Loaded val manifest: {len(val_ds)} images. Test manifest never opened.", flush=True)

    tmp_path = ROOT / "_edge_sim_tmp_checkpoint.pt"

    baseline_check = load_baseline()
    baseline_metrics = evaluate(baseline_check, val_loader)
    print(f"Sanity check -- reloaded baseline val accuracy: {baseline_metrics['accuracy']:.4f} "
          f"(expect ~0.7923 per experiments/shufflenet_v2_x1_0/config/config.json)", flush=True)

    pruning_results = run_pruning_sweep(val_loader, tmp_path)
    quant_results = run_quantization_comparison(val_loader, train_loader, tmp_path)

    out = {
        "model": "ShuffleNetV2-1.0x",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "val_manifest": str(VAL_MANIFEST.relative_to(ROOT)),
        "test_manifest_opened": False,
        "baseline_sanity_check_val_accuracy": baseline_metrics["accuracy"],
        "pruning_sweep": pruning_results,
        "quantization": quant_results,
        "notes": [
            "Latency is CPU-only, batch=1, this dev machine -- not representative of "
            "real phone/edge hardware (same caveat as Section 8 of "
            "build_architecture_comparison_report_2026-09-18.py).",
            "Pruning is one-shot magnitude pruning with NO fine-tuning after masking -- "
            "a sensitivity sweep, not a claim about the best achievable accuracy at "
            "each sparsity level (fine-tuning after pruning typically recovers some "
            "accuracy; not attempted here).",
            "Unstructured pruning zeros individual weights but does not physically "
            "shrink the dense tensors PyTorch stores/computes on CPU -- "
            "'state_dict_disk_mb_dense' will NOT shrink with sparsity; "
            "'sparse_effective_mb_estimate' is a COO-format estimate of what a real "
            "sparse runtime could achieve, not a measured number.",
        ],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
