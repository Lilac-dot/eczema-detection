# -*- coding: utf-8 -*-
"""Edge-deployment simulation (pruning sweep + quantization) across all 9 image
architectures from the architecture-selection comparison
(papers/architecture-selection-report/honors-paper-report.docx, Section 5): the
ResNet18 baseline plus the 8 lightweight candidates.

Generalizes scripts/edge_simulation_shufflenet.py (which covered ShuffleNetV2-1.0x
only) into the full comparison requested for the standalone edge-AI paper
(papers/edge-ai-lightweight-deployment/). Same discipline as that script: validation
set only, test manifest never opened (pruning/quantization variants are an exploratory
sweep, not a finalized candidate list -- docs/transfer_cnn_test_set_access_2026-09-18.md).

Reuses train_transfer_cnn.py's MODEL_CONFIGS/build_model for the 8 transfer-learning
candidates (identical architecture/head to how they were actually trained) and a
separate branch for ResNet18 (different head: 2-class softmax, not a single sigmoid
unit -- scripts/eval_curated_cnn_balanced.py).

CPU-only, this project's 8GB dev machine. No real edge/mobile hardware available --
every latency number here is a CPU-only proxy, the same caveat already demonstrated
concretely in edge_simulation_shufflenet.py (static INT8 measured SLOWER than FP32 on
this machine, the opposite of quantization's usual purpose).
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
import torch.nn.utils.prune as prune
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mem_guard import require_free_mb  # noqa: E402
from paths import ROOT, SKINDISEASE_DIR, MODELS_DIR  # noqa: E402
import train_transfer_cnn as tcnn  # noqa: E402 -- MODEL_CONFIGS, build_model (safe: no __main__ side effects)

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)

IMG_SIZE = 224
VAL_MANIFEST = SKINDISEASE_DIR / "manifest_curated_v3_val.csv"
TRAIN_MANIFEST = SKINDISEASE_DIR / "manifest_curated_v3_train.csv"  # calibration only
OUT_DIR = ROOT / "papers" / "edge-ai-lightweight-deployment"
OUT_JSON = OUT_DIR / "edge_simulation_all_architectures_2026-09-19.json"

assert VAL_MANIFEST.name == "manifest_curated_v3_val.csv"
assert TRAIN_MANIFEST.name == "manifest_curated_v3_train.csv"

RESNET18_PATH = MODELS_DIR / "curated_resnet18_balanced.pt"

CANDIDATES = [
    "efficientnet_b0", "efficientnet_lite0", "mobilenetv3_small", "mobilenetv2_100",
    "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "squeezenet1_1", "repghostnet_050",
]
ALL_MODELS = ["resnet18"] + CANDIDATES

SPARSITIES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


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


def build_resnet18_baseline():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(RESNET18_PATH, map_location="cpu"))
    model.eval()
    return model


def build_candidate(model_name):
    """Rebuilds a transfer-learning candidate with random-init backbone (weights=None
    equivalent -- avoids re-downloading ImageNet weights just to overwrite them) then
    loads this project's own fine-tuned checkpoint over it."""
    cfg = tcnn.MODEL_CONFIGS[model_name]
    if cfg["source"] == "torchvision":
        ctor = getattr(models, cfg["ctor"])
        backbone = ctor(weights=None)
    else:
        backbone = tcnn.timm.create_model(cfg["ctor"], pretrained=False, num_classes=1000)
    final_attr = cfg["final_attr"]
    existing_final = getattr(backbone, final_attr)
    if cfg["pattern"] == "squeezenet":
        new_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(cfg["in_features"], 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, 1),
        )
    else:
        in_features = cfg["in_features"] or tcnn._first_linear_in_features(existing_final)
        new_head = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, 1),
        )
    setattr(backbone, final_attr, new_head)
    ckpt_path = ROOT / "experiments" / model_name / "checkpoints" / "best_overall.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone.load_state_dict(state)
    backbone.eval()
    return backbone


def load_model(model_name):
    if model_name == "resnet18":
        return build_resnet18_baseline()
    return build_candidate(model_name)


@torch.no_grad()
def evaluate(model, loader, is_resnet18):
    model.eval()
    all_labels, all_probs = [], []
    for imgs, labels in loader:
        if is_resnet18:
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy().ravel()
        else:
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
def measure_latency_ms(model, n_runs=20, n_warmup=4):
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
    total_bytes = 0
    pruned_param_ids = set()
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight"):
            w = module.weight.detach()
            nnz = int((w != 0).sum().item())
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


def actual_sparsity(model):
    zeros, total = 0, 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.detach()
            zeros += int((w == 0).sum().item())
            total += w.numel()
    return zeros / total if total else 0.0


def run_pruning_sweep(model_name, val_loader, tmp_path):
    is_resnet18 = model_name == "resnet18"
    results = []
    for s in SPARSITIES:
        if not require_free_mb(400, context=f"{model_name} pruning sparsity={s}"):
            print(f"  Aborting {model_name} pruning sweep at sparsity={s} (low RAM).")
            break
        model = load_model(model_name)
        params = prunable_modules(model)
        if s > 0:
            prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=s)
            for module, name in params:
                prune.remove(module, name)
        metrics = evaluate(model, val_loader, is_resnet18)
        lat_mean, lat_std = measure_latency_ms(model)
        row = {
            "sparsity_target": s,
            "sparsity_actual": actual_sparsity(model),
            "val_metrics": metrics,
            "inference_ms_mean": lat_mean,
            "inference_ms_std": lat_std,
            "state_dict_disk_mb_dense": state_dict_disk_size_mb(model, tmp_path),
            "sparse_effective_mb_estimate": sparse_effective_size_mb(model),
        }
        results.append(row)
        print(f"  sparsity={s:.1f} acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
              f"lat={lat_mean:.1f}ms", flush=True)
        del model
        gc.collect()
    return results


def run_quantization(model_name, val_loader, train_loader, tmp_path):
    is_resnet18 = model_name == "resnet18"
    results = {}

    fp32 = load_model(model_name)
    fp32_metrics = evaluate(fp32, val_loader, is_resnet18)
    lat_mean, lat_std = measure_latency_ms(fp32)
    results["fp32"] = {
        "val_metrics": fp32_metrics, "inference_ms_mean": lat_mean, "inference_ms_std": lat_std,
        "state_dict_disk_mb": state_dict_disk_size_mb(fp32, tmp_path),
    }
    print(f"  fp32 acc={fp32_metrics['accuracy']:.4f} lat={lat_mean:.1f}ms", flush=True)
    del fp32
    gc.collect()

    dyn = load_model(model_name)
    dyn_q = torch.ao.quantization.quantize_dynamic(dyn, {nn.Linear}, dtype=torch.qint8)
    dyn_metrics = evaluate(dyn_q, val_loader, is_resnet18)
    lat_mean, lat_std = measure_latency_ms(dyn_q)
    torch.save(dyn_q.state_dict(), tmp_path)
    dyn_size = tmp_path.stat().st_size / (1024 * 1024)
    tmp_path.unlink()
    results["dynamic_int8_linear_only"] = {
        "val_metrics": dyn_metrics, "inference_ms_mean": lat_mean, "inference_ms_std": lat_std,
        "state_dict_disk_mb": dyn_size,
    }
    print(f"  dynamic_int8 acc={dyn_metrics['accuracy']:.4f} lat={lat_mean:.1f}ms "
          f"size={dyn_size:.2f}MB", flush=True)
    del dyn, dyn_q
    gc.collect()

    try:
        fp32_for_static = load_model(model_name)
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
        static_metrics = evaluate(static_q, val_loader, is_resnet18)
        lat_mean, lat_std = measure_latency_ms(static_q)
        torch.save(static_q.state_dict(), tmp_path)
        static_size = tmp_path.stat().st_size / (1024 * 1024)
        tmp_path.unlink()
        results["static_int8_fx"] = {
            "val_metrics": static_metrics, "inference_ms_mean": lat_mean, "inference_ms_std": lat_std,
            "state_dict_disk_mb": static_size,
            "calibration_batches": n_calib_batches,
        }
        print(f"  static_int8 acc={static_metrics['accuracy']:.4f} lat={lat_mean:.1f}ms "
              f"size={static_size:.2f}MB", flush=True)
        del fp32_for_static, prepared, static_q
        gc.collect()
    except Exception as e:
        print(f"  static_int8 FAILED for {model_name}: {e!r}", flush=True)
        results["static_int8_fx"] = {"error": repr(e)}

    return results


def main():
    if not require_free_mb(400, context="startup"):
        print("Not enough free RAM to start. Aborting.")
        return

    eval_tf = make_eval_transform()
    val_ds = CuratedDataset(VAL_MANIFEST, eval_tf)
    train_ds = CuratedDataset(TRAIN_MANIFEST, eval_tf)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    print(f"Loaded val manifest: {len(val_ds)} images. Test manifest never opened.", flush=True)

    tmp_path = ROOT / "_edge_sim_all_tmp_checkpoint.pt"

    out = {}
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            out = json.load(f)
        print(f"Resuming -- {len(out)} model(s) already done: {list(out.keys())}", flush=True)

    for model_name in ALL_MODELS:
        if model_name in out:
            print(f"\n=== {model_name}: already done, skipping ===", flush=True)
            continue
        if not require_free_mb(400, context=f"before {model_name}"):
            print(f"Stopping before {model_name} (low RAM). Re-run script later to resume.")
            break
        print(f"\n=== {model_name} ===", flush=True)
        sanity = evaluate(load_model(model_name), val_loader, model_name == "resnet18")
        print(f"  sanity check val_accuracy={sanity['accuracy']:.4f}", flush=True)

        print(" pruning sweep:", flush=True)
        pruning_results = run_pruning_sweep(model_name, val_loader, tmp_path)
        print(" quantization:", flush=True)
        quant_results = run_quantization(model_name, val_loader, train_loader, tmp_path)

        out[model_name] = {
            "sanity_check_val_accuracy": sanity["accuracy"],
            "pruning_sweep": pruning_results,
            "quantization": quant_results,
        }
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved progress ({len(out)}/{len(ALL_MODELS)} models done) to {OUT_JSON}", flush=True)

    print(f"\nAll done. Final results at {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
