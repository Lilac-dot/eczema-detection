# -*- coding: utf-8 -*-
"""Extended analysis for the edge-AI compression paper revision (2026-09-19), addressing
reviewer-style requests: bootstrap CIs, sensitivity/specificity/AUROC, class distribution
and majority-class baseline, quantized operator/dtype coverage, combined pruning+INT8,
a code-level (not retrained) SE/swish ablation, and peak-memory measurement.

Validation set only -- test manifest never opened, same discipline as every other
exploratory sweep in this project. No new training occurs in this script (SE/swish
"ablation" is a forward-pass module substitution on ALREADY-TRAINED weights, not a
retrain -- see run_se_swish_ablation()'s docstring for exactly what that can and can't
show).
"""
import copy
import csv
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mem_guard import require_free_mb  # noqa: E402
from paths import ROOT, SKINDISEASE_DIR, MODELS_DIR  # noqa: E402
import train_transfer_cnn as tcnn  # noqa: E402

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)

# Quantized backend: the original runs of this project happened on an x86 (fbgemm)
# Windows machine; this Apple Silicon machine only supports qnnpack. Pick whichever
# this machine actually supports so the same code runs on either, but this means
# INT8 numbers produced here use a different backend than the ones already on record
# -- both are standard affine INT8 quantization, but note this as a methodological
# caveat rather than silently treating them as identical.
QUANT_BACKEND = "x86" if "x86" in torch.backends.quantized.supported_engines else "qnnpack"
torch.backends.quantized.engine = QUANT_BACKEND

IMG_SIZE = 224
VAL_MANIFEST = SKINDISEASE_DIR / "manifest_curated_v3_val.csv"
TRAIN_MANIFEST = SKINDISEASE_DIR / "manifest_curated_v3_train.csv"
OUT_DIR = ROOT / "papers" / "edge-ai-lightweight-deployment"
OUT_JSON = OUT_DIR / "edge_ai_extended_analysis_2026-09-19.json"
SOURCE_JSON = OUT_DIR / "edge_simulation_all_architectures_2026-09-19.json"
RESNET18_PATH = MODELS_DIR / "curated_resnet18_balanced.pt"

assert VAL_MANIFEST.name == "manifest_curated_v3_val.csv"
assert TRAIN_MANIFEST.name == "manifest_curated_v3_train.csv"

CANDIDATES = [
    "efficientnet_b0", "efficientnet_lite0", "mobilenetv3_small", "mobilenetv2_100",
    "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "squeezenet1_1", "repghostnet_050",
]
ALL_MODELS = ["resnet18"] + CANDIDATES
SE_SWISH_MODELS = ["efficientnet_b0", "mobilenetv3_small", "repghostnet_050"]


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
        full_path = path if Path(path).is_absolute() else ROOT / path
        img = Image.open(full_path).convert("RGB")
        return self.transform(img), label


def make_eval_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), normalize,
    ])


def build_resnet18_baseline():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(RESNET18_PATH, map_location="cpu"))
    model.eval()
    return model


def build_candidate(model_name):
    cfg = tcnn.MODEL_CONFIGS[model_name]
    if cfg["source"] == "torchvision":
        backbone = getattr(models, cfg["ctor"])(weights=None)
    else:
        backbone = tcnn.timm.create_model(cfg["ctor"], pretrained=False, num_classes=1000)
    final_attr = cfg["final_attr"]
    existing_final = getattr(backbone, final_attr)
    if cfg["pattern"] == "squeezenet":
        new_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(cfg["in_features"], 128), nn.ReLU(inplace=True),
                                  nn.Dropout(0.3), nn.Linear(128, 1))
    else:
        in_features = cfg["in_features"] or tcnn._first_linear_in_features(existing_final)
        new_head = nn.Sequential(nn.Linear(in_features, 128), nn.ReLU(inplace=True),
                                  nn.Dropout(0.3), nn.Linear(128, 1))
    setattr(backbone, final_attr, new_head)
    ckpt_path = ROOT / "experiments" / model_name / "checkpoints" / "best_overall.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone.load_state_dict(state)
    backbone.eval()
    return backbone


def load_model(model_name):
    return build_resnet18_baseline() if model_name == "resnet18" else build_candidate(model_name)


@torch.no_grad()
def get_predictions(model, loader, is_resnet18):
    model.eval()
    all_labels, all_probs = [], []
    for imgs, labels in loader:
        logits = model(imgs)
        probs = (torch.softmax(logits, dim=1)[:, 1] if is_resnet18
                  else torch.sigmoid(logits).squeeze(-1))
        all_probs.extend(probs.numpy().ravel().tolist())
        all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_probs)


def point_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")   # a.k.a. recall
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    metrics = {
        "accuracy": float(np.mean(preds == labels)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "sensitivity_recall": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["auroc"] = float("nan")
    return metrics


def bootstrap_ci(labels, probs, n_boot=2000, seed=42, alpha=0.05):
    """2000-resample nonparametric bootstrap, same convention as the rest of this
    project's significance testing (e.g. eval_paired_significance_stageB.py)."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    keys = ["accuracy", "precision", "sensitivity_recall", "specificity", "f1", "auroc"]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        bl, bp = labels[idx], probs[idx]
        if len(np.unique(bl)) < 2:
            continue
        m = point_metrics(bl, bp)
        for k in keys:
            if not np.isnan(m[k]):
                samples[k].append(m[k])
    out = {}
    for k in keys:
        arr = np.array(samples[k])
        if len(arr) == 0:
            out[k] = {"mean": None, "ci_lo": None, "ci_hi": None, "n_valid": 0}
            continue
        out[k] = {
            "mean": float(np.mean(arr)),
            "ci_lo": float(np.percentile(arr, 100 * alpha / 2)),
            "ci_hi": float(np.percentile(arr, 100 * (1 - alpha / 2))),
            "n_valid": int(len(arr)),
        }
    return out


def class_distribution(manifest_path):
    labels = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(int(row["label"]))
    labels = np.array(labels)
    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    majority_frac = max(n_pos, n_neg) / n
    return {
        "n": n, "n_eczema_pos": n_pos, "n_other_neg": n_neg,
        "pos_fraction": n_pos / n, "majority_class_accuracy": majority_frac,
    }


def prunable_modules(model):
    return [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def prune_to_sparsity(model, sparsity):
    if sparsity <= 0:
        return model
    params = prunable_modules(model)
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=sparsity)
    for module, name in params:
        prune.remove(module, name)
    return model


def quantize_static(model, train_loader, n_calib_batches=8):
    qconfig_mapping = get_default_qconfig_mapping(QUANT_BACKEND)
    example_inputs = (torch.randn(1, 3, IMG_SIZE, IMG_SIZE),)
    prepared = prepare_fx(model, qconfig_mapping, example_inputs)
    with torch.no_grad():
        for i, (imgs, _) in enumerate(train_loader):
            if i >= n_calib_batches:
                break
            prepared(imgs)
    return convert_fx(prepared)


def quantized_op_coverage(model):
    """Counts modules by whether they ended up as an actual quantized (int8) module,
    a float-fallback module of a type the quantizer normally converts (Conv2d/Linear
    left in fp32 because it wasn't wrapped), or something else (BatchNorm folded away,
    activation, etc. -- not counted as 'coverage' either way)."""
    quantized_types = 0
    float_fallback = 0
    details = []
    for name, module in model.named_modules():
        cls = type(module)
        mod_path = cls.__module__
        is_quantized_conv_or_linear = (
            mod_path.startswith("torch.ao.nn.quantized") or
            mod_path.startswith("torch.ao.nn.intrinsic.quantized")
        ) and cls.__name__ in ("Conv2d", "Linear", "ConvReLU2d", "LinearReLU", "ConvBnReLU2d")
        if is_quantized_conv_or_linear:
            quantized_types += 1
            details.append((name, cls.__name__, "int8"))
        elif isinstance(module, (nn.Conv2d, nn.Linear)) and not mod_path.startswith("torch.ao.nn"):
            float_fallback += 1
            details.append((name, cls.__name__, "fp32_fallback"))
    total_convertible = quantized_types + float_fallback
    coverage = quantized_types / total_convertible if total_convertible else float("nan")
    return {
        "n_quantized_int8": quantized_types,
        "n_fp32_fallback": float_fallback,
        "coverage_fraction": coverage,
        "fallback_module_names": [d[0] for d in details if d[2] == "fp32_fallback"],
    }


@torch.no_grad()
def peak_memory_mb(model, n_runs=10):
    """Peak resident-set-size delta during repeated single-image inference, measured
    via this process's own RSS (psutil) -- a proxy for the model's runtime memory
    footprint, not a clean isolated measurement (this process also holds Python/torch
    overhead, other loaded models, etc. -- stated as a limitation, not hidden)."""
    proc = psutil.Process()
    gc.collect()
    baseline_mb = proc.memory_info().rss / (1024 * 1024)
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    peak = baseline_mb
    for _ in range(n_runs):
        model(dummy)
        cur = proc.memory_info().rss / (1024 * 1024)
        peak = max(peak, cur)
    return {"baseline_rss_mb": baseline_mb, "peak_rss_mb": peak, "delta_mb": peak - baseline_mb}


def run_se_swish_ablation(train_loader, val_loader):
    """Forward-pass module substitution on ALREADY-TRAINED weights -- NOT a retrain.
    For each of the 3 SE/swish models, builds two variants at eval time only:
      (a) SE modules replaced with an identity (bypasses the excitation gate,
          scale=1 always) -- isolates SE's contribution to quantization fragility
          without touching any trained weight.
      (b) Swish-family activations (SiLU/Hardswish) replaced with ReLU/ReLU6.
    Both variants are then quantized (static INT8) exactly as the original was, and
    compared. This is a correlational-to-causal STRENGTHENING, not a clean causal
    proof: the modified networks were never trained with the substituted component, so
    their FP32 accuracy is expected to drop too (reported alongside, not hidden) --
    what matters for the causal question is whether the FP32-to-INT8 GAP shrinks when
    the component is removed, not the absolute accuracy of the modified network.
    """
    results = {}
    for model_name in SE_SWISH_MODELS:
        print(f"\n--- SE/swish ablation: {model_name} ---", flush=True)
        variants = {}

        # baseline (original, already have full numbers elsewhere, but recomputed
        # here so all three variants are measured identically in this script)
        m = load_model(model_name)
        fp32_metrics = point_metrics(*get_predictions(m, val_loader, False))
        m_q = quantize_static(load_model(model_name), train_loader)
        int8_metrics = point_metrics(*get_predictions(m_q, val_loader, False))
        variants["original"] = {"fp32": fp32_metrics, "int8": int8_metrics,
                                  "fp32_to_int8_drop_acc": fp32_metrics["accuracy"] - int8_metrics["accuracy"]}
        del m, m_q
        gc.collect()

        # SE -> identity
        m_nose = load_model(model_name)
        n_replaced = 0
        for name, module in m_nose.named_modules():
            cls = type(module).__name__
            if cls in ("SqueezeExcitation", "SqueezeExcite"):
                parent = m_nose
                *path, last = name.split(".")
                for p in path:
                    parent = getattr(parent, p)
                setattr(parent, last, nn.Identity())
                n_replaced += 1
        if n_replaced > 0:
            fp32_nose = point_metrics(*get_predictions(m_nose, val_loader, False))
            m_nose_q = quantize_static(m_nose, train_loader)
            int8_nose = point_metrics(*get_predictions(m_nose_q, val_loader, False))
            variants["se_removed"] = {
                "n_se_modules_replaced": n_replaced,
                "fp32": fp32_nose, "int8": int8_nose,
                "fp32_to_int8_drop_acc": fp32_nose["accuracy"] - int8_nose["accuracy"],
            }
            del m_nose_q
        del m_nose
        gc.collect()

        # swish/hardswish -> relu/relu6
        m_relu = load_model(model_name)
        n_replaced = 0
        for name, module in list(m_relu.named_modules()):
            cls = type(module).__name__
            if cls in ("SiLU",):
                parent = m_relu
                *path, last = name.split(".")
                for p in path:
                    parent = getattr(parent, p)
                setattr(parent, last, nn.ReLU(inplace=False))
                n_replaced += 1
            elif cls in ("Hardswish",):
                parent = m_relu
                *path, last = name.split(".")
                for p in path:
                    parent = getattr(parent, p)
                setattr(parent, last, nn.ReLU6(inplace=False))
                n_replaced += 1
        if n_replaced > 0:
            fp32_relu = point_metrics(*get_predictions(m_relu, val_loader, False))
            m_relu_q = quantize_static(m_relu, train_loader)
            int8_relu = point_metrics(*get_predictions(m_relu_q, val_loader, False))
            variants["activation_replaced"] = {
                "n_activation_modules_replaced": n_replaced,
                "fp32": fp32_relu, "int8": int8_relu,
                "fp32_to_int8_drop_acc": fp32_relu["accuracy"] - int8_relu["accuracy"],
            }
            del m_relu_q
        del m_relu
        gc.collect()

        results[model_name] = variants
        print(json.dumps(variants, indent=2, default=str)[:800], flush=True)
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

    out = {}
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            out = json.load(f)
        print(f"Resuming -- sections already present: {list(out.keys())}", flush=True)

    if "class_distribution" not in out:
        out["class_distribution"] = {
            "val": class_distribution(VAL_MANIFEST),
            "train": class_distribution(TRAIN_MANIFEST),
        }
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=2)
        print("Saved class_distribution.", flush=True)

    with open(SOURCE_JSON) as f:
        source = json.load(f)

    if "per_model" not in out:
        out["per_model"] = {}

    for model_name in ALL_MODELS:
        if model_name in out["per_model"]:
            print(f"\n=== {model_name}: already done, skipping ===", flush=True)
            continue
        if not require_free_mb(400, context=f"before {model_name}"):
            print(f"Stopping before {model_name} (low RAM). Re-run to resume.")
            break
        print(f"\n=== {model_name} ===", flush=True)
        is_resnet18 = model_name == "resnet18"

        # knee sparsity from the original sweep (first level with >5pt drop or F1<0.05)
        sweep = source[model_name]["pruning_sweep"]
        base_acc = sweep[0]["val_metrics"]["accuracy"]
        knee = 0.9
        for row in sweep:
            if base_acc - row["val_metrics"]["accuracy"] > 0.05 or row["val_metrics"]["f1"] < 0.05:
                knee = row["sparsity_target"]
                break

        model_out = {"knee_sparsity_used": knee}

        # FP32 baseline
        m = load_model(model_name)
        labels, probs = get_predictions(m, val_loader, is_resnet18)
        model_out["fp32"] = {
            "point": point_metrics(labels, probs),
            "bootstrap_ci": bootstrap_ci(labels, probs),
            "memory_mb": peak_memory_mb(m),
        }
        del m; gc.collect()
        print(f"  fp32 done: acc={model_out['fp32']['point']['accuracy']:.4f}", flush=True)

        # pruned to knee
        m = load_model(model_name)
        m = prune_to_sparsity(m, knee)
        labels, probs = get_predictions(m, val_loader, is_resnet18)
        model_out["pruned_at_knee"] = {
            "point": point_metrics(labels, probs),
            "bootstrap_ci": bootstrap_ci(labels, probs),
        }
        del m; gc.collect()
        print(f"  pruned@{knee} done: acc={model_out['pruned_at_knee']['point']['accuracy']:.4f}", flush=True)

        # static INT8
        m = load_model(model_name)
        m_q = quantize_static(m, train_loader)
        labels, probs = get_predictions(m_q, val_loader, is_resnet18)
        model_out["static_int8"] = {
            "point": point_metrics(labels, probs),
            "bootstrap_ci": bootstrap_ci(labels, probs),
            "op_coverage": quantized_op_coverage(m_q),
            "memory_mb": peak_memory_mb(m_q),
        }
        del m, m_q; gc.collect()
        print(f"  static_int8 done: acc={model_out['static_int8']['point']['accuracy']:.4f} "
              f"coverage={model_out['static_int8']['op_coverage']['coverage_fraction']:.2f}", flush=True)

        # combined: prune to a SAFE sparsity (30%, well inside every model's knee) + INT8
        m = load_model(model_name)
        m = prune_to_sparsity(m, 0.3)
        m_q = quantize_static(m, train_loader)
        labels, probs = get_predictions(m_q, val_loader, is_resnet18)
        model_out["pruned30_plus_int8"] = {
            "point": point_metrics(labels, probs),
            "bootstrap_ci": bootstrap_ci(labels, probs),
        }
        del m, m_q; gc.collect()
        print(f"  pruned30+int8 done: acc={model_out['pruned30_plus_int8']['point']['accuracy']:.4f}",
              flush=True)

        out["per_model"][model_name] = model_out
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved progress ({len(out['per_model'])}/{len(ALL_MODELS)}).", flush=True)

    if "se_swish_ablation" not in out:
        if not require_free_mb(400, context="before se_swish_ablation"):
            print("Not enough RAM for SE/swish ablation. Re-run to resume.")
        else:
            out["se_swish_ablation"] = run_se_swish_ablation(train_loader, val_loader)
            with open(OUT_JSON, "w") as f:
                json.dump(out, f, indent=2)
            print("Saved se_swish_ablation.", flush=True)

    print(f"\nAll done. Results at {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
