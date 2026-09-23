# -*- coding: utf-8 -*-
"""Narrow external-validation extension, 2026-09-20: does the ARM/qnnpack INT8
quantization collapse found on this project's OWN internal validation set (Section 5.5
of eczema_compression_deployment_paper_2026-09-20.docx) also appear on SkinDisNet, a
completely independent external clinical photo source?

This deliberately does NOT re-test "does accuracy generalize" -- that is already
answered (zero-shot AUC ~0.48, chance-level, for both ResNet18 and ShuffleNetV2-1.0x;
docs/skindisnet_zeroshot_shufflenetv2_2026-09-18.md) and re-running it would be
redundant. Comparing accuracy magnitudes near chance would also hit the same floor-
effect problem already documented twice this session (robustness_under_compression,
Section 6.2) -- there is no headroom left to show a further drop. Instead, this checks
a structural, floor-effect-immune question: is the qnnpack collapse signature (a fully
degenerate, effectively constant-output classifier: F1 < 0.05) a stable property of the
trained weights + quantization backend, reproducing on totally different images, or an
artifact specific to this project's own internal validation set's pixel statistics?

Two architectures: ShuffleNetV2-1.0x (this project's deployment candidate, NOT flagged
as collapsing internally -- the control case) and MobileNetV2-1.0x (flagged as
collapsing internally -- the case expected to reproduce collapse if the effect is a
stable weights+backend property). Both evaluated FP32 and qnnpack-INT8, zero-shot, no
SkinDisNet fine-tuning. Quantization calibration uses this project's own training split
(SkinDisease/manifest_curated_v3_train.csv), NOT any SkinDisNet data, identical
methodology to every other quantization result in this project -- SkinDisNet is used
only as an external EVALUATION set here, never for calibration or training.

Results are also broken down by SkinDisNet's own disease_class label, specifically
Atopic Dermatitis (AD) vs. Eczema (EC) -- SkinDisNet keeps these as separate classes
even though both map to this project's positive ("Eczema") label -- to report an
AD-specific number directly, not just a merged Eczema+AD figure.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT, SKINDISNET_DIR  # noqa: E402
from eval_external_common import ExternalDataset  # noqa: E402
from edge_ai_extended_analysis import (  # noqa: E402
    build_candidate, quantize_static, point_metrics, bootstrap_ci, CuratedDataset,
    make_eval_transform, TRAIN_MANIFEST, QUANT_BACKEND,
)

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)

IMG_SIZE = 224
MANIFEST = SKINDISNET_DIR / "manifest_skindisnet_clean.csv"
MODELS = ["shufflenet_v2_x1_0", "mobilenetv2_100"]
OUT_JSON = ROOT / "papers" / "edge-ai-lightweight-deployment" / "skindisnet_backend_collapse_check_2026-09-20.json"


def eval_tf():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), normalize,
    ])


@torch.no_grad()
def run_inference_with_class(model, loader):
    model.eval()
    all_labels, all_probs, all_cls = [], [], []
    for imgs, labels, cls in loader:
        probs = torch.sigmoid(model(imgs)).squeeze(-1).numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist() if torch.is_tensor(labels) else list(labels))
        all_cls.extend(cls)
    return np.array(all_labels), np.array(all_probs), np.array(all_cls)


def per_class_recall(labels, probs, classes, threshold=0.5):
    """For each positive-label disease_class (AD, EC), what fraction of that class's
    images were correctly predicted positive -- reported per-subclass since SkinDisNet
    keeps Atopic Dermatitis and Eczema as separate labels even though both map to this
    project's single positive class."""
    preds = (probs >= threshold).astype(int)
    out = {}
    for cls_name in sorted(set(classes)):
        mask = classes == cls_name
        if labels[mask][0] != 1:
            continue  # only meaningful for positive-label subclasses (AD, EC)
        out[cls_name] = {
            "n": int(mask.sum()),
            "recall": float((preds[mask] == 1).mean()),
        }
    return out


def main():
    ds = ExternalDataset(MANIFEST, eval_tf())
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                         collate_fn=lambda batch: (
                             torch.stack([b[0] for b in batch]),
                             torch.tensor([b[1] for b in batch]),
                             [b[2] for b in batch],
                         ))
    print(f"SkinDisNet external eval set: n={len(ds)}")

    train_ds = CuratedDataset(TRAIN_MANIFEST, make_eval_transform())
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    print(f"Quantized backend in use: {QUANT_BACKEND}")

    out = {}
    for model_name in MODELS:
        print(f"\n=== {model_name} ===")
        fp32 = build_candidate(model_name)
        labels, probs, classes = run_inference_with_class(fp32, loader)
        fp32_metrics = point_metrics(labels, probs)
        fp32_ci = bootstrap_ci(labels, probs)
        fp32_subclass = per_class_recall(labels, probs, classes)
        print(f"  FP32:    acc={fp32_metrics['accuracy']:.4f} sens={fp32_metrics['sensitivity_recall']:.4f} "
              f"spec={fp32_metrics['specificity']:.4f} f1={fp32_metrics['f1']:.4f} auroc={fp32_metrics['auroc']:.4f}")
        print(f"  FP32 subclass recall: {fp32_subclass}")

        int8_model = quantize_static(build_candidate(model_name), train_loader)
        labels_q, probs_q, classes_q = run_inference_with_class(int8_model, loader)
        int8_metrics = point_metrics(labels_q, probs_q)
        int8_ci = bootstrap_ci(labels_q, probs_q)
        int8_subclass = per_class_recall(labels_q, probs_q, classes_q)
        collapsed = int8_metrics["f1"] < 0.05
        print(f"  qnnpack INT8: acc={int8_metrics['accuracy']:.4f} sens={int8_metrics['sensitivity_recall']:.4f} "
              f"spec={int8_metrics['specificity']:.4f} f1={int8_metrics['f1']:.4f} auroc={int8_metrics['auroc']:.4f} "
              f"{'*** COLLAPSED ***' if collapsed else ''}")
        print(f"  qnnpack INT8 subclass recall: {int8_subclass}")

        out[model_name] = {
            "n_external": len(ds),
            "fp32": {"point": fp32_metrics, "bootstrap_ci": fp32_ci, "subclass_recall": fp32_subclass},
            "qnnpack_int8": {"point": int8_metrics, "bootstrap_ci": int8_ci,
                              "subclass_recall": int8_subclass, "collapsed": collapsed},
        }
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=2)

    print(f"\nAll done. Results at {OUT_JSON}")


if __name__ == "__main__":
    main()
