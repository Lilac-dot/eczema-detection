# -*- coding: utf-8 -*-
"""External validation on SkinDisNet, ALL 9 architectures, 2026-09-20 (extends the
2-architecture check in skindisnet_backend_collapse_check.py per explicit request during
publication-readiness review). Zero-shot only -- no architecture is fine-tuned on any
SkinDisNet data, and INT8 calibration uses ONLY this project's own training split
(SkinDisease/manifest_curated_v3_train.csv), never SkinDisNet or test data.

Conditions per architecture: FP32 and ARM/qnnpack static INT8. x86/fbgemm INT8 is NOT
attempted here -- this machine's PyTorch build only supports the qnnpack quantized
engine (torch.backends.quantized.supported_engines == ['qnnpack', 'none']); there is no
way to execute an x86/fbgemm-quantized model on this hardware. This is reported as an
infeasible experiment in the output JSON, not fabricated or silently skipped.

SkinDisNet label mapping, verified and documented here explicitly (not assumed): the
manifest (manifest_skindisnet_clean.csv) carries each image's original disease_class
(Seborrheic Dermatitis (SD), Contact Dermatitis (CD), Scabies (SC), Eczema (EC), Tinea
Corporis (TC), Atopic Dermatitis (AD)) alongside a binary `label` column. Verified below
at runtime: label==1 iff disease_class is "Eczema (EC)" or "Atopic Dermatitis (AD)",
label==0 for the other four classes -- i.e. Eczema and Atopic Dermatitis are merged into
this project's single positive ("eczema-like") class, matching this project's own binary
task definition (Eczema vs. everything else). This mapping is REUSED from the existing,
already-built manifest, not redefined here.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT, SKINDISNET_DIR  # noqa: E402
from eval_external_common import ExternalDataset  # noqa: E402
from edge_ai_extended_analysis import (  # noqa: E402
    build_candidate, build_resnet18_baseline, quantize_static, point_metrics,
    bootstrap_ci, CuratedDataset, make_eval_transform, TRAIN_MANIFEST, ALL_MODELS,
    QUANT_BACKEND,
)

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)

IMG_SIZE = 224
MANIFEST = SKINDISNET_DIR / "manifest_skindisnet_clean.csv"
OUT_JSON = ROOT / "papers" / "edge-ai-lightweight-deployment" / "skindisnet_all9_external_validation_2026-09-20.json"


def eval_tf():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), normalize,
    ])


def verify_label_mapping():
    import csv
    counts = {}
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cls, lbl = row["disease_class"], int(row["label"])
            counts.setdefault(cls, set()).add(lbl)
    print("Verified SkinDisNet label mapping (disease_class -> label values present):", flush=True)
    mapping_ok = True
    for cls, lbls in sorted(counts.items()):
        print(f"  {cls}: {lbls}", flush=True)
        if len(lbls) != 1:
            mapping_ok = False
    if not mapping_ok:
        raise RuntimeError("SkinDisNet manifest has inconsistent label mapping within a "
                            "disease_class -- aborting rather than proceeding on an "
                            "unverified mapping.")
    return {cls: list(lbls)[0] for cls, lbls in counts.items()}


@torch.no_grad()
def run_inference_with_class(model, loader, is_resnet18):
    model.eval()
    all_labels, all_probs, all_cls = [], [], []
    for imgs, labels, cls in loader:
        if is_resnet18:
            probs = torch.softmax(model(imgs), dim=1)[:, 1].numpy()
        else:
            probs = torch.sigmoid(model(imgs)).squeeze(-1).numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist() if torch.is_tensor(labels) else list(labels))
        all_cls.extend(cls)
    return np.array(all_labels), np.array(all_probs), np.array(all_cls)


def per_class_recall(labels, probs, classes, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    out = {}
    for cls_name in sorted(set(classes)):
        mask = classes == cls_name
        if labels[mask][0] != 1:
            continue
        out[cls_name] = {"n": int(mask.sum()), "recall": float((preds[mask] == 1).mean())}
    return out


def confusion(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def load_fresh(model_name):
    return build_resnet18_baseline() if model_name == "resnet18" else build_candidate(model_name)


def main():
    label_map = verify_label_mapping()

    ds = ExternalDataset(MANIFEST, eval_tf())
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                         collate_fn=lambda batch: (
                             torch.stack([b[0] for b in batch]),
                             torch.tensor([b[1] for b in batch]),
                             [b[2] for b in batch],
                         ))
    print(f"\nSkinDisNet external eval set: n={len(ds)}", flush=True)

    train_ds = CuratedDataset(TRAIN_MANIFEST, make_eval_transform())
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    print(f"Quantized backend in use: {QUANT_BACKEND}", flush=True)

    out = {"label_mapping_verified": label_map,
           "x86_fbgemm_int8": "INFEASIBLE -- this machine only supports the qnnpack quantized engine",
           "per_model": {}}
    if OUT_JSON.exists():
        prev = json.load(open(OUT_JSON))
        out["per_model"] = prev.get("per_model", {})
        print(f"Resuming -- already done: {list(out['per_model'].keys())}", flush=True)

    for model_name in ALL_MODELS:
        if model_name in out["per_model"]:
            print(f"\n=== {model_name}: already done, skipping ===", flush=True)
            continue
        print(f"\n=== {model_name} ===", flush=True)
        is_resnet18 = model_name == "resnet18"

        try:
            fp32 = load_fresh(model_name)
            labels, probs, classes = run_inference_with_class(fp32, loader, is_resnet18)
            fp32_metrics = point_metrics(labels, probs)
            fp32_ci = bootstrap_ci(labels, probs)
            fp32_sub = per_class_recall(labels, probs, classes)
            fp32_cm = confusion(labels, probs)
            print(f"  FP32:    acc={fp32_metrics['accuracy']:.4f} sens={fp32_metrics['sensitivity_recall']:.4f} "
                  f"spec={fp32_metrics['specificity']:.4f} f1={fp32_metrics['f1']:.4f} "
                  f"auroc={fp32_metrics['auroc']:.4f}", flush=True)

            int8_model = quantize_static(load_fresh(model_name), train_loader)
            labels_q, probs_q, classes_q = run_inference_with_class(int8_model, loader, is_resnet18)
            int8_metrics = point_metrics(labels_q, probs_q)
            int8_ci = bootstrap_ci(labels_q, probs_q)
            int8_sub = per_class_recall(labels_q, probs_q, classes_q)
            int8_cm = confusion(labels_q, probs_q)
            collapsed = int8_metrics["f1"] < 0.05
            degraded = (not collapsed) and (int8_metrics["sensitivity_recall"] < 0.20
                                             or int8_metrics["specificity"] < 0.20)
            verdict = "collapsed" if collapsed else ("severely_degraded" if degraded else "robust")
            print(f"  qnnpack: acc={int8_metrics['accuracy']:.4f} sens={int8_metrics['sensitivity_recall']:.4f} "
                  f"spec={int8_metrics['specificity']:.4f} f1={int8_metrics['f1']:.4f} "
                  f"auroc={int8_metrics['auroc']:.4f} -- {verdict.upper()}", flush=True)

            out["per_model"][model_name] = {
                "status": "ok",
                "fp32": {"point": fp32_metrics, "bootstrap_ci": fp32_ci, "subclass_recall": fp32_sub,
                         "confusion_matrix": fp32_cm},
                "qnnpack_int8": {"point": int8_metrics, "bootstrap_ci": int8_ci, "subclass_recall": int8_sub,
                                  "confusion_matrix": int8_cm, "verdict": verdict},
            }
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            out["per_model"][model_name] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}

        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=2)

    print(f"\nAll done. Results at {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
