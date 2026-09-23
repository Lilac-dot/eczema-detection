"""
Zero-shot external validation of the SELECTED image-channel architecture
(ShuffleNetV2-1.0x, honors-paper-report.docx Section 5.6) against SkinDisNet, compared
directly to the ResNet18 baseline's already-known zero-shot number
(AUC 0.4865 on the full uncleaned manifest, docs/cross_dataset_matrix_2026-09-15.md).

This is the report's own named next step (Section 11, Future Work): "Repeat the
cross-dataset external-validation audit already performed for the ResNet18 baseline ...
on ShuffleNetV2-1.0x specifically, before treating its internal test-set performance as
evidence of real-world generalization." The existing 0.6512 AUC / 0.4865 AUC numbers on
record are for ResNet18 (either fine-tuned or zero-shot) -- neither has ever been measured
for ShuffleNetV2-1.0x.

Runs BOTH models on the same cleaned manifest (manifest_skindisnet_clean.csv, built by
build_skindisnet_clean_manifest.py -- dedup + low-skin-content filtering) so this result
is not only new for ShuffleNetV2-1.0x, it is also a slightly cleaner re-measurement of the
ResNet18 zero-shot number than the one on record (which used the unfiltered manifest).
Both numbers are reported so neither is presented as a silent revision of the other.

Zero-shot only: neither model is fine-tuned on any SkinDisNet data here -- this is not the
same experiment as the multisource fine-tuning work (cross_dataset_generalization_status).
"""
import datetime
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from paths import ROOT, SKINDISNET_DIR, MODELS_DIR
from train_transfer_cnn import build_model, IMG_SIZE, DEVICE
from eval_external_common import ExternalDataset, paired_bootstrap_test

MANIFEST = SKINDISNET_DIR / "manifest_skindisnet_clean.csv"
RESNET18_PATH = MODELS_DIR / "curated_resnet18_balanced.pt"
SHUFFLENET_CKPT = ROOT / "experiments" / "shufflenet_v2_x1_0" / "checkpoints" / "best_overall.pt"
TODAY = datetime.date.today().isoformat()
OUT_JSON = ROOT / f"skindisnet_zeroshot_shufflenetv2_vs_resnet18_{TODAY}.json"


def eval_tf():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])


def load_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(RESNET18_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()


def load_shufflenet():
    model = build_model("shufflenet_v2_x1_0")
    model.load_state_dict(torch.load(SHUFFLENET_CKPT, map_location=DEVICE))
    return model.to(DEVICE).eval()


def run_inference(model, loader, is_two_class_softmax):
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels, _cls in loader:
            imgs = imgs.to(DEVICE)
            if is_two_class_softmax:
                probs = torch.softmax(model(imgs), dim=1)[:, 1].cpu().numpy()
            else:
                probs = torch.sigmoid(model(imgs)).cpu().numpy().ravel()
            all_labels.extend(labels.tolist() if torch.is_tensor(labels) else list(labels))
            all_probs.extend(probs.tolist())
    return np.array(all_labels), np.array(all_probs)


def metrics_from_probs(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "accuracy": float((y_pred == y_true).mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def bootstrap_ci(y_true, y_prob, metric_fn, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    vals = []
    for _ in range(n):
        idx = rng.randint(0, n_samples, n_samples)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        vals.append(metric_fn(yt, y_prob[idx]))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    print(f"Manifest: {MANIFEST}")
    ds = ExternalDataset(MANIFEST, eval_tf())
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                         collate_fn=lambda batch: (
                             torch.stack([b[0] for b in batch]),
                             torch.tensor([b[1] for b in batch]),
                             [b[2] for b in batch],
                         ))
    print(f"n = {len(ds)} images (clean manifest)")

    print("\n--- ResNet18 baseline (zero-shot, re-measured on cleaned manifest) ---")
    resnet = load_resnet18()
    y_true, resnet_probs = run_inference(resnet, loader, is_two_class_softmax=True)
    resnet_metrics = metrics_from_probs(y_true, resnet_probs)
    resnet_auc_ci = bootstrap_ci(y_true, resnet_probs, roc_auc_score)
    print(json.dumps(resnet_metrics, indent=2), "\nAUC 95% CI:", resnet_auc_ci)

    print("\n--- ShuffleNetV2-1.0x (selected architecture, zero-shot, NEW) ---")
    shuffle = load_shufflenet()
    _, shuffle_probs = run_inference(shuffle, loader, is_two_class_softmax=False)
    shuffle_metrics = metrics_from_probs(y_true, shuffle_probs)
    shuffle_auc_ci = bootstrap_ci(y_true, shuffle_probs, roc_auc_score)
    print(json.dumps(shuffle_metrics, indent=2), "\nAUC 95% CI:", shuffle_auc_ci)

    print("\n--- Paired significance (ShuffleNetV2-1.0x - ResNet18), same images ---")
    sig = paired_bootstrap_test(y_true, shuffle_probs, resnet_probs)
    print(json.dumps(sig, indent=2))

    out = {
        "date": TODAY,
        "manifest": str(MANIFEST),
        "n": len(ds),
        "note": "zero-shot only, neither model fine-tuned on SkinDisNet",
        "resnet18_baseline": {"metrics": resnet_metrics, "auc_ci": resnet_auc_ci},
        "shufflenet_v2_1_0x_selected": {"metrics": shuffle_metrics, "auc_ci": shuffle_auc_ci},
        "paired_significance_shufflenet_minus_resnet18": sig,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
