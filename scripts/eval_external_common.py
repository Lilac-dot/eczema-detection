"""
Shared evaluation logic for testing the deployed Stage B model (curated_resnet18_balanced.pt,
trained only on the DermNet-style curated archive) against an external, independently-sourced
manifest -- unmodified, no fine-tuning. Called by eval_external_scin.py and
eval_external_skindisnet.py.

Extends the pattern in eval_curated_cnn_balanced.py with softmax probabilities (for AUC) and
a bootstrap 95% CI on AUC/F1, since a single point estimate on a few hundred external images
needs an honest uncertainty band, same as the report already does for Stage A (Section 4.6).
"""
import csv
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import roc_auc_score

from paths import MODELS_DIR, ROOT

MODEL_PATH = MODELS_DIR / "curated_resnet18_balanced.pt"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BOOTSTRAP = 1000
SEED = 42


class ExternalDataset(Dataset):
    def __init__(self, manifest_path, transform):
        self.rows = []
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.rows.append((row["path"], int(row["label"]), row["disease_class"]))
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label, cls = self.rows[idx]
        from pathlib import Path
        full_path = path if Path(path).is_absolute() else ROOT / path
        img = Image.open(full_path).convert("RGB")
        return self.transform(img), label, cls


def load_model(model_path=None):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path or MODEL_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()


def bootstrap_ci(y_true, y_prob, y_pred, metric_fn, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    vals = []
    for _ in range(n):
        idx = rng.randint(0, n_samples, n_samples)
        yt, yp, ypred = y_true[idx], y_prob[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue  # AUC undefined for a resample with only one class present
        vals.append(metric_fn(yt, yp, ypred))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def auc_metric(yt, yp, ypred):
    return roc_auc_score(yt, yp)


def f1_metric(yt, yp, ypred):
    tp = ((ypred == 1) & (yt == 1)).sum()
    fp = ((ypred == 1) & (yt == 0)).sum()
    fn = ((ypred == 0) & (yt == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def default_eval_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])


def run_eval(manifest_path, dataset_label, model_path=None, eval_tf=None, return_raw=False):
    eval_tf = eval_tf or default_eval_transform()
    ds = ExternalDataset(manifest_path, eval_tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                         collate_fn=lambda batch: (
                             torch.stack([b[0] for b in batch]),
                             torch.tensor([b[1] for b in batch]),
                             [b[2] for b in batch],
                         ))

    model = load_model(model_path)

    all_probs, all_labels, all_preds, all_classes = [], [], [], []
    with torch.no_grad():
        for imgs, labels, classes in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_classes.extend(classes)

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    auc = roc_auc_score(y_true, y_prob)

    auc_lo, auc_hi = bootstrap_ci(y_true, y_prob, y_pred, auc_metric)
    f1_lo, f1_hi = bootstrap_ci(y_true, y_prob, y_pred, f1_metric)

    confused_with_eczema = Counter()
    missed_eczema = Counter()
    for p, l, cls in zip(all_preds, all_labels, all_classes):
        if p == 1 and l == 0:
            confused_with_eczema[cls] += 1
        elif p == 0 and l == 1:
            missed_eczema[cls] += 1

    print(f"\n=== {dataset_label} ===")
    print(f"n={len(y_true)} (Eczema={int((y_true == 1).sum())}, Other={int((y_true == 0).sum())})")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}  (95% CI bootstrap: {auc_lo:.4f}-{auc_hi:.4f})")
    print(f"Precision (Eczema): {precision:.4f}")
    print(f"Recall (Eczema): {recall:.4f}")
    print(f"F1 (Eczema): {f1:.4f}  (95% CI bootstrap: {f1_lo:.4f}-{f1_hi:.4f})")
    print(f"Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
    if confused_with_eczema:
        print("Other-class images misclassified as Eczema:")
        for cls, n in confused_with_eczema.most_common():
            print(f"  {cls}: {n}")
    if missed_eczema:
        print("Eczema images missed (predicted Other):")
        for cls, n in missed_eczema.most_common():
            print(f"  {cls}: {n}")

    result = dict(n=len(y_true), acc=acc, auc=auc, auc_ci=(auc_lo, auc_hi),
                  precision=precision, recall=recall, f1=f1, f1_ci=(f1_lo, f1_hi),
                  tp=tp, tn=tn, fp=fp, fn=fn,
                  confused_with_eczema=dict(confused_with_eczema), missed_eczema=dict(missed_eczema))
    if return_raw:
        # Opt-in only -- existing callers are unaffected (same dict keys as before, plus these).
        # Safe to pair against another run_eval() call on the SAME manifest_path: the
        # DataLoader uses shuffle=False, so row order always matches the manifest file's
        # own row order and two independent calls line up index-for-index.
        result["y_true"] = y_true.tolist()
        result["y_prob"] = y_prob.tolist()
    return result


def paired_bootstrap_test(y_true, y_prob_a, y_prob_b, n=1000, seed=42):
    """Paired bootstrap significance test for AUC_a - AUC_b on the SAME test images.

    Unlike two independent bootstrap_ci() calls (one per model), this resamples the same
    image indices for both models on every draw -- required because the two models' scores
    on the same image are correlated (same underlying difficulty), so treating them as two
    independent samples would overstate the uncertainty of their difference. This is the
    standard paired-bootstrap construction for comparing two classifiers on one shared test
    set (e.g. Efron & Tibshirani 1993).

    y_true, y_prob_a, y_prob_b must be same-length arrays over the SAME images in the SAME
    order (e.g. two run_eval(..., return_raw=True) calls on the same manifest_path).

    Returns dict(observed_diff, ci=(lo, hi), p_value) for AUC_a - AUC_b. A 95% CI that
    excludes 0, or p < 0.05, indicates a statistically significant difference; report the
    CI and p-value together, not just a significant/not-significant verdict.
    """
    y_true = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)
    assert len(y_true) == len(y_prob_a) == len(y_prob_b), \
        "paired_bootstrap_test requires all three arrays to be the same length (same images, same order)"

    observed_diff = roc_auc_score(y_true, y_prob_a) - roc_auc_score(y_true, y_prob_b)

    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    diffs = []
    for _ in range(n):
        idx = rng.randint(0, n_samples, n_samples)  # SAME indices for both models this draw
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue  # AUC undefined for a resample with only one class present
        auc_a = roc_auc_score(yt, y_prob_a[idx])
        auc_b = roc_auc_score(yt, y_prob_b[idx])
        diffs.append(auc_a - auc_b)
    diffs = np.array(diffs)

    ci_lo, ci_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    # Two-sided bootstrap p-value: fraction of resampled diffs on the opposite side of 0
    # from the observed diff, doubled (standard construction), capped at 1.0.
    if observed_diff >= 0:
        p_one_sided = (diffs <= 0).mean()
    else:
        p_one_sided = (diffs >= 0).mean()
    p_value = min(1.0, 2 * p_one_sided)

    return dict(observed_diff=float(observed_diff), ci=(ci_lo, ci_hi), p_value=float(p_value),
                n_valid_draws=int(len(diffs)))
