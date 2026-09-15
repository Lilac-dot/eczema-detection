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

from paths import MODELS_DIR

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
        img = Image.open(path).convert("RGB")
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


def run_eval(manifest_path, dataset_label, model_path=None, eval_tf=None):
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

    return dict(n=len(y_true), acc=acc, auc=auc, auc_ci=(auc_lo, auc_hi),
                precision=precision, recall=recall, f1=f1, f1_ci=(f1_lo, f1_hi),
                tp=tp, tn=tn, fp=fp, fn=fn,
                confused_with_eczema=dict(confused_with_eczema), missed_eczema=dict(missed_eczema))
