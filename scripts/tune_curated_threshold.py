"""
Try to improve F1 on the curated Eczema-vs-other-disease task WITHOUT retraining:

1. Instead of the CNN's default 50% cutoff, find the threshold (using the validation set)
   that maximizes F1, then apply that fixed threshold to the test set. Same trick already
   used for Stage A.
2. Do the same for the LightGBM model.
3. Try a simple ensemble: average the CNN's and LightGBM's predicted probabilities
   together, then tune a threshold for THAT combined score too. Worth trying because the
   two models seem to pick up on different signal (CNN sees raw image structure/texture,
   LightGBM sees hand-picked colour statistics) -- combining them can catch cases where
   one is right and the other is wrong.

Prints a comparison table; does not save any new model files.
"""
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import lightgbm as lgb

ROOT = Path(r"C:\Users\tishy\Documents\Honors\SkinDisease")
MODELS_DIR = Path(r"C:\Users\tishy\Documents\Honors\models")
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def cnn_probs(manifest_name):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    eval_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), normalize])
    ds = CuratedDataset(ROOT / f"manifest_curated_{manifest_name}.csv", eval_tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODELS_DIR / "curated_resnet18_v2.pt", map_location=DEVICE))
    model = model.to(DEVICE).eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            probs = F.softmax(model(imgs), dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def lgbm_probs(manifest_name):
    feat_rows = {}
    with open(ROOT / "color_features_curated.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            path = row[0]
            feats = np.array([float(v) for v in row[3:]], dtype=np.float64)
            feat_rows[path] = feats

    booster = lgb.Booster(model_file=str(MODELS_DIR / "curated_lightgbm.txt"))
    X, y = [], []
    with open(ROOT / f"manifest_curated_{manifest_name}.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            X.append(feat_rows[row["path"]])
            y.append(int(row["label"]))
    X = np.array(X)
    probs = booster.predict(X)
    return probs, np.array(y)


def best_threshold(probs, labels):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 197):
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def evaluate(probs, labels, threshold, name):
    preds = (probs >= threshold).astype(int)
    acc = (preds == labels).mean()
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(f"[{name}] threshold={threshold:.3f} n={len(labels)} acc={acc:.4f} "
          f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
          f"(TP={tp} FP={fp} FN={fn} TN={tn})")
    return dict(acc=acc, precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    print("Computing CNN probabilities...")
    cnn_val_p, val_y = cnn_probs("val")
    cnn_test_p, test_y = cnn_probs("test")

    print("Computing LightGBM probabilities...")
    lgbm_val_p, val_y2 = lgbm_probs("val")
    lgbm_test_p, test_y2 = lgbm_probs("test")
    assert (val_y == val_y2).all() and (test_y == test_y2).all(), "label order mismatch between manifests"

    ens_val_p = (cnn_val_p + lgbm_val_p) / 2
    ens_test_p = (cnn_test_p + lgbm_test_p) / 2

    print("\n=== Default 0.5 threshold (for reference) ===")
    evaluate(cnn_test_p, test_y, 0.5, "CNN @0.5")
    evaluate(lgbm_test_p, test_y, 0.5, "LightGBM @0.5")
    evaluate(ens_test_p, test_y, 0.5, "Ensemble @0.5")

    print("\n=== Threshold tuned on validation set, applied to test set ===")
    for name, val_p, test_p in [("CNN", cnn_val_p, cnn_test_p),
                                  ("LightGBM", lgbm_val_p, lgbm_test_p),
                                  ("Ensemble", ens_val_p, ens_test_p)]:
        t, val_f1 = best_threshold(val_p, val_y)
        print(f"\n{name}: best threshold on val = {t:.3f} (val F1={val_f1:.4f})")
        evaluate(test_p, test_y, t, f"{name} @tuned")


if __name__ == "__main__":
    main()
