"""Same idea as tune_curated_threshold.py, applied to the final 50/50-balanced dataset:
compute CNN and LightGBM probabilities on val/test, check a simple average ensemble."""
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
    ds = CuratedDataset(ROOT / f"manifest_curated_v3_{manifest_name}.csv", eval_tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODELS_DIR / "curated_resnet18_balanced.pt", map_location=DEVICE))
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
    with open(ROOT / "color_features_curated_v3.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            path = row[0]
            feats = np.array([float(v) for v in row[3:]], dtype=np.float64)
            feat_rows[path] = feats

    booster = lgb.Booster(model_file=str(MODELS_DIR / "curated_lightgbm_v3.txt"))
    X, y = [], []
    with open(ROOT / f"manifest_curated_v3_{manifest_name}.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            X.append(feat_rows[row["path"]])
            y.append(int(row["label"]))
    X = np.array(X)
    probs = booster.predict(X)
    return probs, np.array(y)


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


def main():
    print("Computing CNN probabilities...")
    cnn_test_p, test_y = cnn_probs("test")

    print("Computing LightGBM probabilities...")
    lgbm_test_p, test_y2 = lgbm_probs("test")
    assert (test_y == test_y2).all()

    ens_test_p = (cnn_test_p + lgbm_test_p) / 2

    print("\n=== Test set results (0.5 threshold) ===")
    evaluate(cnn_test_p, test_y, 0.5, "CNN")
    evaluate(lgbm_test_p, test_y, 0.5, "LightGBM")
    evaluate(ens_test_p, test_y, 0.5, "Ensemble")


if __name__ == "__main__":
    main()
