"""Evaluate the trained Stage B model on the held-out test split."""
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

ROOT = Path(r"C:\Users\tishy\Documents\Honors\dataset")
MODEL_PATH = Path(r"C:\Users\tishy\Documents\Honors\models\stage_b_resnet18.pt")

CLASS_TO_IDX = {"Normal": 0, "Eczema": 1}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EczemaDataset(Dataset):
    def __init__(self, manifest_path, transform):
        self.rows = []
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.rows.append((row["path"], CLASS_TO_IDX[row["class"]]))
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label = self.rows[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def main():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    test_ds = EczemaDataset(ROOT / "manifest_test.csv", eval_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    tp = tn = fp = fn = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            for p, l in zip(preds.tolist(), labels.tolist()):
                if p == 1 and l == 1: tp += 1
                elif p == 0 and l == 0: tn += 1
                elif p == 1 and l == 0: fp += 1
                elif p == 0 and l == 1: fn += 1

    acc = correct / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"Test set size: {total}")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Confusion matrix (rows=true, cols=pred) [Normal, Eczema]:")
    print(f"  Normal: TN={tn}  FP={fp}")
    print(f"  Eczema: FN={fn}  TP={tp}")
    print(f"Precision (Eczema): {precision:.4f}")
    print(f"Recall (Eczema):    {recall:.4f}")
    print(f"F1 (Eczema):        {f1:.4f}")


if __name__ == "__main__":
    main()
