"""Evaluate the curated Eczema-vs-other-disease CNN on its held-out test split."""
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

ROOT = Path(r"C:\Users\tishy\Documents\Honors\SkinDisease")
MODEL_PATH = Path(r"C:\Users\tishy\Documents\Honors\models\curated_resnet18.pt")
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CuratedDataset(Dataset):
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


def main():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    test_ds = CuratedDataset(ROOT / "manifest_curated_test.csv", eval_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0,
                              collate_fn=lambda batch: (
                                  torch.stack([b[0] for b in batch]),
                                  torch.tensor([b[1] for b in batch]),
                                  [b[2] for b in batch],
                              ))

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    tp = tn = fp = fn = 0
    correct = 0
    total = 0
    from collections import Counter
    confused_with_eczema = Counter()  # true "other" class predicted as Eczema
    missed_eczema_as = Counter()      # true Eczema predicted as "other" -- but binary, so just count

    with torch.no_grad():
        for imgs, labels, classes in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            for p, l, cls in zip(preds.tolist(), labels.tolist(), classes):
                if p == 1 and l == 1: tp += 1
                elif p == 0 and l == 0: tn += 1
                elif p == 1 and l == 0:
                    fp += 1
                    confused_with_eczema[cls] += 1
                elif p == 0 and l == 1:
                    fn += 1

    acc = correct / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"Test set size: {total}")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Confusion matrix (rows=true, cols=pred) [Other, Eczema]:")
    print(f"  Other:  TN={tn}  FP={fp}")
    print(f"  Eczema: FN={fn}  TP={tp}")
    print(f"Precision (Eczema): {precision:.4f}")
    print(f"Recall (Eczema):    {recall:.4f}")
    print(f"F1 (Eczema):        {f1:.4f}")
    print(f"\nWhich 'other' diseases got misclassified as Eczema (false positives), by class:")
    for cls, n in confused_with_eczema.most_common():
        print(f"  {cls}: {n}")


if __name__ == "__main__":
    main()
