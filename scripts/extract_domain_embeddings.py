"""Extract penultimate-layer (512-dim) ResNet18 embeddings for every image in the
internal curated dataset, SCIN, and SkinDisNet, using the DEPLOYED Stage B model
(models/curated_resnet18_balanced.pt) as a fixed feature extractor -- no retraining.

Used by scripts/domain_shift_analysis.py to quantify how separable these three
datasets are in the deployed model's own learned feature space (proxy A-distance),
as a formal explanation for the cross-dataset generalization gap documented in
docs/external_validation_2026-09-15.md.

Read-only with respect to existing project files: only reads the original manifests
(not any "_clean" variant another process may be creating concurrently) and writes a
new cache file, dataset/embeddings_internal_scin_skindisnet.npz.
"""
import csv
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image

from paths import ROOT, MODELS_DIR, SKINDISEASE_DIR, SCIN_DIR, SKINDISNET_DIR
from eval_external_common import default_eval_transform, load_model

OUT_PATH = ROOT / "dataset" / "embeddings_internal_scin_skindisnet.npz"
DEVICE = torch.device("cpu")

MANIFESTS = {
    "Internal": SKINDISEASE_DIR / "manifest_curated_v3.csv",
    "SCIN": SCIN_DIR / "manifest_scin.csv",
    "SkinDisNet": SKINDISNET_DIR / "manifest_skindisnet.csv",
}


class ManifestImageDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label, cls, source = self.rows[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, cls, source, path


def load_all_rows():
    all_rows = []
    for source, manifest_path in MANIFESTS.items():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                all_rows.append((row["path"], int(row["label"]), row["disease_class"], source))
    return all_rows


def build_feature_extractor():
    model = load_model(MODELS_DIR / "curated_resnet18_balanced.pt")
    # Everything up to (and including) avgpool, i.e. drop only the final fc layer --
    # this is the exact 512-dim representation the deployed model's classifier head
    # sees, so the domain-separability measure below is about what THIS model
    # actually learned to look at, not a generic/untrained ImageNet feature space.
    extractor = nn.Sequential(*list(model.children())[:-1])
    return extractor.to(DEVICE).eval()


def main():
    rows = load_all_rows()
    print(f"Total images: {len(rows)}")
    for source in MANIFESTS:
        n = sum(1 for r in rows if r[3] == source)
        print(f"  {source}: {n}")

    extractor = build_feature_extractor()
    tf = default_eval_transform()
    ds = ManifestImageDataset(rows, tf)
    loader = DataLoader(
        ds, batch_size=32, shuffle=False, num_workers=0,
        collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch]),
            torch.tensor([b[1] for b in batch]),
            [b[2] for b in batch],
            [b[3] for b in batch],
            [b[4] for b in batch],
        ),
    )

    all_embeddings, all_labels, all_classes, all_sources, all_paths = [], [], [], [], []
    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for imgs, labels, classes, sources, paths in loader:
            imgs = imgs.to(DEVICE)
            feats = extractor(imgs)  # (B, 512, 1, 1)
            feats = feats.flatten(1).cpu().numpy()
            all_embeddings.append(feats)
            all_labels.extend(labels.tolist())
            all_classes.extend(classes)
            all_sources.extend(sources)
            all_paths.extend(paths)
            n_done += len(paths)
            if n_done % 640 == 0 or n_done == len(rows):
                elapsed = time.time() - t0
                print(f"  {n_done}/{len(rows)} ({elapsed:.0f}s elapsed)")

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.array(all_labels)
    sources = np.array(all_sources)
    classes = np.array(all_classes)
    paths = np.array(all_paths)

    print(f"\nEmbeddings shape: {embeddings.shape}")
    np.savez_compressed(
        OUT_PATH,
        embeddings=embeddings, labels=labels, sources=sources,
        classes=classes, paths=paths,
    )
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
