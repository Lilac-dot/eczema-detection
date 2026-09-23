"""
Corruption-robustness benchmark for the already-selected Stage B image model
(ShuffleNetV2-1.0x, experiments/shufflenet_v2_x1_0/checkpoints/best_overall.pt).

WHY THIS EXISTS: literature review (2026-09-19) found that "standardized image
acquisition improves downstream model reliability" has already been demonstrated
for skin cancer via software gating (DermAI, 2026), and that a reusable synthetic
corruption-benchmark methodology exists for dermatology broadly (ISIC-C/Dermnet-C,
Hu et al. 2024, arXiv 2405.11289) but has never been applied to eczema/AD. This
script applies that corruption-benchmark paradigm to this project's own eczema
classifier, as the software-only half of the "does controlling acquisition
conditions change how an AD image model degrades" research question -- the
half that needs no new hardware and no new data download to start answering.

TEST-SET ISOLATION: this script only ever opens manifest_curated_v3_val.csv.
The official test-set read for the architecture comparison already happened
(results/transfer_cnn_final_comparison_2026-09-18.json) -- this is a new,
exploratory sweep across corruption types and severities, not a finalized
comparison, so it belongs on validation, exactly like the edge-deployment
pruning/quantization sweep (papers/edge-ai-lightweight-deployment/).

Corruption types (uint8 PIL-image domain, applied BEFORE resize/normalize, at
severities 1-5 loosely following the ImageNet-C severity-scaling convention --
severity 0 is the clean/uncorrupted baseline):
  - gaussian_blur      (defocus / camera-shake proxy)
  - brightness_down     (underexposure)
  - brightness_up        (overexposure)
  - contrast_down        (washed-out lighting)
  - gaussian_noise        (sensor noise, low-light/cheap-camera proxy)
  - jpeg_compression       (aggressive phone-app / messaging-app recompression)
  - color_shift_warm        (white-balance error)

For each of the 36 conditions (7 types x 5 severities + clean), reports
accuracy/precision/recall/F1/ROC-AUC/ECE/Brier on the full validation set, then
runs a confidence-based selective-classification (risk-coverage) analysis on
clean vs. two corruption severities to test whether softmax confidence remains
a usable "trust this prediction or not" signal as corruption gets worse.

Usage:
    python robustness_corruption_benchmark.py [--model shufflenet_v2_x1_0]
"""
import argparse
import io
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from paths import ROOT as PROJECT_ROOT, SKINDISEASE_DIR
from train_transfer_cnn import MODEL_CONFIGS, build_model, IMG_SIZE, DEVICE
from mem_guard import require_free_mb

VAL_MANIFEST = SKINDISEASE_DIR / "manifest_curated_v3_val.csv"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
DOCS_DIR = PROJECT_ROOT / "docs"
RESULTS_DIR = PROJECT_ROOT / "results"
BATCH_SIZE = 32
MIN_FREE_MB = 400

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TO_TENSOR_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    NORMALIZE,
])

# ---------------------------------------------------------------------------
# Corruption functions -- each takes a PIL RGB image + severity (1-5) and
# returns a corrupted PIL RGB image. Severity scales chosen to span
# "barely visible" (1) to "severely degraded, still plausible as a real bad
# phone photo" (5) -- not chosen to match any specific published scale exactly,
# since none of the reviewed benchmarks (ISIC-C/Dermnet-C) publish their exact
# per-severity parameters in a way this project could reuse directly.
# ---------------------------------------------------------------------------

def gaussian_blur(img, severity):
    radius = {1: 0.6, 2: 1.2, 3: 2.0, 4: 3.0, 5: 4.5}[severity]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def brightness_down(img, severity):
    factor = {1: 0.85, 2: 0.70, 3: 0.55, 4: 0.40, 5: 0.25}[severity]
    return ImageEnhance.Brightness(img).enhance(factor)


def brightness_up(img, severity):
    factor = {1: 1.15, 2: 1.30, 3: 1.50, 4: 1.75, 5: 2.00}[severity]
    return ImageEnhance.Brightness(img).enhance(factor)


def contrast_down(img, severity):
    factor = {1: 0.85, 2: 0.70, 3: 0.55, 4: 0.40, 5: 0.25}[severity]
    return ImageEnhance.Contrast(img).enhance(factor)


def gaussian_noise(img, severity):
    sigma = {1: 5, 2: 10, 3: 20, 4: 35, 5: 50}[severity]
    arr = np.asarray(img).astype(np.float32)
    noise = np.random.RandomState(0).normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def jpeg_compression(img, severity):
    quality = {1: 60, 2: 40, 3: 25, 4: 15, 5: 5}[severity]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def color_shift_warm(img, severity):
    factor = {1: 1.10, 2: 1.20, 3: 1.35, 4: 1.50, 5: 1.70}[severity]
    arr = np.asarray(img).astype(np.float32)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * factor, 0, 255)   # boost red
    arr[:, :, 2] = np.clip(arr[:, :, 2] / factor, 0, 255)   # cut blue
    return Image.fromarray(arr.astype(np.uint8))


CORRUPTIONS = {
    "gaussian_blur": gaussian_blur,
    "brightness_down": brightness_down,
    "brightness_up": brightness_up,
    "contrast_down": contrast_down,
    "gaussian_noise": gaussian_noise,
    "jpeg_compression": jpeg_compression,
    "color_shift_warm": color_shift_warm,
}
SEVERITIES = [1, 2, 3, 4, 5]


class CorruptedValDataset(torch.utils.data.Dataset):
    """Loads manifest_curated_v3_val.csv images fresh from disk and applies one
    corruption function at one severity (or none, if corruption_fn is None) before
    the standard resize/normalize eval transform -- corruption is applied in the
    original uint8 image domain, matching how a real degraded photo would look,
    not in normalized tensor space."""

    def __init__(self, manifest_path, corruption_fn=None, severity=None):
        self.rows = []
        import csv
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.rows.append((row["path"], int(row["label"])))
        self.corruption_fn = corruption_fn
        self.severity = severity

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label = self.rows[idx]
        img = Image.open(path).convert("RGB")
        if self.corruption_fn is not None:
            img = self.corruption_fn(img, self.severity)
        return TO_TENSOR_TF(img), label


def expected_calibration_error(y_true, y_prob, n_bins=15):
    """Standard binned ECE: |accuracy - confidence| within each confidence bin,
    weighted by bin population."""
    confidences = np.maximum(y_prob, 1 - y_prob)
    predictions = (y_prob >= 0.5).astype(int)
    correct = (predictions == y_true).astype(float)
    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)  # confidence is always in [0.5, 1]
    ece = 0.0
    n = len(y_true)
    bin_stats = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi) if hi < 1.0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        weight = mask.sum() / n
        ece += weight * abs(bin_acc - bin_conf)
        bin_stats.append({"lo": float(lo), "hi": float(hi), "n": int(mask.sum()),
                           "acc": float(bin_acc), "conf": float(bin_conf)})
    return float(ece), bin_stats


def brier_score(y_true, y_prob):
    return float(np.mean((y_prob - y_true) ** 2))


def run_inference(model, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_labels, all_probs = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            probs = torch.sigmoid(model(imgs)).cpu().numpy().ravel()
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())
    return np.array(all_labels), np.array(all_probs)


def metrics_from_probs(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    ece, bin_stats = expected_calibration_error(y_true, y_prob)
    return {
        "n": int(len(y_true)),
        "accuracy": float((y_pred == y_true).mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "ece": ece,
        "brier": brier_score(y_true, y_prob),
        "calibration_bins": bin_stats,
    }


def risk_coverage_curve(y_true, y_prob, coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)):
    """Selective classification: sort predictions by confidence descending, keep
    the top `coverage` fraction, report accuracy on the retained (most confident)
    subset. Tests whether confidence is still a usable 'trust this prediction'
    signal, not just whether the model is accurate overall."""
    confidences = np.maximum(y_prob, 1 - y_prob)
    order = np.argsort(-confidences)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]
    n = len(y_true)
    out = []
    for cov in coverages:
        k = max(1, int(round(n * cov)))
        y_pred_k = (y_prob_sorted[:k] >= 0.5).astype(int)
        acc_k = float((y_pred_k == y_true_sorted[:k]).mean())
        out.append({"coverage": cov, "n_retained": k, "accuracy": acc_k})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="shufflenet_v2_x1_0", choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    if not require_free_mb(MIN_FREE_MB, context="robustness benchmark startup"):
        raise SystemExit("Not enough free RAM to safely start this benchmark.")

    ckpt_path = EXPERIMENTS_DIR / args.model / "checkpoints" / "best_overall.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"No trained checkpoint at {ckpt_path}")

    print(f"Loading {args.model} from {ckpt_path}")
    model = build_model(args.model)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    conditions = [("clean", None, 0)]
    for name, fn in CORRUPTIONS.items():
        for sev in SEVERITIES:
            conditions.append((name, fn, sev))

    results = {}
    t_start = time.time()
    for name, fn, sev in conditions:
        key = "clean" if fn is None else f"{name}_sev{sev}"
        t0 = time.time()
        ds = CorruptedValDataset(VAL_MANIFEST, corruption_fn=fn, severity=sev)
        y_true, y_prob = run_inference(model, ds)
        m = metrics_from_probs(y_true, y_prob)
        dt = time.time() - t0
        print(f"[{dt:5.1f}s] {key:28s} acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
              f"auc={m['roc_auc']:.4f} ece={m['ece']:.4f}")
        results[key] = {
            "corruption": name, "severity": sev, "metrics": m,
            "y_true": y_true.tolist(), "y_prob": y_prob.tolist(),
        }
    total_time = time.time() - t_start
    print(f"\nTotal benchmark time: {total_time:.1f}s over {len(conditions)} conditions")

    # --- Selective-classification analysis on clean vs. moderate vs. severe ---
    rc_conditions = ["clean", "gaussian_blur_sev3", "gaussian_blur_sev5",
                      "brightness_down_sev3", "brightness_down_sev5"]
    risk_coverage = {}
    for key in rc_conditions:
        if key not in results:
            continue
        y_true = np.array(results[key]["y_true"])
        y_prob = np.array(results[key]["y_prob"])
        risk_coverage[key] = risk_coverage_curve(y_true, y_prob)

    # --- Save raw results (drop per-image labels/probs from the JSON body but keep
    # them for plotting in this run; a slimmer summary-only JSON is also written) ---
    RESULTS_DIR.mkdir(exist_ok=True)
    full_out = {
        "model": args.model,
        "checkpoint": str(ckpt_path),
        "val_manifest": str(VAL_MANIFEST),
        "n_val": int(len(results["clean"]["y_true"])),
        "total_time_s": round(total_time, 1),
        "conditions": {k: {"corruption": v["corruption"], "severity": v["severity"],
                            "metrics": v["metrics"]} for k, v in results.items()},
        "risk_coverage": risk_coverage,
    }
    out_path = RESULTS_DIR / f"robustness_corruption_results_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(full_out, f, indent=2)
    print(f"Saved summary results to {out_path}")

    # Separately save per-image probs (larger file) for anyone who wants to redo
    # the calibration/selective-classification analysis without rerunning inference.
    probs_out_path = RESULTS_DIR / f"robustness_corruption_probs_{args.model}.json"
    with open(probs_out_path, "w") as f:
        json.dump({k: {"y_true": v["y_true"], "y_prob": v["y_prob"]} for k, v in results.items()}, f)
    print(f"Saved per-image probabilities to {probs_out_path}")

    # --- Plot: accuracy vs. severity, one line per corruption type ---
    fig, ax = plt.subplots(figsize=(8, 6))
    clean_acc = results["clean"]["metrics"]["accuracy"]
    ax.axhline(clean_acc, color="black", linestyle=":", label="Clean baseline")
    for name in CORRUPTIONS:
        accs = [results[f"{name}_sev{s}"]["metrics"]["accuracy"] for s in SEVERITIES]
        ax.plot([0] + SEVERITIES, [clean_acc] + accs, marker="o", label=name)
    ax.set_xlabel("Corruption severity (0 = clean)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{args.model} -- accuracy under synthetic corruption (val set, n={full_out['n_val']})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS_DIR / f"robustness_accuracy_vs_severity_{args.model}_2026-09-19.png", dpi=150)
    plt.close(fig)

    # --- Plot: ECE vs. severity ---
    fig, ax = plt.subplots(figsize=(8, 6))
    clean_ece = results["clean"]["metrics"]["ece"]
    ax.axhline(clean_ece, color="black", linestyle=":", label="Clean baseline")
    for name in CORRUPTIONS:
        eces = [results[f"{name}_sev{s}"]["metrics"]["ece"] for s in SEVERITIES]
        ax.plot([0] + SEVERITIES, [clean_ece] + eces, marker="o", label=name)
    ax.set_xlabel("Corruption severity (0 = clean)")
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title(f"{args.model} -- calibration under synthetic corruption")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS_DIR / f"robustness_ece_vs_severity_{args.model}_2026-09-19.png", dpi=150)
    plt.close(fig)

    # --- Plot: risk-coverage curves ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for key, curve in risk_coverage.items():
        covs = [c["coverage"] for c in curve]
        accs = [c["accuracy"] for c in curve]
        ax.plot(covs, accs, marker="o", label=key)
    ax.set_xlabel("Coverage (fraction of predictions retained, most-confident first)")
    ax.set_ylabel("Accuracy on retained predictions")
    ax.set_title(f"{args.model} -- selective classification (risk-coverage)")
    ax.legend(fontsize=8)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(DOCS_DIR / f"robustness_risk_coverage_{args.model}_2026-09-19.png", dpi=150)
    plt.close(fig)

    print("\nPlots saved to docs/robustness_*_2026-09-19.png")
    print("\nDone. This never opened manifest_curated_v3_test.csv.")


if __name__ == "__main__":
    main()
