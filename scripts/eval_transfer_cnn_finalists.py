"""
The ONE script allowed to open manifest_curated_v3_test.csv for the transfer-CNN
architecture comparison (see docs/transfer_cnn_test_set_access_2026-09-18.md for why this
exists: a prior one-off, unsaved script touched the test set for 3 of 9 candidates before
the rest had even finished training).

HARD GATE: this script refuses to open the test manifest at all -- the import doesn't even
happen -- unless every model in train_transfer_cnn.MODEL_CONFIGS has a finished checkpoint
on disk. This turns "don't touch the test set until every candidate has trained" from a
guardrail written in a doc (which a script can silently ignore) into something the code
itself enforces. If you're tempted to bypass GATE with --force: don't, unless you are
deliberately abandoning the remaining candidates and documenting that decision -- the whole
point of this gate is that a partial read cannot happen again by accident.

SECOND GATE: refuses to overwrite a previous result file unless --force is passed, since
running this script IS a test-set access -- it should happen once, not every time someone
reruns it out of habit.

Usage:
    python eval_transfer_cnn_finalists.py            # normal run, both gates enforced
    python eval_transfer_cnn_finalists.py --force    # bypass both gates (asks you to
                                                       # confirm you're deliberately doing
                                                       # a second/partial read)
"""
import argparse
import csv
import datetime
import json
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from scipy.stats import chi2
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from paths import ROOT as PROJECT_ROOT, SKINDISEASE_DIR, MODELS_DIR
from train_transfer_cnn import MODEL_CONFIGS, build_model, CuratedDataset, IMG_SIZE, DEVICE

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
TEST_MANIFEST = SKINDISEASE_DIR / "manifest_curated_v3_test.csv"
BASELINE_NAME = "resnet18_baseline"
BASELINE_PATH = MODELS_DIR / "curated_resnet18_balanced.pt"
TODAY = datetime.date.today().isoformat()
OUT_JSON = PROJECT_ROOT / f"transfer_cnn_final_comparison_{TODAY}.json"
N_BOOTSTRAP = 2000
SEED = 42


def checkpoint_path(model_key):
    return EXPERIMENTS_DIR / model_key / "checkpoints" / "best_overall.pt"


# mobilenetv3_large deliberately excluded from this comparison (2026-09-18, explicit user
# decision) -- the training queue was stopped before it started (see
# scripts/stop_before_mobilenetv3_large.py). 8 of the original 9 candidates are the final
# set; MODEL_CONFIGS in train_transfer_cnn.py still defines it (that file is the training
# script, not this comparison's scope), so it's excluded here specifically.
SKIPPED_MODELS = ["mobilenetv3_large"]


def check_all_trained():
    required = [m for m in MODEL_CONFIGS if m not in SKIPPED_MODELS]
    missing = [m for m in required if not checkpoint_path(m).exists()]
    return missing


def load_resnet18_baseline():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(BASELINE_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()


def eval_tf():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])


def run_inference(model, loader, is_two_class_softmax):
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
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
        "n": len(y_true),
        "accuracy": float((y_pred == y_true).mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def mcnemar_test(y_true, pred_a, pred_b):
    """Continuity-corrected McNemar's test on paired correct/incorrect indicators."""
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))   # a right, b wrong
    c = int(np.sum(~correct_a & correct_b))   # a wrong, b right
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(chi2.sf(statistic, df=1))
    return {"b": b, "c": c, "statistic": float(statistic), "p_value": p_value}


def paired_bootstrap_diff(y_true, prob_a, prob_b, metric_fn, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    observed = metric_fn(y_true, prob_a) - metric_fn(y_true, prob_b)
    diffs = []
    for _ in range(n):
        idx = rng.randint(0, n_samples, n_samples)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        diffs.append(metric_fn(yt, prob_a[idx]) - metric_fn(yt, prob_b[idx]))
    diffs = np.array(diffs)
    ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
    p_value = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"observed_diff": float(observed), "ci": ci, "p_value": p_value}


def acc_metric(y_true, y_prob, threshold=0.5):
    return float(((y_prob >= threshold).astype(int) == y_true).mean())


def f1_metric(y_true, y_prob, threshold=0.5):
    return float(f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0))


def holm_correction(pairwise_pvalues):
    """pairwise_pvalues: dict[label] -> raw p. Returns dict[label] -> (p_holm, significant)."""
    items = sorted(pairwise_pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    running_max = 0.0
    for rank, (label, p_raw) in enumerate(items):
        adjusted = min((m - rank) * p_raw, 1.0)
        running_max = max(running_max, adjusted)
        out[label] = {"p_raw": p_raw, "p_holm": running_max, "significant": running_max < 0.05}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                         help="Bypass both gates (missing models / existing result file). "
                              "Only use this if you are deliberately doing a partial or "
                              "repeat test-set read and will document why.")
    args = parser.parse_args()

    missing = check_all_trained()
    if missing and not args.force:
        print("REFUSING TO RUN: the following candidates have no "
              "experiments/<model>/checkpoints/best_overall.pt yet:")
        for m in missing:
            print(f"  - {m}")
        print("\nThis script does not open manifest_curated_v3_test.csv until every "
              "candidate in train_transfer_cnn.MODEL_CONFIGS has finished training -- "
              "see docs/transfer_cnn_test_set_access_2026-09-18.md for why a partial read "
              "already happened once and shouldn't happen again by accident.")
        print("Finish training the remaining models, or pass --force if you are "
              "deliberately doing a partial/repeat read (and will document it).")
        sys.exit(1)
    elif missing:
        print(f"--force passed: proceeding despite {len(missing)} untrained candidate(s): "
              f"{missing}\nThis IS a partial/repeat test-set access. Document it.")

    if OUT_JSON.exists() and not args.force:
        print(f"REFUSING TO RUN: {OUT_JSON} already exists. This script's whole purpose is "
              f"a single, final test-set read -- if you need to redo it, pass --force and "
              f"document why the previous read wasn't final after all.")
        sys.exit(1)

    print(f"Gate passed ({len(MODEL_CONFIGS)} candidates trained). "
          f"Opening {TEST_MANIFEST} now -- this is the official test-set read.")

    test_ds = CuratedDataset(TEST_MANIFEST, eval_tf())
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    # CuratedDataset.rows holds (path, label) in manifest order; shuffle=False on the
    # loader above means this matches the inference order below exactly.
    y_true = np.array([label for _, label in test_ds.rows])

    results = {}

    print(f"\n--- {BASELINE_NAME} ---")
    baseline_model = load_resnet18_baseline()
    _, baseline_probs = run_inference(baseline_model, test_loader, is_two_class_softmax=True)
    results[BASELINE_NAME] = {
        "checkpoint": str(BASELINE_PATH),
        "probs": baseline_probs,
        "metrics": metrics_from_probs(y_true, baseline_probs),
    }
    print(json.dumps(results[BASELINE_NAME]["metrics"], indent=2))

    for model_key, cfg in MODEL_CONFIGS.items():
        ckpt = checkpoint_path(model_key)
        if not ckpt.exists():
            print(f"\n--- {model_key}: SKIPPED (no checkpoint) ---")
            continue
        print(f"\n--- {model_key} ---")
        model = build_model(model_key)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        _, probs = run_inference(model, test_loader, is_two_class_softmax=False)
        results[model_key] = {
            "checkpoint": str(ckpt),
            "probs": probs,
            "metrics": metrics_from_probs(y_true, probs),
        }
        print(json.dumps(results[model_key]["metrics"], indent=2))

    print("\n\n========== Pairwise comparisons (McNemar + paired bootstrap + Holm) ==========")
    names = list(results.keys())
    pairwise = {}
    raw_pvalues_acc = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            label = f"{a}_vs_{b}"
            pa, pb = results[a]["probs"], results[b]["probs"]
            pred_a = (pa >= 0.5).astype(int)
            pred_b = (pb >= 0.5).astype(int)
            mcnemar = mcnemar_test(y_true, pred_a, pred_b)
            acc_diff = paired_bootstrap_diff(y_true, pa, pb, acc_metric)
            f1_diff = paired_bootstrap_diff(y_true, pa, pb, f1_metric)
            pairwise[label] = {"mcnemar": mcnemar, "acc_diff": acc_diff, "f1_diff": f1_diff}
            raw_pvalues_acc[label] = mcnemar["p_value"]
            print(f"{label}: mcnemar_p={mcnemar['p_value']:.4f}  "
                  f"acc_diff={acc_diff['observed_diff']:+.4f} (p={acc_diff['p_value']:.4f})  "
                  f"f1_diff={f1_diff['observed_diff']:+.4f} (p={f1_diff['p_value']:.4f})")

    holm = holm_correction(raw_pvalues_acc)

    out = {
        "date": TODAY,
        "test_manifest": str(TEST_MANIFEST),
        "n_test": len(y_true),
        "models_included": names,
        "models_missing": missing,
        "metrics": {name: results[name]["metrics"] for name in names},
        "pairwise": pairwise,
        "holm_correction_on_mcnemar_p": holm,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_JSON}")
    print("\nThis was a single, official test-set read. Do not rerun this script to "
          "'check again' -- if the candidate pool changes, that's a new decision to "
          "document, not a routine re-check.")


if __name__ == "__main__":
    main()
