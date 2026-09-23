# -*- coding: utf-8 -*-
"""Pruning-knee significance test (McNemar), requested 2026-09-23.

The original sweep (edge_simulation_all_architectures.py, 2026-09-19) stored only aggregate
metrics, so its knee used a fixed ">5 pt drop" heuristic. This script reruns the IDENTICAL
one-shot global L1 unstructured pruning sweep (same checkpoints, same 0-90% sparsities, same
496-image validation split, same eval transform and 0.5 threshold) but keeps every image's
prediction, then tests each sparsity against the unpruned model with the exact McNemar test
(McNemar 1947; the paired test recommended for comparing two classifiers on one test set by
Dietterich 1998), Holm-corrected across the 9 sparsities of each architecture (Holm 1979).

Knee (significance version) = the lowest sparsity whose accuracy is LOWER than the dense
model's with Holm-adjusted McNemar p < 0.05.

Sanity check: every recomputed accuracy is compared with the 2026-09-19 JSON; differences
are reported (the original ran on x86/Windows, this on ARM/macOS, so tiny floating-point
differences in magnitude ties or near-0.5 probabilities are possible).

Validation split only; the test manifest is never opened.
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.utils.prune as prune
from scipy.stats import binomtest
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edge_simulation_all_architectures as sim  # noqa: E402 (no side effects beyond seeding)
from paths import ROOT  # noqa: E402

OUT_JSON = ROOT / "papers" / "edge-ai-lightweight-deployment" / "pruning_knee_mcnemar_2026-09-23.json"
ORIG_JSON = ROOT / "papers" / "edge-ai-lightweight-deployment" / "edge_simulation_all_architectures_2026-09-19.json"
ALPHA = 0.05


@torch.no_grad()
def predict(model, loader, is_resnet18):
    model.eval()
    preds, labels = [], []
    for imgs, y in loader:
        out = model(imgs)
        p = torch.softmax(out, dim=1)[:, 1] if is_resnet18 else torch.sigmoid(out).ravel()
        preds.extend((p.numpy() >= 0.5).astype(int).tolist())
        labels.extend(y.numpy().tolist())
    return np.array(preds), np.array(labels)


def mcnemar_exact(correct_a, correct_b):
    """b = A right & B wrong, c = A wrong & B right; exact two-sided binomial test on b vs c."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5, alternative="two-sided").pvalue
    return b, c, float(p)


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


def main():
    import os
    os.chdir(ROOT)  # manifest paths are relative to the project root
    orig = json.load(open(ORIG_JSON))
    loader = DataLoader(sim.CuratedDataset(sim.VAL_MANIFEST, sim.make_eval_transform()),
                        batch_size=32, shuffle=False, num_workers=0)
    out = {"alpha": ALPHA, "test": "exact McNemar (two-sided), Holm-corrected over sparsities 0.1-0.9",
           "split": "validation (496 images)", "per_model": {}}
    for name in sim.ALL_MODELS:
        is_r = name == "resnet18"
        rows, correct = [], {}
        for s in sim.SPARSITIES:
            model = sim.load_model(name)
            params = sim.prunable_modules(model)
            if s > 0:
                prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=s)
                for module, pname in params:
                    prune.remove(module, pname)
            preds, labels = predict(model, loader, is_r)
            correct[s] = preds == labels
            acc = float(correct[s].mean())
            acc_orig = [r for r in orig[name]["pruning_sweep"] if r["sparsity_target"] == s][0]["val_metrics"]["accuracy"]
            rows.append({"sparsity": s, "accuracy": acc, "accuracy_2026_09_19": acc_orig,
                         "diff_vs_original_pt": round((acc - acc_orig) * 100, 3),
                         "predictions": preds.tolist()})
            del model
            gc.collect()
        dense = correct[0.0]
        tests = [mcnemar_exact(dense, correct[s]) for s in sim.SPARSITIES[1:]]
        adj = holm([t[2] for t in tests])
        for r, (b, c, p), pa in zip(rows[1:], tests, adj):
            r.update(dense_right_pruned_wrong=b, dense_wrong_pruned_right=c, p_raw=p, p_holm=pa,
                     significant_drop=bool(pa < ALPHA and r["accuracy"] < rows[0]["accuracy"]))
        knee = next((r["sparsity"] for r in rows[1:] if r["significant_drop"]), None)
        out["per_model"][name] = {"labels": labels.tolist(), "knee_mcnemar": knee, "sweep": rows}
        print(f"{name:20s} knee_mcnemar={knee} | " + " ".join(
            f"{r['sparsity']:.1f}:{r['accuracy'] * 100:.1f}"
            + (f"(p={r['p_holm']:.3f})" if "p_holm" in r else "")
            + ("" if abs(r['diff_vs_original_pt']) < 1e-6 else f"[Δ{r['diff_vs_original_pt']:+.1f}]")
            for r in rows), flush=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    print("Saved:", OUT_JSON)


if __name__ == "__main__":
    main()
