"""
Experiment 2b evaluation: compares the zero-shot baseline and the plain multi-source
fine-tune (Experiment 2) against the domain-weighted multi-source fine-tune
(curated_resnet18_multisource_weighted.pt), all on the SAME held-out test partitions
from split_external_manifest.py. Run after train_curated_cnn_multisource_weighted.py.
"""
import json

from paths import SKINDISEASE_DIR, SCIN_DIR, SKINDISNET_DIR, MODELS_DIR
from eval_external_common import run_eval

BASELINE = MODELS_DIR / "curated_resnet18_balanced.pt"
MULTISOURCE = MODELS_DIR / "curated_resnet18_multisource.pt"
WEIGHTED = MODELS_DIR / "curated_resnet18_multisource_weighted.pt"

DATASETS = [
    (SKINDISEASE_DIR / "manifest_curated_v3_test.csv", "Internal test set"),
    (SCIN_DIR / "manifest_scin_test.csv", "SCIN held-out test"),
    (SKINDISNET_DIR / "manifest_skindisnet_test.csv", "SkinDisNet held-out test"),
]

if __name__ == "__main__":
    results = {}
    for name, path in [("baseline", BASELINE), ("multisource", MULTISOURCE), ("weighted", WEIGHTED)]:
        for manifest_path, label in DATASETS:
            print(f"\n########## {name.upper()} -- {label} ##########")
            results[f"{name}__{label}"] = run_eval(manifest_path, f"{label} ({name})", model_path=path)

    print("\n\n========== SUMMARY (baseline -> multisource -> weighted) ==========")
    for manifest_path, label in DATASETS:
        b = results[f"baseline__{label}"]
        m = results[f"multisource__{label}"]
        w = results[f"weighted__{label}"]
        print(f"{label}:")
        print(f"  AUC {b['auc']:.4f} -> {m['auc']:.4f} -> {w['auc']:.4f}")
        print(f"  F1  {b['f1']:.4f} -> {m['f1']:.4f} -> {w['f1']:.4f}")
        print(f"  Acc {b['acc']:.4f} -> {m['acc']:.4f} -> {w['acc']:.4f}")

    with open(MODELS_DIR.parent / "experiment2b_weighted_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to experiment2b_weighted_results.json")
