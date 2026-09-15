"""
Experiment 2 evaluation: compares the zero-shot baseline model (curated_resnet18_balanced.pt)
against the multi-source fine-tuned model (curated_resnet18_multisource.pt), both run on the
SAME held-out test partitions (the _test.csv splits from split_external_manifest.py, never
seen during Experiment 2's training) plus the untouched internal test set, so "before" and
"after" are directly comparable. Run after train_curated_cnn_multisource.py.
"""
import json

from paths import SKINDISEASE_DIR, SCIN_DIR, SKINDISNET_DIR, MODELS_DIR
from eval_external_common import run_eval

BASELINE = MODELS_DIR / "curated_resnet18_balanced.pt"
MULTISOURCE = MODELS_DIR / "curated_resnet18_multisource.pt"

DATASETS = [
    (SKINDISEASE_DIR / "manifest_curated_v3_test.csv", "Internal test set"),
    (SCIN_DIR / "manifest_scin_test.csv", "SCIN held-out test"),
    (SKINDISNET_DIR / "manifest_skindisnet_test.csv", "SkinDisNet held-out test"),
]

if __name__ == "__main__":
    results = {}
    for manifest_path, label in DATASETS:
        print(f"\n########## BASELINE (zero-shot) -- {label} ##########")
        results[f"baseline__{label}"] = run_eval(manifest_path, f"{label} (baseline)", model_path=BASELINE)

    for manifest_path, label in DATASETS:
        print(f"\n########## MULTISOURCE FINE-TUNED -- {label} ##########")
        results[f"multisource__{label}"] = run_eval(manifest_path, f"{label} (multisource)", model_path=MULTISOURCE)

    print("\n\n========== SUMMARY (baseline -> multisource) ==========")
    for manifest_path, label in DATASETS:
        b = results[f"baseline__{label}"]
        m = results[f"multisource__{label}"]
        print(f"{label}: AUC {b['auc']:.4f} -> {m['auc']:.4f}  |  "
              f"F1 {b['f1']:.4f} -> {m['f1']:.4f}  |  Acc {b['acc']:.4f} -> {m['acc']:.4f}")

    with open(MODELS_DIR.parent / "experiment2_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to experiment2_results.json")
