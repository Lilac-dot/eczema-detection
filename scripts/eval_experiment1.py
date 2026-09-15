"""
Experiment 1 evaluation: the heavy-augmentation model (curated_resnet18_augmented.pt),
zero-shot, on the internal test set and the SAME full external manifests Phase 1 used
(no external data was touched during this experiment's training, so these numbers are
directly comparable to docs/external_validation_2026-09-15.md's baseline figures).
"""
import json

from paths import SKINDISEASE_DIR, SCIN_DIR, SKINDISNET_DIR, MODELS_DIR
from eval_external_common import run_eval

MODEL = MODELS_DIR / "curated_resnet18_augmented.pt"

DATASETS = [
    (SKINDISEASE_DIR / "manifest_curated_v3_test.csv", "Internal test set (augmented model)"),
    (SCIN_DIR / "manifest_scin.csv", "SCIN full external (augmented model)"),
    (SKINDISNET_DIR / "manifest_skindisnet.csv", "SkinDisNet full external (augmented model)"),
]

if __name__ == "__main__":
    results = {}
    for manifest_path, label in DATASETS:
        results[label] = run_eval(manifest_path, label, model_path=MODEL)

    with open(MODELS_DIR.parent / "experiment1_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to experiment1_results.json")
