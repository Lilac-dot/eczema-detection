"""
Evaluates the 6 new cells of the cross-dataset generalization matrix (SCIN-trained and
SkinDisNet-trained models, each tested on all three sources) using the exact same
pipeline as the existing Internal-trained cells (eval_external_common.py), so all 9
matrix cells are computed identically and comparable.

Does not touch models/curated_resnet18_balanced.pt or any existing model/manifest.
Writes cross_dataset_matrix_results.json (repo root, not committed -- same convention
as experiment1/2/2b_results.json).
"""
import json
import sys

from paths import ROOT, MODELS_DIR, SCIN_DIR, SKINDISNET_DIR, SKINDISEASE_DIR
sys.path.insert(0, str(ROOT / "scripts"))
from eval_external_common import run_eval

SCIN_MODEL = MODELS_DIR / "scin_resnet18.pt"
SKINDISNET_MODEL = MODELS_DIR / "skindisnet_resnet18.pt"

INTERNAL_TEST = SKINDISEASE_DIR / "manifest_curated_v3_test.csv"
SCIN_TEST = SCIN_DIR / "manifest_scin_test.csv"
SKINDISNET_TEST_CLEAN = SKINDISNET_DIR / "manifest_skindisnet_test_clean.csv"


def main():
    results = {}

    print("\n" + "=" * 60)
    print("SCIN-trained model")
    print("=" * 60)
    results["scin_to_scin"] = run_eval(SCIN_TEST, "SCIN -> SCIN", model_path=SCIN_MODEL)
    results["scin_to_internal"] = run_eval(INTERNAL_TEST, "SCIN -> Internal", model_path=SCIN_MODEL)
    results["scin_to_skindisnet"] = run_eval(SKINDISNET_TEST_CLEAN, "SCIN -> SkinDisNet", model_path=SCIN_MODEL)

    print("\n" + "=" * 60)
    print("SkinDisNet-trained model")
    print("=" * 60)
    results["skindisnet_to_skindisnet"] = run_eval(SKINDISNET_TEST_CLEAN, "SkinDisNet -> SkinDisNet", model_path=SKINDISNET_MODEL)
    results["skindisnet_to_internal"] = run_eval(INTERNAL_TEST, "SkinDisNet -> Internal", model_path=SKINDISNET_MODEL)
    results["skindisnet_to_scin"] = run_eval(SCIN_TEST, "SkinDisNet -> SCIN", model_path=SKINDISNET_MODEL)

    out_path = ROOT / "results" / "cross_dataset_matrix_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=list)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
