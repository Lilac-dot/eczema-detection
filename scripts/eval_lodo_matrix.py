"""
Evaluates the 3 LODO models (train_lodo_resnet18.py) on their held-out third dataset's
test set, plus re-evaluates the relevant single-source models (curated_resnet18_balanced.pt,
scin_resnet18.pt, skindisnet_resnet18.pt) on those SAME test partitions with raw
per-image predictions (return_raw=True), so paired_bootstrap_test can compare LODO vs.
the corresponding single-source baseline on identical test images -- see
docs/cross_dataset_matrix_2026-09-15.md ("LODO" and "Significance testing" sections).

Held-out test sets used throughout: Internal (SkinDisease/manifest_curated_v3_test.csv,
507), SCIN (dataset/SCIN/manifest_scin_test.csv, 294), SkinDisNet -- the CLEANED version
(SkinDisNet/manifest_skindisnet_test_clean.csv, 513), consistent with the rest of this
project's post-cleanup evaluations.

Does not touch any existing model/manifest. Writes lodo_matrix_results.json (repo root,
not committed, same convention as cross_dataset_matrix_results.json).
"""
import json
import sys

from paths import ROOT, MODELS_DIR, SCIN_DIR, SKINDISNET_DIR, SKINDISEASE_DIR
sys.path.insert(0, str(ROOT / "scripts"))
from eval_external_common import run_eval, paired_bootstrap_test

INTERNAL_MODEL = MODELS_DIR / "curated_resnet18_balanced.pt"
SCIN_MODEL = MODELS_DIR / "scin_resnet18.pt"
SKINDISNET_MODEL = MODELS_DIR / "skindisnet_resnet18.pt"

LODO_INTERNAL_SCIN = MODELS_DIR / "lodo_internal_scin_resnet18.pt"          # -> tested on SkinDisNet
LODO_INTERNAL_SKINDISNET = MODELS_DIR / "lodo_internal_skindisnet_resnet18.pt"  # -> tested on SCIN
LODO_SCIN_SKINDISNET = MODELS_DIR / "lodo_scin_skindisnet_resnet18.pt"      # -> tested on Internal

INTERNAL_TEST = SKINDISEASE_DIR / "manifest_curated_v3_test.csv"
SCIN_TEST = SCIN_DIR / "manifest_scin_test.csv"
SKINDISNET_TEST_CLEAN = SKINDISNET_DIR / "manifest_skindisnet_test_clean.csv"


def evaluated(model_path, manifest_path, label):
    if not model_path.exists():
        print(f"SKIPPED (model not found): {label}")
        return None
    return run_eval(manifest_path, label, model_path=model_path, return_raw=True)


def main():
    results = {}

    print("\n" + "=" * 60)
    print("LODO cells")
    print("=" * 60)
    results["lodo_internal_scin_to_skindisnet"] = evaluated(
        LODO_INTERNAL_SCIN, SKINDISNET_TEST_CLEAN, "LODO(Internal+SCIN) -> SkinDisNet")
    results["lodo_internal_skindisnet_to_scin"] = evaluated(
        LODO_INTERNAL_SKINDISNET, SCIN_TEST, "LODO(Internal+SkinDisNet) -> SCIN")
    results["lodo_scin_skindisnet_to_internal"] = evaluated(
        LODO_SCIN_SKINDISNET, INTERNAL_TEST, "LODO(SCIN+SkinDisNet) -> Internal")

    print("\n" + "=" * 60)
    print("Matched single-source baselines (same test partitions, for paired comparison)")
    print("=" * 60)
    results["internal_to_skindisnet"] = evaluated(
        INTERNAL_MODEL, SKINDISNET_TEST_CLEAN, "Internal-only -> SkinDisNet (clean test)")
    results["internal_to_scin"] = evaluated(
        INTERNAL_MODEL, SCIN_TEST, "Internal-only -> SCIN")
    results["scin_to_internal"] = evaluated(
        SCIN_MODEL, INTERNAL_TEST, "SCIN-only -> Internal")
    results["skindisnet_to_internal"] = evaluated(
        SKINDISNET_MODEL, INTERNAL_TEST, "SkinDisNet-only -> Internal")

    print("\n" + "=" * 60)
    print("Paired significance tests (does adding a second source help?)")
    print("=" * 60)
    sig_tests = {}

    def add_paired_test(name, lodo_key, baseline_key, description):
        lodo, base = results.get(lodo_key), results.get(baseline_key)
        if lodo is None or base is None:
            print(f"SKIPPED (missing result): {name}")
            return
        test = paired_bootstrap_test(lodo["y_true"], lodo["y_prob"], base["y_prob"])
        sig_tests[name] = test
        sig = "SIGNIFICANT" if test["p_value"] < 0.05 else "not significant"
        print(f"{description}: AUC diff (LODO - baseline) = {test['observed_diff']:+.4f}, "
              f"95% CI [{test['ci'][0]:+.4f}, {test['ci'][1]:+.4f}], p={test['p_value']:.4f} "
              f"-- {sig}")

    add_paired_test(
        "internal_scin_lodo_vs_internal_only__on_skindisnet",
        "lodo_internal_scin_to_skindisnet", "internal_to_skindisnet",
        "LODO(Internal+SCIN) vs. Internal-only, both on SkinDisNet test")
    add_paired_test(
        "internal_skindisnet_lodo_vs_internal_only__on_scin",
        "lodo_internal_skindisnet_to_scin", "internal_to_scin",
        "LODO(Internal+SkinDisNet) vs. Internal-only, both on SCIN test")
    add_paired_test(
        "scin_skindisnet_lodo_vs_scin_only__on_internal",
        "lodo_scin_skindisnet_to_internal", "scin_to_internal",
        "LODO(SCIN+SkinDisNet) vs. SCIN-only, both on Internal test")
    add_paired_test(
        "scin_skindisnet_lodo_vs_skindisnet_only__on_internal",
        "lodo_scin_skindisnet_to_internal", "skindisnet_to_internal",
        "LODO(SCIN+SkinDisNet) vs. SkinDisNet-only, both on Internal test")

    out = {"results": results, "significance_tests": sig_tests}
    out_path = ROOT / "results" / "lodo_matrix_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=list)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
