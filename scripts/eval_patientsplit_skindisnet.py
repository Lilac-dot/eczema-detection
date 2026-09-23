"""Evaluates whether the SkinDisNet fine-tuning recovery (docs patient-leakage audit,
2026-09-17) survives once the leakage is removed.

Three models, evaluated on the NEW patient-clean SkinDisNet test partition
(manifest_skindisnet_test_patientsplit.csv, 539 images / 125 patients, zero patient overlap
with its own training partition by construction):
  - baseline (curated_resnet18_balanced.pt): never trained on any SkinDisNet data at all,
    so this is a valid zero-shot reference regardless of which SkinDisNet split is used.
  - leaky (curated_resnet18_multisource_weighted.pt): trained on the original, patient-
    leaky, image-level SkinDisNet split -- evaluated here on the NEW patient-clean test set
    (a harder, fairer test than the one it was originally reported against).
  - patientsplit (curated_resnet18_multisource_weighted_patientsplit.pt): trained on the
    new patient-clean SkinDisNet split -- the corrected experiment.

Also reports internal-test and SCIN-test for all three, to check for any forgetting effect
from the patient-level re-split itself (should be minimal/similar to the original weighted
run, since internal/SCIN data and recipe are unchanged).

Paired bootstrap tests (same test images, same order) are run for patientsplit-vs-baseline
and patientsplit-vs-leaky on the new SkinDisNet test set.
"""
import json

from paths import SKINDISEASE_DIR, SCIN_DIR, SKINDISNET_DIR, MODELS_DIR
from eval_external_common import run_eval, paired_bootstrap_test

MODELS = {
    "baseline": MODELS_DIR / "curated_resnet18_balanced.pt",
    "leaky": MODELS_DIR / "curated_resnet18_multisource_weighted.pt",
    "patientsplit": MODELS_DIR / "curated_resnet18_multisource_weighted_patientsplit.pt",
}

SKINDISNET_TEST = SKINDISNET_DIR / "manifest_skindisnet_test_patientsplit.csv"


def main():
    raw = {}
    print("=== SkinDisNet, NEW patient-clean test partition (539 images / 125 patients) ===")
    for name, path in MODELS.items():
        print(f"\n--- {name} ---")
        raw[name] = run_eval(SKINDISNET_TEST, f"SkinDisNet-patientclean-test ({name})",
                              model_path=path, return_raw=True)

    print("\n=== Forgetting check: internal test and SCIN test (naive 0.5 threshold) ===")
    for name, path in MODELS.items():
        run_eval(SKINDISEASE_DIR / "manifest_curated_v3_test.csv", f"Internal-test ({name})", model_path=path)
        run_eval(SCIN_DIR / "manifest_scin_test.csv", f"SCIN-test ({name})", model_path=path)

    print("\n\n========== Paired bootstrap tests on the patient-clean SkinDisNet test set ==========")
    b = raw["baseline"]
    l = raw["leaky"]
    ps = raw["patientsplit"]
    assert b["y_true"] == l["y_true"] == ps["y_true"], "row order mismatch"

    pt_ps_vs_baseline = paired_bootstrap_test(b["y_true"], ps["y_prob"], b["y_prob"])
    pt_ps_vs_leaky = paired_bootstrap_test(b["y_true"], ps["y_prob"], l["y_prob"])
    pt_leaky_vs_baseline = paired_bootstrap_test(b["y_true"], l["y_prob"], b["y_prob"])

    print(f"patientsplit - baseline: diff={pt_ps_vs_baseline['observed_diff']:.4f} "
          f"CI=({pt_ps_vs_baseline['ci'][0]:.4f}, {pt_ps_vs_baseline['ci'][1]:.4f}) "
          f"p={pt_ps_vs_baseline['p_value']:.4f}")
    print(f"patientsplit - leaky (leaky model tested on the CLEAN set): "
          f"diff={pt_ps_vs_leaky['observed_diff']:.4f} "
          f"CI=({pt_ps_vs_leaky['ci'][0]:.4f}, {pt_ps_vs_leaky['ci'][1]:.4f}) "
          f"p={pt_ps_vs_leaky['p_value']:.4f}")
    print(f"leaky - baseline (leaky model tested on the CLEAN set, i.e. its OWN train/test "
          f"leakage removed from the test side): diff={pt_leaky_vs_baseline['observed_diff']:.4f} "
          f"CI=({pt_leaky_vs_baseline['ci'][0]:.4f}, {pt_leaky_vs_baseline['ci'][1]:.4f}) "
          f"p={pt_leaky_vs_baseline['p_value']:.4f}")

    print("\n\n========== Summary ==========")
    for name in MODELS:
        print(f"{name}: SkinDisNet-patientclean-test AUC={raw[name]['auc']:.4f} "
              f"(95% CI {raw[name]['auc_ci'][0]:.4f}-{raw[name]['auc_ci'][1]:.4f}) "
              f"F1={raw[name]['f1']:.4f}")

    out = {name: {k: v for k, v in raw[name].items() if k not in ("y_true", "y_prob")} for name in raw}
    out["paired_tests"] = dict(
        patientsplit_minus_baseline=pt_ps_vs_baseline,
        patientsplit_minus_leaky=pt_ps_vs_leaky,
        leaky_minus_baseline_on_clean_test=pt_leaky_vs_baseline,
    )
    with open(MODELS_DIR.parent / "results" / "patientsplit_skindisnet_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nSaved to patientsplit_skindisnet_results.json")


if __name__ == "__main__":
    main()
