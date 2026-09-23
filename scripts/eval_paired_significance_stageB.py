"""One-off analysis for the report revision: formal paired significance testing between
the deployed Stage B baseline (curated_resnet18_balanced.pt) and the equal-weighted
multisource fine-tune (curated_resnet18_multisource_weighted.pt), on the SAME held-out
external test images, using the paired_bootstrap_test already implemented in
eval_external_common.py. Also reports the multisource (natural-proportion) model for
completeness, and states explicitly, per AUC value, whether the 95% CI excludes chance
(0.5). Inference only -- no training, no model modification, no test-set-influenced
decision made here (this script does not select a model or a threshold, only reports
statistics about models/thresholds already fixed elsewhere).
"""
import json

from paths import SKINDISEASE_DIR, SCIN_DIR, SKINDISNET_DIR, MODELS_DIR
from eval_external_common import run_eval, paired_bootstrap_test

MODELS = {
    "baseline": MODELS_DIR / "curated_resnet18_balanced.pt",
    "multisource": MODELS_DIR / "curated_resnet18_multisource.pt",
    "weighted": MODELS_DIR / "curated_resnet18_multisource_weighted.pt",
}

DATASETS = [
    (SKINDISEASE_DIR / "manifest_curated_v3_test.csv", "internal_test"),
    (SCIN_DIR / "manifest_scin_test.csv", "scin_heldout_test"),
    (SKINDISNET_DIR / "manifest_skindisnet_test.csv", "skindisnet_heldout_test"),
]


def chance_verdict(auc, ci):
    lo, hi = ci
    if lo > 0.5:
        return f"AUC {auc:.4f}, 95% CI [{lo:.4f}, {hi:.4f}] -- excludes chance (CI entirely above 0.5)"
    elif hi < 0.5:
        return f"AUC {auc:.4f}, 95% CI [{lo:.4f}, {hi:.4f}] -- excludes chance (CI entirely below 0.5, worse than random)"
    else:
        return f"AUC {auc:.4f}, 95% CI [{lo:.4f}, {hi:.4f}] -- does NOT exclude chance (CI spans 0.5)"


def main():
    raw = {}
    for manifest_path, dslabel in DATASETS:
        raw[dslabel] = {}
        for name, path in MODELS.items():
            print(f"\n--- {name} on {dslabel} ---")
            raw[dslabel][name] = run_eval(manifest_path, f"{dslabel} ({name})",
                                           model_path=path, return_raw=True)

    print("\n\n========== Chance-level check (95% CI vs AUC=0.5) ==========")
    chance_summary = {}
    for manifest_path, dslabel in DATASETS:
        chance_summary[dslabel] = {}
        for name in MODELS:
            r = raw[dslabel][name]
            verdict = chance_verdict(r["auc"], r["auc_ci"])
            print(f"{dslabel} / {name}: {verdict}")
            chance_summary[dslabel][name] = verdict

    print("\n\n========== Paired bootstrap test: baseline vs weighted (same images) ==========")
    paired_results = {}
    for manifest_path, dslabel in DATASETS:
        b = raw[dslabel]["baseline"]
        w = raw[dslabel]["weighted"]
        m = raw[dslabel]["multisource"]
        assert b["y_true"] == w["y_true"] == m["y_true"], f"row order mismatch on {dslabel}"
        pt_bw = paired_bootstrap_test(b["y_true"], w["y_prob"], b["y_prob"])
        pt_bm = paired_bootstrap_test(b["y_true"], m["y_prob"], b["y_prob"])
        pt_mw = paired_bootstrap_test(b["y_true"], w["y_prob"], m["y_prob"])
        print(f"\n{dslabel}:")
        print(f"  weighted - baseline: diff={pt_bw['observed_diff']:.4f} "
              f"CI=({pt_bw['ci'][0]:.4f}, {pt_bw['ci'][1]:.4f}) p={pt_bw['p_value']:.4f}")
        print(f"  multisource - baseline: diff={pt_bm['observed_diff']:.4f} "
              f"CI=({pt_bm['ci'][0]:.4f}, {pt_bm['ci'][1]:.4f}) p={pt_bm['p_value']:.4f}")
        print(f"  weighted - multisource: diff={pt_mw['observed_diff']:.4f} "
              f"CI=({pt_mw['ci'][0]:.4f}, {pt_mw['ci'][1]:.4f}) p={pt_mw['p_value']:.4f}")
        paired_results[dslabel] = dict(weighted_minus_baseline=pt_bw,
                                        multisource_minus_baseline=pt_bm,
                                        weighted_minus_multisource=pt_mw)

    out = dict(chance_summary=chance_summary, paired_results=paired_results)
    with open(MODELS_DIR.parent / "results" / "paired_significance_stageB_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nSaved to paired_significance_stageB_results.json")


if __name__ == "__main__":
    main()
