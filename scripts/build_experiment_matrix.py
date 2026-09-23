# -*- coding: utf-8 -*-
"""Consolidated experiment matrix for the eczema-classification compression/deployment
paper (reframed 2026-09-19: this is an eczema image-classification study with
compression as the deployment mechanism, not a generic edge-AI benchmark). Reads every
already-completed result file on disk and produces one table per architecture --
baseline performance, params, size, memory, INT8, pruning, corruption-robustness,
SE/swish ablation, CI presence, validation/source status -- so nothing gets rerun that
doesn't need to be, per the explicit instruction to reuse valid existing results.

Rows for QAT / structured pruning / pruning+fine-tune are filled in as those background
jobs complete; this script is safe to re-run at any time (it just reads what's on disk).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "papers" / "edge-ai-lightweight-deployment"
RESULTS_DIR = ROOT / "results"

ARCHS = [
    "resnet18", "efficientnet_b0", "efficientnet_lite0", "mobilenetv2_100",
    "mobilenetv3_small", "shufflenet_v2_x0_5", "shufflenet_v2_x1_0",
    "squeezenet1_1", "repghostnet_050",
]


def load(path):
    return json.load(open(path)) if path.exists() else None


def main():
    ext = load(PAPER_DIR / "edge_ai_extended_analysis_2026-09-19.json")
    struct = load(PAPER_DIR / "edge_ai_structured_pruning_2026-09-19.json")
    qat = load(PAPER_DIR / "edge_ai_qat_2026-09-19.json")
    pf = load(PAPER_DIR / "edge_ai_pruning_finetune_2026-09-19.json")
    corr_summary = load(RESULTS_DIR / "robustness_cross_architecture_summary.json")
    corr_by_model = {row["model"]: row for row in (corr_summary or {}).get("rows", [])}

    rows = []
    for arch in ARCHS:
        row = {"architecture": arch}
        m = (ext or {}).get("per_model", {}).get(arch, {})

        fp32 = m.get("fp32", {})
        fp32_pt = fp32.get("point", {})
        row["baseline_accuracy"] = fp32_pt.get("accuracy")
        row["baseline_precision"] = fp32_pt.get("precision")
        row["baseline_sensitivity_recall"] = fp32_pt.get("sensitivity_recall")
        row["baseline_specificity"] = fp32_pt.get("specificity")
        row["baseline_f1"] = fp32_pt.get("f1")
        row["baseline_auroc"] = fp32_pt.get("auroc")
        fp32_ci = fp32.get("bootstrap_ci", {})
        for metric in ("accuracy", "sensitivity_recall", "specificity"):
            ci = fp32_ci.get(metric, {})
            row[f"baseline_{metric}_ci95"] = (
                f"[{ci.get('ci_lo'):.4f}, {ci.get('ci_hi'):.4f}]" if ci.get("ci_lo") is not None else None
            )
        row["memory_mb_fp32"] = fp32.get("memory_mb", {}).get("delta_mb")

        s_arch = (struct or {}).get("per_model", {}).get(arch, {})
        if s_arch:
            row["params_fp32"] = s_arch.get("fp32", {}).get("params")
            row["checkpoint_size_mb_fp32"] = s_arch.get("fp32", {}).get("checkpoint_size_mb")

        int8 = m.get("static_int8", {})
        int8_pt = int8.get("point", {})
        row["int8_accuracy"] = int8_pt.get("accuracy")
        row["int8_sensitivity_recall"] = int8_pt.get("sensitivity_recall")
        row["int8_specificity"] = int8_pt.get("specificity")
        row["int8_f1"] = int8_pt.get("f1")
        row["int8_coverage_fraction"] = int8.get("op_coverage", {}).get("coverage_fraction")
        row["int8_acc_drop_from_fp32"] = (
            row["baseline_accuracy"] - row["int8_accuracy"]
            if row.get("baseline_accuracy") is not None and row.get("int8_accuracy") is not None
            else None
        )

        pruned = m.get("pruned_at_knee", {})
        pruned_pt = pruned.get("point", {})
        row["pruning_knee_sparsity"] = m.get("knee_sparsity_used")
        row["pruned_at_knee_accuracy_unstructured_no_finetune"] = pruned_pt.get("accuracy")
        row["pruned_at_knee_sensitivity_recall"] = pruned_pt.get("sensitivity_recall")
        row["pruned_at_knee_specificity"] = pruned_pt.get("specificity")
        row["pruned_at_knee_f1"] = pruned_pt.get("f1")

        combined = m.get("pruned30_plus_int8", {})
        row["pruned30_plus_int8_accuracy"] = combined.get("point", {}).get("accuracy")

        if s_arch:
            struct_at_30 = s_arch.get("structured_pruning", {}).get("0.3", {})
            struct_pt = struct_at_30.get("point", {})
            row["structured_pruned30_accuracy_no_finetune"] = struct_pt.get("accuracy")
            row["structured_pruned30_sensitivity_recall"] = struct_pt.get("sensitivity_recall")
            row["structured_pruned30_specificity"] = struct_pt.get("specificity")
            row["structured_pruned30_size_reduction_fraction"] = struct_at_30.get("size_reduction_fraction")
        elif arch in ("shufflenet_v2_x0_5", "shufflenet_v2_x1_0"):
            row["structured_pruning_note"] = "excluded: torch-pruning fails on channel-shuffle structure"

        c = corr_by_model.get(arch, {})
        row["corruption_clean_acc_check"] = c.get("clean_acc")
        row["corruption_blur_drop_sev1to5"] = c.get("gaussian_blur_drop")
        row["corruption_jpeg_drop_sev1to5"] = c.get("jpeg_compression_drop")

        se = (ext or {}).get("se_swish_ablation", {}).get(arch)
        row["se_swish_ablation_available"] = se is not None

        q = (qat or {}).get(arch)
        if q:
            qat_pt = q.get("val_metrics", {})
            row["qat_int8_accuracy"] = qat_pt.get("accuracy")
            row["qat_int8_sensitivity_recall"] = qat_pt.get("sensitivity_recall")
            row["qat_int8_specificity"] = qat_pt.get("specificity")
            row["qat_int8_f1"] = qat_pt.get("f1")
            row["qat_epochs_completed"] = q.get("epochs_completed")
        elif arch in ("efficientnet_b0", "mobilenetv3_small", "repghostnet_050"):
            row["qat_status"] = "queued/in progress"

        pf_arch = (pf or {}).get("per_model", {}).get(arch)
        if pf_arch:
            row["pruning_finetune_status"] = "done"
            row["pruning_finetune_sparsity_levels_done"] = list(pf_arch.get("sparsity_levels", {}).keys())
            for lvl, res in pf_arch.get("sparsity_levels", {}).items():
                post = res.get("after_finetune", {}).get("point", {})
                row[f"pf_sparsity{lvl}_accuracy_after_finetune"] = post.get("accuracy")
                row[f"pf_sparsity{lvl}_sensitivity_recall_after_finetune"] = post.get("sensitivity_recall")
                row[f"pf_sparsity{lvl}_specificity_after_finetune"] = post.get("specificity")
                row[f"pf_sparsity{lvl}_accuracy_recovered_vs_no_finetune"] = res.get("accuracy_recovered")
        elif arch in ("resnet18", "shufflenet_v2_x1_0", "mobilenetv3_small"):
            row["pruning_finetune_status"] = "queued/in progress"

        row["source_files"] = "; ".join(filter(None, [
            "edge_ai_extended_analysis_2026-09-19.json" if arch in (ext or {}).get("per_model", {}) else None,
            "edge_ai_structured_pruning_2026-09-19.json" if s_arch else None,
            "robustness_cross_architecture_summary.json" if c else None,
            f"results/robustness_corruption_results_{arch}.json" if c else None,
            "edge_ai_qat_2026-09-19.json" if q else None,
            "edge_ai_pruning_finetune_2026-09-19.json" if pf_arch else None,
        ]))
        row["validated"] = bool(fp32) and bool(c)

        rows.append(row)

    out_json = PAPER_DIR / "experiment_matrix_2026-09-19.json"
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2, default=str)

    cols = list({k for r in rows for k in r.keys()})
    cols.sort(key=lambda c: (c != "architecture", c))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")

    out_md = PAPER_DIR / "experiment_matrix_2026-09-19.md"
    out_md.write_text(
        "# Eczema Classification -- Compression/Deployment Experiment Matrix (2026-09-19)\n\n"
        "Auto-generated from result files already on disk -- see build_experiment_matrix.py. "
        "Re-run after each background job (QAT / structured pruning / pruning+fine-tune) "
        "finishes to refresh.\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_json} and {out_md} ({len(rows)} architectures)")


if __name__ == "__main__":
    main()
