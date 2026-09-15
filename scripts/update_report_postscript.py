"""
One-off script: inserts a new "10. Postscript" section into the main report,
covering the 2026-09-15 external validation and generalization-improvement work,
positioned right before the References section. Run once; not part of the regular
pipeline.
"""
import docx
from docx.oxml.ns import qn

PATH = "Multi_Sensor_Wearable_AD_Monitoring_Paper_2026-09-15_revised.docx"

d = docx.Document(PATH)

refs_heading = None
for p in d.paragraphs:
    if p.text.strip() == "References" and p.style.name == "Heading 1":
        refs_heading = p
        break
assert refs_heading is not None, "Could not find References heading"

inserted_paragraphs = []


def add_para(text, style="Normal"):
    p = refs_heading.insert_paragraph_before(text, style=style)
    inserted_paragraphs.append(p)
    return p


def add_table(rows):
    """Build the table at the true end of the document, then relocate its XML
    element to sit immediately before the References heading."""
    n_rows, n_cols = len(rows), len(rows[0])
    t = d.add_table(rows=n_rows, cols=n_cols, style="Table Grid")
    for r, row_vals in enumerate(rows):
        for c, val in enumerate(row_vals):
            cell = t.cell(r, c)
            cell.text = str(val)
            if r == 0:
                for run in cell.paragraphs[0].runs:
                    run.bold = True
    refs_heading._p.addprevious(t._tbl)
    return t


# ---- 10. Postscript ----
add_para("10. Postscript: External Validation and Generalization Experiments "
          "(Added 15 September 2026)", style="Heading 1")
add_para(
    "This section was added after the rest of the report, to record a direct "
    "follow-up on the open question named in Section 7.2 and Table 6: no external, "
    "independently sourced test set had been checked against the deployed Stage B "
    "model. It has now been checked, and the results below update that claim's "
    "evidence status. As with the rest of this report, results here are reported "
    "as found, including a methodological error caught and corrected during this "
    "work (Section 10.2)."
)

# ---- 10.1 ----
add_para("10.1 Zero-shot external validation", style="Heading 2")
add_para(
    "The deployed Stage B model (the balanced CNN reported in Section 5.5, 81.07% "
    "accuracy / F1 81.18% on its own internal test set) was evaluated, unmodified, "
    "on two independently sourced datasets it had never seen: SCIN (Ward et al., "
    "JAMA Network Open 2024;7(11):e2446615), crowd-sourced consumer smartphone "
    "photographs from US Google Search users -- a genuinely different collection "
    "mechanism from this project's own DermNet-style archive -- and SkinDisNet "
    "(Sultana et al., Data in Brief 2025;63:112239), clinical smartphone photographs "
    "from two Bangladesh hospitals. Both were harmonized to the same Eczema-vs-Other "
    "binary framing as the internal task."
)
add_table([
    ["Test set", "n", "Accuracy", "AUC (95% CI)", "F1 (95% CI)"],
    ["Internal test (baseline)", "507", "0.8107", "0.8644 (0.8307-0.8973)", "0.8118 (0.7732-0.8493)"],
    ["SCIN (external)", "976", "0.5092", "0.5347 (0.5003-0.5692)", "0.1492 (0.1119-0.1886)"],
    ["SkinDisNet (external)", "1,710", "0.6673", "0.4865 (0.4559-0.5183)", "0.1123 (0.0794-0.1461)"],
])
add_para(
    "Table 7. Zero-shot Stage B performance, internal vs. external, no retraining."
)
add_para(
    "Both external AUCs are statistically indistinguishable from chance, a large "
    "and clearly-outside-confidence-interval drop from the internal baseline. This "
    "is not the same failure mode as the shortcut identified in Section 5.1: mean "
    "whole-image brightness, checked directly, shows no meaningful Eczema-vs-Other "
    "gap in either external set (SCIN 0.0015, SkinDisNet 0.0215, against the "
    "internal set's own now-small 0.056 gap and the original confound's 0.43), and "
    "the colour-feature LightGBM model -- the simpler, historically more "
    "shortcut-prone of the two Stage B models -- degrades to the same near-chance "
    "region as the CNN on both external sets rather than diverging from it, which "
    "would be the expected signature if the CNN alone had learned some subtler, "
    "colour-feature-invisible artifact. The more defensible interpretation is "
    "genuine distributional shift: the model has learned photographic conventions "
    "specific to its training archive (consistent macro-photography framing, "
    "lighting, focus, compression) that do not transfer to a different "
    "photographic source, rather than having learned nothing transferable about "
    "eczema at all -- it performs exactly as designed on same-source data. "
    "Table 6's claim (\"Image model distinguishes eczema from selected look-alike "
    "diseases\") should be updated accordingly: the external check that was open "
    "has now been run, and the model does not generalize zero-shot to either "
    "source tested. Full methodology: docs/external_validation_2026-09-15.md."
)

# ---- 10.2 ----
add_para("10.2 Attempts to improve cross-dataset generalization", style="Heading 2")
add_para(
    "Three interventions were tried against this result, each evaluated on "
    "identical held-out test data for direct comparability."
)
add_table([
    ["Test set", "Baseline AUC / F1", "+Augmentation (zero-shot)", "+Fine-tune, natural mix", "+Fine-tune, equal-weighted"],
    ["Internal test", "0.8644 / 0.8118", "n/a / 0.7515 acc.", "0.8530 / 0.7976", "0.8317 / 0.7758"],
    ["SCIN held-out", "0.5535 / 0.1395", "0.5358 / n/a", "0.5783 / 0.5214", "0.5794 / 0.5761"],
    ["SkinDisNet held-out", "0.4680 / 0.0324", "0.5016 / n/a", "0.6090 / 0.3823", "0.6515 / 0.4218"],
])
add_para(
    "Table 8. AUC / F1 across the three interventions, on held-out external test "
    "partitions never used in any training (SCIN and SkinDisNet splits from "
    "scripts/split_external_manifest.py). The Internal and SCIN/SkinDisNet rows "
    "under \"Baseline\" for the fine-tuning columns are computed on the smaller "
    "held-out partitions specifically, not the full external sets used in Table 7, "
    "so the two tables' baseline numbers differ slightly by design."
)
add_para(
    "Heavier augmentation alone (stronger colour jitter, random-resized crop, "
    "blur, and simulated recompression, with no external data touched) produced "
    "no meaningful change on either external set and cost internal accuracy "
    "(0.8107 to 0.7515), indicating the failure is not simple pixel-level "
    "brittleness that generic robustness training can fix. Fine-tuning on a "
    "proper train/val/test split of real external examples, mixed with internal "
    "data in each source's natural (unequal) proportion, produced a real "
    "AUC-level improvement on SkinDisNet (0.4680 to 0.6090) and a smaller, more "
    "threshold-driven improvement on SCIN (F1 rose sharply but AUC barely moved, "
    "0.5535 to 0.5783 -- the model is classifying more SCIN images correctly at "
    "the chosen threshold without ranking them much better). Weighting the three "
    "training sources to contribute equally per epoch, rather than in proportion "
    "to their raw size, improved on this further for both external sets -- most "
    "clearly for SkinDisNet (AUC 0.6090 to 0.6515) -- at the cost of the largest "
    "internal-accuracy drop of the three interventions (AUC 0.8644 to 0.8317)."
)
add_para(
    "A methodological note, in keeping with this report's practice of recording "
    "process failures alongside results (Section 7.7): the equal-weighted "
    "fine-tuning result was first evaluated against an intermediate "
    "\"best-validation-so-far\" checkpoint while its training process was still "
    "running, producing a materially different and more pessimistic result that "
    "was briefly treated as final before the discrepancy was caught and the "
    "model was re-evaluated against its true, completed state. The lesson "
    "generalizes directly from this report's own established practice: a "
    "plausible-looking intermediate result is not a substitute for confirming a "
    "process has actually finished before reporting its outcome."
)
add_para(
    "None of the three new models (curated_resnet18_augmented.pt, "
    "curated_resnet18_multisource.pt, curated_resnet18_multisource_weighted.pt) "
    "replace the deployed curated_resnet18_balanced.pt; they are retained "
    "separately as this comparison's record. A further, untested lever -- "
    "increasing the volume of real SCIN training examples specifically, rather "
    "than reweighting the existing amount -- remains open for future work, "
    "motivated by SCIN's AUC still trailing SkinDisNet's despite the F1 gain. "
    "Full methodology: docs/external_generalization_improvement_2026-09-15.md."
)

d.save(PATH)
print("Saved.")
print(f"Inserted {len(inserted_paragraphs)} paragraphs before the References heading.")
