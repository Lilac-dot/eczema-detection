"""Revision: insert the new Section 7 (7.1-7.9), reorganized around findings
rather than claim-by-claim apology, before '8. Proposed Hardware Architecture'."""
import docx

PATH = "Multi_Sensor_Wearable_AD_Monitoring_Paper_2026-09-15_revised.docx"
d = docx.Document(PATH)
paras = d.paragraphs

hw_heading = None
for p in paras:
    if p.text.strip() == "8. Proposed Hardware Architecture (Not Yet Implemented)" and p.style.name == "Heading 1":
        hw_heading = p
        break
assert hw_heading is not None

def add(text, style="Normal"):
    return hw_heading.insert_paragraph_before(text, style=style)

add("7.1 Wearable stress detection", style="Heading 2")
add(
    "Partially supported. The leave-one-subject-out protocol is methodologically sound "
    "and free of subject-level leakage, and the personal-baseline calibration fix "
    "(Section 4.5) produced a real, verified improvement for the deployed LightGBM "
    "model (mean AUC 0.871 to 0.9405, pooled F1 0.646 to 0.7602). But fifteen subjects "
    "is a small sample, one subject (S5) remains a threshold-transfer failure even "
    "after the fix, and the underlying condition is a lab-induced social-stress test "
    "in healthy adults, not naturalistic daily-life stress in an eczema population."
)

add("7.2 Image classification within source distribution", style="Heading 2")
add(
    "Partially supported. The specific cross-source shortcut that inflated the "
    "original 95.66% figure (Section 5.2) was substantially reduced in the rebuilt "
    "task, evidenced by the model's errors clustering on genuine clinical look-alikes "
    "rather than an arbitrary split, reaching 81.07% accuracy / F1 81.18% on "
    "same-source held-out data (Section 5.5). This establishes that the model learns "
    "something real about the seven-way disease comparison within its own source "
    "archive. It does not, on its own, establish generalization beyond that archive -- "
    "Section 7.3 addresses that question directly."
)

add("7.3 External generalization failure", style="Heading 2")
add(
    "Not supported -- a negative result. Zero-shot evaluation of the deployed model "
    "on two independently sourced datasets it had never seen showed near-chance "
    "discrimination: AUC 0.5347 on SCIN and 0.4865 on SkinDisNet, against 0.8644 "
    "internally (Section 5.6, Table 7). This is a large, statistically clear drop, not "
    "an ambiguous one. Diagnostic checks rule out the specific failure mode that broke "
    "the original Eczema-vs-Normal classifier: neither external set shows a brightness "
    "gap anywhere near the original 0.43 confound (both under 0.03), and the "
    "colour-feature LightGBM model -- historically the more shortcut-prone of the two "
    "Stage B models -- degrades to the same near-chance region as the CNN rather than "
    "diverging from it, the signature expected of genuine distributional shift rather "
    "than a residual, findable shortcut. The most defensible interpretation is that the "
    "model has learned photographic conventions specific to its training archive "
    "(consistent macro-photography framing, lighting, focus, compression) that do not "
    "transfer to a different photographic source, rather than having learned nothing "
    "transferable about eczema."
)

add("7.4 Multisource generalization improvement", style="Heading 2")
add(
    "Partially supported. Three interventions were tested against the result in "
    "Section 7.3 (Section 5.7, Table 8). Generic data augmentation, with no external "
    "data touched, produced no meaningful change on either external set and cost "
    "internal accuracy, indicating the failure is not simple pixel-level brittleness. "
    "Fine-tuning on a proper train/val/test split of real external examples, mixed in "
    "each source's natural proportion, produced a real AUC-level improvement on "
    "SkinDisNet (0.4680 to 0.6090) and a smaller, more threshold-driven improvement on "
    "SCIN. Weighting the three training sources to contribute equally per epoch "
    "improved on this further for both external sets -- most clearly for SkinDisNet "
    "(AUC reaching 0.6515) -- at the cost of the largest internal-accuracy drop of the "
    "three interventions (AUC 0.8644 to 0.8317). Domain diversity and source balancing "
    "improved cross-dataset generalization, but did not fully solve it: SCIN's AUC "
    "remained close to chance throughout, and no intervention approached internal "
    "performance on external data."
)

add("7.5 Sleep detection failure", style="Heading 2")
add(
    "Not supported -- a negative result, reported rather than omitted (Section 4.8). "
    "The AAUWSS-trained sleep-disruption model scored at or below chance (mean LOSO "
    "AUC 0.4642, pooled AUC 0.3867), and two independent, literature-standard "
    "actigraphy formulas (Cole-Kripke [6], Sadeh [12]) applied to the same data scored "
    "similarly at chance, indicating the difficulty was not specific to this project's "
    "modelling choices. This does not mean sleep disturbance is irrelevant to atopic "
    "dermatitis -- Section 2.6 documents a real, literature-supported association -- "
    "only that this project did not succeed in building a generalizable "
    "sleep-disruption detector from the one dataset available to it. The component is "
    "excluded from the deployed fusion pipeline accordingly."
)

add("7.6 Multimodal fusion limitation", style="Heading 2")
add(
    "Not supported, and not tested. No dataset used anywhere in this project pairs "
    "wearable signals (stress or sleep) with eczema severity outcomes for the same "
    "patients, so the premise that stress or sleep disturbance predicts eczema "
    "severity rests entirely on external literature (Khan et al. [8] for stress, "
    "Bawany et al. [4] for sleep, the latter explicitly the weaker claim per that "
    "review's own conclusion) -- not on any experiment run here. Because no such "
    "paired dataset exists, the claim that combining stress and image information "
    "improves eczema monitoring is not tested: Stage C's demonstration necessarily "
    "uses image and wearable examples from different people and moments (Section 6.3), "
    "so it shows only that the architecture runs and combines its two inputs "
    "correctly, not that the result is clinically meaningful. Consequently the "
    "trigger-modulated risk index it produces is not supported as a representation of "
    "eczema severity: the formula has never been calibrated or validated against any "
    "severity ground truth, because none is available to this project."
)

add("7.7 Hardware/deployment limitation", style="Heading 2")
add(
    "Not supported. No physical wearable has been built or tested (Section 8); the "
    "wrist-node architecture is a proposal grounded in the trained models' actual "
    "sensor requirements, not a validated device. Even the proposed design's central "
    "premise -- that a low-cost sensor package can reproduce the signal "
    "characteristics the WESAD-trained model needs -- is untested, and is named "
    "explicitly as the next practical step (Section 8.6) rather than something "
    "already demonstrated."
)

add("7.8 Overall scientific interpretation", style="Heading 2")
add(
    "The project successfully demonstrates several independently evaluated "
    "components -- a calibrated wearable stress detector, a shortcut-corrected image "
    "classifier, and a working proof-of-concept fusion architecture -- but the "
    "complete multimodal eczema-monitoring hypothesis remains untested, because the "
    "necessary paired, longitudinal, patient-level dataset linking wearable signals, "
    "skin photographs, and clinician-assessed severity is unavailable in the public "
    "domain. Two further findings sharpen this picture beyond what was known when the "
    "project began: the image classifier's apparent within-distribution competence "
    "does not transfer across photographic sources (Section 7.3), and targeted "
    "exposure to real external data -- not generic robustness training -- is what "
    "partially closes that gap (Section 7.4). Table 6 summarizes the evidence status "
    "of every claim this report is in a position to make."
)

add("7.9 Reproducibility and methodological hygiene", style="Heading 2")
add(
    "During development, several reproducibility and implementation issues were "
    "identified and corrected, and the final pipeline reflects those corrections: a "
    "missing random seed in the Stage B training script (the reported 81.07% figure "
    "was not exactly reproducible from a rerun), hardcoded absolute file paths "
    "preventing the pipeline from running on another machine, an unchecked "
    "cross-class duplicate risk in the disease-merge step (an Eczema-labelled photo "
    "could in principle have been a duplicate of an image labelled with a different "
    "disease), a ResNet18 fine-tuning bug in which supposedly frozen backbone layers "
    "were still updating their batch-normalisation statistics during training, and a "
    "stale top-level project report predating the WISDM-to-WESAD pivot, left where it "
    "could be mistaken for the current state of the project. None of these invalidate "
    "a specific reported number on their own, but each was a real gap between what "
    "the project claimed about itself and what the code actually did; each is now "
    "closed."
)

d.save(PATH)
print("Section 7 rewritten: 7.1-7.9 inserted.")
