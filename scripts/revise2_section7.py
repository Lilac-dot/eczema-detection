"""Major revision pass 2, part B: rename and restructure Section 7 around
findings (finding first, qualification second), remove claim-by-claim framing,
consolidate repeated limitations, positive tone throughout."""
import docx

PATH = "Multi_Sensor_Wearable_AD_Monitoring_Paper_2026-09-15_revised.docx"
d = docx.Document(PATH)
paras = d.paragraphs

def find(text, style=None):
    for p in paras:
        if p.text.strip() == text and (style is None or p.style.name == style):
            return p
    return None

h7 = find("7. Discussion: What Is and Is Not Supported", "Heading 1")
hw_h = find("8. Proposed Hardware Architecture (Not Yet Implemented)", "Heading 1")
assert h7 and hw_h

body = d.element.body
children = list(body)
start_idx = children.index(h7._p)
end_idx = children.index(hw_h._p)
old_block = children[start_idx + 1:end_idx]  # keep the heading paragraph itself, retitle it
for el in old_block:
    el.getparent().remove(el)

for r in h7.runs:
    r.text = ""
h7.runs[0].text = "7. Discussion and Scientific Interpretation"

def add(text, style="Normal"):
    return hw_h.insert_paragraph_before(text, style=style)

def bullet(text):
    return hw_h.insert_paragraph_before(text, style="List Bullet")

add(
    "This section interprets the experimental results introduced in Sections 4 through 6, organized "
    "around what the experiments found. Table 6 gives a concise evidence-status summary; the "
    "discussion below explains the findings behind it. Per Section 1.3, the general boundary of what "
    "these experiments can and cannot establish is stated once there and is not repeated below except "
    "where it bears directly on a specific result."
)
add("Table 6. Summary of claims and evidence status.")

add("7.1 Subject dependence in wearable stress detection", style="Heading 2")
add(
    "The leave-one-subject-out evaluation of the wearable stress model exposed a form of failure "
    "invisible to its own headline metric: several folds combined a near-perfect AUC with an F1 of "
    "0.00, because the decision threshold tuned on one subject did not transfer to another. This is a "
    "genuine methodological finding about wearable stress-model evaluation -- a strong pooled AUC can "
    "conceal complete per-subject failure -- identified through subject-level analysis this project "
    "chose to run rather than being visible from the aggregate result alone."
)

add("7.2 What personal-baseline calibration achieved", style="Heading 2")
add(
    "Personal-baseline calibration produced a substantial, verified improvement: mean AUC rose from "
    "0.871 to 0.9405, pooled F1 from 0.646 to 0.7602, and the per-fold decision-threshold spread "
    "collapsed from an 88-fold range to a 2.8-fold range. Three previously near-random subjects (S7, "
    "S13, S17) reached 97.1%, 57.1%, and 83.3% accuracy respectively. The result indicates that "
    "physiological variability between individuals remains important even after calibration: one "
    "subject (S5) continued to show threshold-transfer failure, showing the fix reduced rather than "
    "eliminated subject-dependent instability."
)

add("7.3 Shortcut learning in dermatological image classification", style="Heading 2")
add(
    "Auditing the original 95.66%-accuracy Eczema-vs-Normal classifier, rather than accepting the "
    "figure, revealed it was driven by a photographic-source brightness confound rather than lesion "
    "features. Rebuilding the task from a same-source archive removed this specific confound, "
    "evidenced by the rebuilt model's errors clustering on genuine clinical look-alikes, and produced "
    "a trustworthy internal result of 81.07% accuracy and 81.18% F1. This is a concrete demonstration "
    "of shortcut learning caught and corrected within a real project pipeline, not merely a "
    "theoretical risk."
)

add("7.4 Cross-dataset generalization failure", style="Heading 2")
add(
    "Zero-shot evaluation of the rebuilt classifier on two independently sourced datasets it had "
    "never seen established that its internal competence does not transfer: AUC fell from 0.8644 "
    "internally to 0.5347 on SCIN and 0.4865 on SkinDisNet, both near chance. Diagnostic checks "
    "established this was a different phenomenon from the original brightness shortcut -- neither "
    "external set shows a comparable brightness gap, and a simpler colour-feature model degraded in "
    "step with the CNN rather than diverging from it -- pointing to genuine distributional shift tied "
    "to archive-specific photographic convention. This experiment demonstrates that strong "
    "within-source dermatological image performance does not imply cross-source generalization, a "
    "finding with direct relevance beyond this project."
)

add("7.5 What multisource training achieved", style="Heading 2")
add(
    "Testing three interventions against this gap established which kind of fix actually works: "
    "generic augmentation, with no external data touched, produced no meaningful change. Fine-tuning "
    "on real external examples did: natural-proportion multisource training raised SkinDisNet AUC to "
    "0.6090, and equal-weighted multisource training raised it further to 0.6515, the strongest "
    "result obtained. SCIN's ranking quality improved much less (to approximately 0.5794), and "
    "internal accuracy declined under both multisource interventions, establishing a real, quantified "
    "trade-off between source-specific and externally robust performance rather than a free "
    "improvement."
)

add("7.6 Negative result: sleep-disruption detection", style="Heading 2")
add(
    "The AAUWSS-trained sleep-disruption model scored at chance (mean LOSO AUC 0.4642, pooled AUC "
    "0.3867), and two independent, literature-standard actigraphy formulas applied to the same data "
    "scored similarly at chance, establishing that the difficulty was not specific to this project's "
    "modelling choices. Sleep disturbance remains a literature-supported correlate of AD severity "
    "(Section 2.6); this project's specific attempt to detect it from the available wrist-worn data "
    "did not succeed, and the component was excluded from the deployed fusion pipeline on that "
    "evidence."
)

add("7.7 Implications for multimodal eczema monitoring", style="Heading 2")
add(
    "Taken together, Stages A and B establish that their respective sub-problems -- physiological "
    "stress detection and dermatological image classification -- can be approached with public data, "
    "audited for hidden failure modes, and iteratively improved. Combining their outputs into a single "
    "eczema-monitoring signal is a separate claim this project's evidence does not extend to: no "
    "dataset pairing wearable recordings with skin photographs and clinician-assessed severity for the "
    "same patients was identified, so Stage C's fusion (Section 6) demonstrates computational "
    "integration rather than a validated combined predictor. The premise that stress or sleep "
    "disturbance modulates eczema risk is a literature-motivated design choice (Khan et al. [8], "
    "Bawany et al. [4]), not a claim tested by this project's own data."
)

add("7.8 Hardware and deployment pathway", style="Heading 2")
add(
    "The models and fusion logic developed in this project run today as software on public data "
    "(Section 8.1). What has not yet been built is the physical wrist sensor node and its validation "
    "against the Empatica E4 reference device the models were trained on (Section 8.6) -- the "
    "concrete, well-specified next engineering phase, not an open-ended gap."
)

add("7.9 Overall contribution of the study", style="Heading 2")
add(
    "The principal contribution of this study is therefore not a single headline accuracy value, but "
    "an experimentally evaluated development pipeline showing how multimodal eczema-monitoring "
    "components behave when subjected to progressively stronger tests of validity. The study "
    "established a calibrated wearable stress detector, exposed and corrected a photographic shortcut "
    "in dermatological image classification, quantified severe cross-dataset degradation, "
    "demonstrated partial recovery through multisource training, documented a reproducible negative "
    "result for sleep-disruption detection, and implemented a working decision-level fusion "
    "architecture. Together, these results provide validated component-level evidence and a concrete "
    "specification of the remaining experimental step: longitudinal collection of paired wearable, "
    "photographic, and clinical data from the same participants."
)

add("7.10 Reproducibility and methodological hygiene", style="Heading 2")
add(
    "A final reproducibility audit identified and corrected several implementation issues, improving "
    "the alignment between the reported methodology and the executable code: a missing random seed in "
    "the Stage B training script, hardcoded absolute file paths, an unchecked cross-class duplicate "
    "risk in the disease-merge step, a ResNet18 fine-tuning bug in which frozen backbone layers were "
    "still updating batch-normalisation statistics, and a stale top-level report predating the "
    "WISDM-to-WESAD pivot. None of these invalidate a specific reported number on their own; each "
    "reflects the same audit-and-correct approach applied throughout this project, and each is now "
    "closed."
)

d.save(PATH)
print("Section 7 fully restructured: 7.1-7.10, finding-first style, positive framing.")
