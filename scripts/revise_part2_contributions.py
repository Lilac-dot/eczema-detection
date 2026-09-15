"""Revision Part 2: replace the flat contributions bullet list (old indices 16-25)
with a categorized A-E structure, inserted before '2. Literature Review'."""
import docx

PATH = "Multi_Sensor_Wearable_AD_Monitoring_Paper_2026-09-15_revised.docx"
d = docx.Document(PATH)
paras = d.paragraphs

lit_review_heading = None
for p in paras:
    if p.text.strip() == "2. Literature Review" and p.style.name == "Heading 1":
        lit_review_heading = p
        break
assert lit_review_heading is not None

# Old bullets to remove: the 10 List Bullet paragraphs between "1.1 Contributions"
# intro and "2. Literature Review".
contrib_heading = None
for p in paras:
    if p.text.strip() == "1.1 Contributions":
        contrib_heading = p
        break
assert contrib_heading is not None

start = False
to_delete = []
for p in paras:
    if p is contrib_heading:
        start = True
        continue
    if p is lit_review_heading:
        break
    if start and p.style.name == "List Bullet":
        to_delete.append(p)

for p in to_delete:
    p._element.getparent().remove(p._element)

def add_bold_para(text):
    p = lit_review_heading.insert_paragraph_before(text, style="Normal")
    p.runs[0].bold = True
    return p

def add_bullet(text):
    return lit_review_heading.insert_paragraph_before(text, style="List Bullet")

add_bold_para("A. Experimental contributions")
add_bullet("Evaluates wearable physiological stress detection on the public WESAD dataset "
           "using subject-independent (leave-one-subject-out) cross-validation, and identifies "
           "a severe, previously hidden subject-dependent decision-threshold instability "
           "concealed by a strong pooled AUC (Section 4.4).")
add_bullet("Identifies photographic-source shortcut learning in the original Eczema-vs-Normal "
           "image classifier: a superficially strong 95.66% accuracy result reflected image "
           "source and lighting, not dermatological features (Sections 5.1-5.2).")
add_bullet("Conducts external, cross-dataset validation of the rebuilt image classifier on two "
           "independently sourced datasets, demonstrating severe zero-shot performance "
           "degradation (AUC dropping from 0.8644 internally to 0.5347 and 0.4865 externally) "
           "that the internal result alone did not reveal (Sections 5.6, 10.1).")
add_bullet("Evaluates three interventions against this generalization gap -- generic "
           "augmentation, natural-proportion multi-source fine-tuning, and equal-weighted "
           "multi-source fine-tuning -- and shows the last produces the strongest, though "
           "still partial, external recovery (SkinDisNet AUC reaching 0.6515; Sections 5.7, 10.2).")

add_bold_para("B. Methodological contributions")
add_bullet("Demonstrates that subject-specific test-time (personal-baseline) calibration "
           "substantially improves the deployed LightGBM stress model's stability and "
           "accuracy, while helping a more complex CNN only partially (Sections 4.5-4.6).")
add_bullet("Develops a same-source Eczema-versus-clinical-look-alike image classification "
           "task specifically designed to remove the photographic-source shortcut identified "
           "in the original task (Sections 5.3-5.4).")
add_bullet("Redesigns Stage C's fusion rule from a flat weighted average to a gated, "
           "trigger-modulated formula, reflecting the literature's framing of stress and "
           "sleep as flare triggers rather than independent severity signals (Section 6.2).")

add_bold_para("C. Negative findings")
add_bullet("Reports a negative result for wearable sleep-disruption detection on the AAUWSS "
           "dataset (near-chance leave-one-subject-out AUC of 0.4642), corroborated by two "
           "independent, literature-standard actigraphy formulas also scoring at chance on "
           "the same data, and excludes this component from the deployed pipeline rather "
           "than omitting or softening the result (Section 4.8).")
add_bullet("Reports that generic data augmentation does not meaningfully improve the image "
           "classifier's cross-dataset generalization, isolating real exposure to external-"
           "source data (not just augmentation) as the factor that does (Sections 5.7, 10.2).")

add_bold_para("D. Engineering contribution")
add_bullet("Implements a decision-level (late) multimodal fusion architecture combining the "
           "wearable- and image-derived scores, and demonstrates that this architecture runs "
           "correctly end to end on real model output (Section 6).")

add_bold_para("E. Identified research gap")
add_bullet("Explicitly identifies the missing paired, longitudinal, patient-level dataset -- "
           "linking wearable recordings, skin photographs, and clinician-assessed severity "
           "for the same people over time -- that would be required to genuinely validate "
           "the complete system, rather than merely its individual components (Sections 6.3, "
           "7, 9).")

d.save(PATH)
print("Part 2 done: contributions restructured into A-E categories.")
print("Deleted", len(to_delete), "old bullet paragraphs.")
