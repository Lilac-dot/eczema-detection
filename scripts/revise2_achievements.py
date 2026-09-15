"""Major revision pass 2, part A: expand 1.1 into 'What This Project Achieved'
(detailed A-E categories) and add 1.2 achievement summary table."""
import docx

PATH = "Multi_Sensor_Wearable_AD_Monitoring_Paper_2026-09-15_revised.docx"
d = docx.Document(PATH)
paras = d.paragraphs

def find(text, style=None):
    for p in paras:
        if p.text.strip() == text and (style is None or p.style.name == style):
            return p
    return None

h11 = find("1.1 Contributions", "Heading 2")
lit_h = find("2. Literature Review", "Heading 1")
assert h11 and lit_h

body = d.element.body
children = list(body)
start_idx = children.index(h11._p)
end_idx = children.index(lit_h._p)
old_block = children[start_idx:end_idx]
for el in old_block:
    el.getparent().remove(el)

def add(text, style="Normal"):
    return lit_h.insert_paragraph_before(text, style=style)

def bullet(text):
    return lit_h.insert_paragraph_before(text, style="List Bullet")

def bold(text):
    p = add(text)
    p.runs[0].bold = True
    return p

add("1.1 What This Project Achieved", style="Heading 2")
add(
    "This project developed, evaluated, stress-tested, and iteratively improved multiple "
    "components of a proposed multimodal eczema-monitoring framework using public "
    "datasets. The experiments revealed failure modes not visible from headline internal "
    "metrics -- subject-dependent physiological variability, photographic-source "
    "shortcut learning, severe cross-dataset image distribution shift, and failure of "
    "sleep-disruption detection on the available dataset -- and then tested concrete "
    "interventions for each, achieving measurable improvements in the corresponding "
    "tasks. A working software-level multimodal fusion pipeline was implemented on top "
    "of these components. The remaining gap is end-to-end clinical validation, which "
    "requires paired longitudinal patient data; this section details what was actually "
    "accomplished toward that goal."
)

bold("A. Experimental achievements")

bold("1. Wearable stress detection")
bullet("Implemented and evaluated a LightGBM-based physiological stress detector on WESAD, using the "
       "wrist modalities the intended wearable architecture actually targets -- EDA, skin temperature, "
       "BVP/PPG, and 3-axis acceleration -- under leave-one-subject-out cross-validation.")
bullet("Initial mean AUC was approximately 0.871. Detailed subject-level analysis, rather than accepting "
       "this pooled figure at face value, revealed severe threshold instability concealed within it.")
bullet("Developed and evaluated subject-specific personal-baseline calibration in response. Mean AUC "
       "improved to 0.9405; pooled F1 improved from 0.646 to 0.7602; mean per-subject F1 reached 0.635.")
bullet("One subject (S5) remained a difficult threshold-transfer case (F1 = 0) even after calibration -- "
       "a meaningful result in itself, since it demonstrates the project tested whether an aggregate "
       "score translated consistently across individuals rather than stopping at the aggregate.")

bold("2. Image classification")
bullet("Implemented an initial eczema-vs-normal image classifier, reaching 95.66% accuracy, and audited "
       "the result rather than accepting it: this audit discovered a photographic-source/brightness "
       "shortcut underlying the figure, demonstrating it was not trustworthy evidence of lesion "
       "recognition.")
bullet("Rebuilt the task using a same-source Eczema-vs-clinically-similar-diseases formulation, "
       "obtaining a final internal result of 81.07% accuracy, 81.18% F1, 79.92% precision, and 82.47% "
       "recall. Replacing a misleading 95.66% figure with a lower but defensible one is itself an "
       "achievement in experimental methodology -- the project prioritized trustworthiness over a "
       "higher but meaningless number.")

bold("3. Cross-dataset validation")
bullet("The rebuilt image model was tested zero-shot on two independently sourced datasets, SCIN and "
       "SkinDisNet, rather than stopping at internal validation. Internal AUC was approximately 0.8644; "
       "external AUC was 0.5347 on SCIN and 0.4865 on SkinDisNet.")
bullet("This directly demonstrated that strong within-source image performance does not imply "
       "cross-source generalization, exposing distribution shift as a major practical issue for "
       "dermatological image models -- one of the strongest methodological findings of the project, and "
       "a test most comparable published work in this space does not run at all.")

bold("4. Generalization interventions")
bullet("Three interventions were tested against the generalization gap: generic augmentation, "
       "natural-proportion multisource fine-tuning, and equal-weighted multisource fine-tuning.")
bullet("SkinDisNet AUC rose from approximately 0.4680 (zero-shot) to 0.6090 (natural-proportion "
       "multisource training) to 0.6515 (equal-weighted multisource training) -- the strongest result "
       "of any intervention. SCIN remained substantially weaker, reaching approximately 0.5794 at best.")
bullet("Generic augmentation alone was insufficient; real domain diversity produced substantially better "
       "external performance; equal weighting of sources produced the strongest recovery. The "
       "improvement was partial rather than complete, and internal performance decreased under "
       "multisource training, demonstrating a genuine, quantified trade-off between source-specific and "
       "externally robust performance -- an experimental contribution in its own right, not a footnote "
       "to the earlier negative result.")

bold("5. Sleep detection")
bullet("Tested a sleep-disruption detection hypothesis on AAUWSS: mean LOSO AUC 0.4642, pooled AUC "
       "0.3867, mean F1 0.1071. Two independent, literature-standard actigraphy baselines (Cole-Kripke, "
       "Sadeh) were run against the same data and also scored near chance, confirming the result was "
       "not an artifact of this project's modelling choices. The hypothesis was tested, it did not hold "
       "on the available data, the finding was independently corroborated, and the component was "
       "excluded from the fusion pipeline on that evidence.")

bold("B. Methodological achievements")
add("The project developed and demonstrated:")
bullet("Subject-specific baseline calibration for wearable stress detection.")
bullet("A same-source image classification formulation designed to reduce photographic-source shortcut "
       "learning.")
bullet("External cross-dataset validation as a required evaluation step, rather than relying on internal "
       "accuracy alone.")
bullet("Multisource training experiments specifically targeting domain shift.")
bullet("Decision-level multimodal fusion combining independently generated wearable and image scores.")
bullet("A software pipeline that runs the models and fusion functions end-to-end on real model outputs.")

bold("C. Engineering achievements")
add(
    "The project produced actual runnable software components, distinct from and not diminished by the "
    "absence of physical hardware: Stage A stress preprocessing and inference, the deployed LightGBM "
    "stress model, Stage B image preprocessing and classifier inference, and the fusion functions "
    "trigger_index(), flare_risk(), and fuse(), demonstrated end-to-end on real model outputs rather "
    "than synthetic placeholders. A software implementation of the full computational architecture "
    "exists and runs today; what remains is a separate, later engineering phase (Section 8)."
)

bold("D. Research and methodological insights")
add("The experiments produced the following findings:")
bullet("Aggregate performance can conceal individual-level failure.")
bullet("High image accuracy can be produced by dataset shortcuts rather than genuine recognition.")
bullet("Internal image performance can dramatically overestimate external performance.")
bullet("Real external-source diversity can partially improve generalization.")
bullet("Generic augmentation is not necessarily sufficient to solve domain shift.")
bullet("A negative result is strengthened when independent baselines reproduce it.")
bullet("Multimodal healthcare systems cannot be validated by combining independently trained models "
       "alone; paired patient-level data are necessary for end-to-end validation.")

bold("E. Dataset / research-gap achievement")
add(
    "The project systematically identified the data structure required for eventual end-to-end "
    "validation: wearable physiological signals, skin photographs, and clinician- or otherwise "
    "appropriately assessed eczema severity, collected longitudinally from the same patients and "
    "aligned in time. This is a concrete research requirement identified through the experiments "
    "themselves, not an assumption made in advance."
)

d.save(PATH)
print("1.1 expanded into 'What This Project Achieved' (A-E, detailed).")
