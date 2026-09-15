"""Revision: replace Section 9 (Conclusion and Future Work) with a confident,
evidence-driven conclusion and a ranked future-work list."""
import docx

PATH = "Multi_Sensor_Wearable_AD_Monitoring_Paper_2026-09-15_revised.docx"
d = docx.Document(PATH)
paras = d.paragraphs

def find(text, style=None):
    for p in paras:
        if p.text.strip() == text and (style is None or p.style.name == style):
            return p
    return None

h9 = find("9. Conclusion and Future Work", "Heading 1")
appendix_h = find("Appendix A: Complete Model and Results Reference", "Heading 1")
assert h9 and appendix_h

body = d.element.body
children = list(body)
start_idx = children.index(h9._p)
end_idx = children.index(appendix_h._p)
old_section9 = children[start_idx + 1:end_idx]  # keep the '9.' heading itself
for el in old_section9:
    el.getparent().remove(el)

def add(text, style="Normal"):
    return appendix_h.insert_paragraph_before(text, style=style)

def add_bullet(text):
    return appendix_h.insert_paragraph_before(text, style="List Bullet")

add(
    "This project established a validated development pipeline for several individual "
    "components and experimentally identified the main barriers preventing their "
    "integration into a clinically validated multimodal system. Each component was "
    "checked, not merely built: a strong pooled result was tested for hidden "
    "instability (Section 4.4), a suspiciously high accuracy was tested for shortcut "
    "learning (Section 5.2), an internally strong classifier was tested for cross-"
    "dataset generalization (Section 5.6), and a sleep-detection attempt was tested "
    "against independent baselines before being accepted as a genuine negative result "
    "(Section 4.8). In summary:"
)
add_bullet("The wearable stress model improved substantially with personal-baseline calibration "
           "(mean AUC 0.871 to 0.9405), though one subject (S5) still fails and the model detects "
           "general physiological stress, not eczema-specific signals (Section 4).")
add_bullet("Shortcut learning was discovered in the original Eczema-vs-Normal image classifier and "
           "removed by rebuilding the task from a same-source archive (Sections 5.1-5.4).")
add_bullet("External validation revealed severe domain shift: the rebuilt classifier's 81.07% "
           "internal accuracy does not transfer, dropping to chance-level discrimination on two "
           "independent datasets (Section 5.6).")
add_bullet("Multisource training partially improved this generalization, most clearly for "
           "SkinDisNet (AUC 0.4680 to 0.6515), without closing the gap to internal performance "
           "(Section 5.7).")
add_bullet("Sleep detection failed on the available data and was correctly excluded from the "
           "deployed pipeline rather than hidden or reframed as a success (Section 4.8).")
add_bullet("Fusion was implemented and demonstrated end to end as a proof of concept only, with no "
           "paired patient data to train or validate it against (Section 6).")
add_bullet("The key missing ingredient, named explicitly rather than assumed away, is paired "
           "longitudinal patient data linking wearable signals, skin photographs, and clinician-"
           "assessed severity for the same people over time (Sections 6.3, 7.8).")
add_bullet("The next engineering step is physical wearable validation: building the wrist sensor "
           "node and testing it against the Empatica E4 reference device (Section 8.6).")
add_bullet("The next clinical research step is pursuing IRB approval to collect a small paired "
           "longitudinal dataset -- the only route to validating the complete system rather than "
           "its individual components.")

add("Future Work", style="Heading 2")
add(
    "Ranked by dependency and tractability, not by ambition -- later priorities depend on earlier "
    "ones, and none of them should be read as more urgent than the sequence implies."
)
add_bullet("Priority 1: Build the wrist sensor node proposed in Section 8.3.")
add_bullet("Priority 2: Compare the custom hardware's raw signals against the Empatica E4 under a "
           "shared protocol (Section 8.6).")
add_bullet("Priority 3: Test whether the WESAD-trained stress model, with its existing "
           "personal-baseline calibration procedure, transfers to the new hardware's actual signal "
           "characteristics.")
add_bullet("Priority 4: Continue improving external dermatology-image generalization -- the "
           "untested volume hypothesis (more real external training examples, not just reweighting "
           "the existing amount) is the most direct next lever (Section 5.7).")
add_bullet("Priority 5: Obtain IRB approval and collect a small paired longitudinal dataset: "
           "wearable signals, skin photographs, clinician- or patient-reported severity assessment, "
           "and timestamps linking them to the same individuals over time.")
add_bullet("Priority 6: Only after paired data exists, train and validate Stage C -- and only then "
           "would fitting component-level clinical indices (e.g. SCORAD-style extent/intensity/"
           "symptom sub-scores) to model outputs become a meaningful exercise, since it requires "
           "exactly the labelled data this project does not have.")
add_bullet("Priority 7: Lesion-vs-surrounding-skin thermal imaging (Section 8.7), as a longer-term "
           "direction requiring new hardware and IRB approval, and explicitly not the immediate "
           "next step relative to Priorities 1-2.")

d.save(PATH)
print("Section 9 rewritten: confident conclusion + ranked future work.")
