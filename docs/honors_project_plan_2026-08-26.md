# Honors Project Plan — 2026-08-26

Based on everything established this session across `Progress_Log_2026-08-26.docx` and
the dated docs in `docs/`. Three real decision points need your (and your advisor's)
input, marked **DECISION** below — the rest is a proposed sequence I'd start on.

## Where things actually stand right now

- **Stage A (motion/scratch):** WISDM cleaned, first model built (LightGBM, teeth-brushing
  proxy, subject-level split, test AUC 0.83, modest F1 ~0.25). A paper you found (Chun et
  al. 2021, ADAM sensor) gave a specific, literature-backed reason the ceiling is low:
  scratch's real signal is 100–800 Hz, WISDM samples at 20 Hz. No public scratch-labelled
  dataset exists anywhere (checked a systematic review + 2 papers directly — all say "data
  on request" at best).
- **Stage B (image severity):** Original Eczema/Normal set (2,757 images) cleaned, two
  models trained (CNN 95.66%, LightGBM 95.18%) — both proven to be inflated by a
  photo-source shortcut, not real lesion detection. Two candidate fixes (brightness
  normalization, background cropping) were both tested and both failed to remove it. A
  second dataset (`SkinDisease/`, 20 disease classes incl. Eczema, DermNet-style
  same-source images) looks like it could genuinely fix this for Eczema-vs-other-disease
  — currently finishing a corruption/duplicate cleaning pass on it. Its own "Normal" class
  is separately confirmed contaminated (not real skin content, inconsistent across its own
  train/test split) and shouldn't be used.
- **Stage C (fusion):** not started — blocked on A and B each producing a trustworthy
  per-session output first.
- **Ethics/IRB:** unchanged from the original report — still the long-lead-time item
  blocking real patient data, and now also blocking any self-collected scratch pilot.

## Proposed sequence

### 1. Finish Stage B's rebuild (in progress, days not weeks)
- Finish cleaning `SkinDisease/` (corruption/duplicate check running now, plus the
  cross-split duplicate check specifically, since a train/test leak would break
  evaluation validity).
- **DECISION: which classes count as "other skin condition"?** All 19 non-Eczema classes
  lumped together, or a curated subset that's actually visually similar to eczema
  (candidates: Psoriasis, Lichen, Seborrh_Keratoses, Rosacea, Tinea, Candidiasis)? A
  curated subset is the more defensible differential-diagnosis framing; "everything else"
  is easier but risks the "other" class being trivially separable just from being
  heterogeneous, which would be a new, milder version of the same shortcut problem.
- Build Eczema-vs-other-disease using the same dual-model approach already established
  (ResNet18 CNN + colour-features/LightGBM) — the LightGBM cross-check is exactly the
  tool that already caught the last shortcut, so keep running it as a sanity check on any
  new result, not just the CNN.
- **DECISION: what happens to "Normal"?** Recommendation: drop it from Stage B entirely
  given the contamination evidence, and state that explicitly as a scoped-out limitation
  in the paper, backed by the investigation you already have documented. Revisit only if
  a properly single-sourced normal-skin set turns up later.

### 2. Decide the Stage A path
**DECISION: how much further to invest in Stage A given the sampling-rate ceiling?**
Three options, not mutually exclusive:
- (a) **Ship the WISDM model as a documented proxy/pipeline demonstration**, citing the
  ADAM paper's finding as a specific, literature-grounded limitation rather than a vague
  "needs more data" caveat. Lowest effort, defensible, but the Stage A result stays weak.
- (b) **Request real data from SIGMA/ADAM authors** — both explicitly said data is
  available on request. Costs an email, could stall or go nowhere, but is close to free
  to attempt in parallel with anything else.
- (c) **Scope a small self-collected pilot** (a few volunteers, scratch vs. a few control
  gestures) — the strongest option, but needs a higher-sampling-rate device than a typical
  smartwatch to actually capture the discriminating signal, and needs IRB sign-off, which
  is the same bottleneck already blocking real patient data. This is the one worth
  starting the paperwork conversation for now, regardless of which option you pick, since
  it's the longest lead-time item in this entire plan.

### 3. Stage C (fusion) — once A and B each have a settled output
Not enough is fixed yet to design this in detail. Once Stage B's Eczema-vs-other-disease
model exists and Stage A's path is decided, this needs its own short design pass:
what per-session summary each stage actually hands off (a severity score from B, a
scratch-frequency/intensity estimate from A), and a first fusion model (gradient
boosting, consistent with what's worked well elsewhere in this project).

### 4. Writing/framing
Three things worth building the paper's methodology section around, since they're
genuinely stronger content than a single clean headline number:
- The shortcut-learning diagnosis itself — two independent model families agreeing,
  two candidate fixes tested and ruled out with evidence, not assumption. This is
  exactly the kind of dataset-validity rigor Maulana et al. (2024) flagged as missing
  from their own and prior work.
- The Stage A sampling-rate limitation, backed directly by Chun et al. (2021) rather than
  asserted generically.
- Explicit scoping decisions (dropping "Normal," choosing a disease-class subset) stated
  as deliberate, evidenced choices — not silently made and left for a reader to notice.

## Suggested near-term priority order

1. Finish `SkinDisease` cleaning (already running).
2. You + advisor: decide the two Stage B DECISIONs above (class subset, Normal's fate).
3. Build Eczema-vs-other-disease (both models) — this is the highest-payoff, lowest-effort
   win available right now.
4. Start the IRB conversation in parallel — it's the slowest-moving piece, so it shouldn't
   wait for 1–3 to finish.
5. Decide the Stage A path once IRB timeline is clearer (an IRB approval that's already
   in motion changes whether option (c) is realistic on your timeline).
6. Stage C design pass.
7. Writing pass.
