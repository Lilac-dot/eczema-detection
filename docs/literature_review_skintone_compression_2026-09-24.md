# Systematic literature search — compression × skin-tone fairness (2026-09-24)

**Question:** has anyone measured how *standard* deployment compression (INT8 quantization,
magnitude pruning) changes an image classifier's performance across skin tones, especially
in dermatology?

Reproducible script: `scripts/systematic_search_skintone_compression.py`. Raw records,
screening lists and the query log are in `docs/lit_search_2026-09-24/`.

## Method (PRISMA-style)

**Identification**
- 16 keyword queries on the Semantic Scholar Graph API (up to 100 results each).
- 7 fielded abstract queries on arXiv.
- Forward citations (every paper citing) of 7 seed papers:
  - Hooker 2019, Hooker 2020
  - Iofinova 2023
  - Stoychev & Gunes 2022
  - FairPrune 2022
  - FairQuant 2026
  - Groh 2021 (Fitzpatrick17k)

The first run silently returned 0 results for 11 queries because of API rate limiting. The
script was changed to record failures explicitly and cache successful calls, then re-run
until **0 requests failed** (3 retry passes).

**Automated screening (title + abstract)**
- A record passes if it mentions a compression term **and** a fairness/subgroup term.
- It is then split by whether it also mentions a medical, skin or skin-tone term.

**Eligibility:** manual review of the shortlist. A paper counts as a direct competitor only
if it *measures the change in per-skin-tone (or per-demographic) performance caused by
standard compression, in dermatology.*

| Stage | Records |
|---|---|
| Identified (all sources) | 2,368 |
| After de-duplication | 1,993 |
| Pass compression + fairness + medical/skin screen | 45 (40 unique titles) |
| Pass compression + fairness only (general background) | 110 |
| Manually eligible as related work | ~30 (listed below) |
| **Direct competitors (per-skin-tone audit of standard compression in dermatology)** | **0** |

**Known gap in the search:** FairQuantize (MICCAI 2024) was not returned by any query,
because Semantic Scholar did not index it under these terms. It was added by hand from the
earlier manual review. Other MICCAI/Springer papers could be missing the same way, so
before submission check the MICCAI 2024–2026 proceedings and Google Scholar directly.

## Related work found, by group

### A. Dermatology: compression as a *fairness method* (mitigation, not audit)
- FairPrune (MICCAI 2022)
- Channel pruning for fairness (MICCAI 2024)
- FairQuantize (MICCAI 2024)
- Skin-tone normalization + channel pruning (2025)
- FairLRF (2025)
- FairCompressAgent (2026, FPGA)
- BID-Net (ICASSP 2025, distillation)
- **FairQuant (Woergaard & Selvan, 2026): closest.**
  - Fitzpatrick17k (114-class) + ISIC, ResNet18/50 and small ViTs.
  - Reports a uniform 8-bit baseline as average and worst-group accuracy only.
  - Quantization is **simulated (fake-quant)**, with **no per-skin-type breakdown**, no
    pruning and no mobile CNNs.

→ All of these *propose methods*. None isolates what ordinary deployment compression does
to each skin-type group.

### B. Fairness + efficiency on devices in dermatology
- **ESFair 2023 challenge (ESWEEK 2023).** Teams optimised accuracy + fairness (across
  skin-tone subgroups) + latency on embedded hardware, on a dermatology dataset. It is a
  joint-optimisation *competition*, not a controlled compression audit.
- Li et al. (ICIST 2023): an ESFair entry using distillation + noise augmentation.
- Edge teledermatology via distillation, not fairness audits: MTAKD (Sci. Rep. 2025) and
  Diverse Representation KD (IEEE Access 2025).
- Skinclusive AI (thesis) and SkinGuardian: skin cancer, fairness-trained, then quantized
  for deployment.

### C. General compression-bias audits (not dermatology)
- Hooker et al. 2019 and 2020: "compression identified exemplars", CelebA.
- Tran et al., NeurIPS 2022, *Pruning has a disparate impact on model accuracy*. The
  mechanism is gradient norms and distance to the decision boundary.
- Iofinova et al., CVPR 2023: pruned vision models, uncertainty-linked bias.
- Stoychev & Gunes 2022: facial expression, gender.
- Compressed models and race bias in face recognition (BIOSIG 2023).
- *Explaining How Quantization Disparately Skews a Model* (2025): PTQ lowers logit variance
  and hurts minority groups.
- FairQuanti (ACM TOPS 2025).
- Quantization bias in vision-language models (2024).
- Mobile accuracy–fairness–efficiency trilemma (Sci. Rep. 2026): mobile, gender attribute,
  found that magnitude pruning is Pareto-dominated by fairness-aware pruning.

### D. Medical-imaging quantization in general
- *Quantization of DNNs for Medical Image Analysis: A Systematic Review and Meta-Analysis*
  (Technologies 2026, 72 studies, PRISMA). Its gap list covers interoperability, energy,
  governance and regulation. **Fairness and subgroup effects are not among them.**

### E. Pooling datasets
- Compton et al. (MLHC 2023) and Shen et al. (MLHC 2024): adding data sources can hurt
  worst-group performance.
- No dermatology skin-tone evidence of this was found.

## Novelty verdict

Within the searched literature, **no study audits the per-skin-type effect of standard
deployment compression on a dermatology classifier.** The following elements of this
project have no direct precedent:

1. Per-Fitzpatrick-group ΔAUC with case-level CIs for **real INT8 kernels on ARM**
   (qnnpack) and magnitude pruning, not simulated quantization.
2. **Inflammatory disease** (eczema vs look-alikes). All prior dermatology
   compression-fairness work is skin cancer or 114-class Fitzpatrick17k.
3. **Mobile, backend-stable CNNs** (ShuffleNetV2, SqueezeNet) alongside ResNet18.
4. **Skin tone of the INT8 calibration set** as an experimental variable (analogues exist
   only for LLMs).
5. Three independent datasets with case-grouped cross-validation, and the finding that
   **naive dataset pooling cost FST V–VI −0.105 AUC**, far more than compression did.
6. Mechanism checks against Hooker (representation) and Iofinova/Tran (uncertainty/margin).

**How to position it:** as an empirical audit that complements the mitigation literature
(group A) and the ESFair joint-optimisation setting (group B). Its main answer is that
standard INT8 and moderate pruning do **not** measurably widen the skin-tone gap here,
while architecture and dataset composition matter more. Frame it as a test of Hooker's and
Tran's predictions in a new, clinically relevant setting, not as the discovery of
compression bias.

**Must cite:** FairQuant, FairQuantize, FairPrune, ESFair 2023, Hooker 2019/2020, Tran
2022, Iofinova 2023, Compton 2023, and the 2026 medical-quantization systematic review.
