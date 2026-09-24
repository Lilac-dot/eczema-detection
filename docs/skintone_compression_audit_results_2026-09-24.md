# Skin-tone compression audit — results (2026-09-24)

Does compressing an eczema-vs-look-alike classifier for phone deployment cost patients with
darker skin more than patients with lighter skin?

All numbers are **out-of-fold cross-validation** results on the development split
(5-fold, case-grouped). The frozen test split (`dataset/skintone/manifest_test.csv`,
910 images / 614 cases) has **not** been opened.

## 1. Data

| Source | Clean images | Eczema share | Fitzpatrick coverage (development) |
|---|---|---|---|
| SCIN (consumer phone photos) | 1,741 | 54% | I–II 45%, III–IV 45%, V–VI 8% |
| Fitzpatrick17k-C (clinical atlases) | 2,034 | 19% | I–II 50%, III–IV 34%, V–VI 14% |
| DermaCon-IN (Indian clinics) | 2,328 | 15% | III–IV 71%, V–VI 29% |

Labels are harmonised to eczema vs. the 7 Stage B look-alike classes
(`scripts/skintone_labels.py`). The cleaning step removed 83 Fitzpatrick17k images that
duplicate the curated training set, and merged near-duplicates into one case group
(`scripts/clean_skintone_manifest.py`). 165 DermaCon-IN images (the largest files) could
not be downloaded on this connection and are missing.

## 2. Merging the sources hurts — one model per source

ResNet18, 5-fold CV, paired case-bootstrap on identical held-out images
(`scripts/compare_skintone_runs.py`):

| Source | Own-source model | Pooled | Pooled − own [95% CI] |
|---|---|---|---|
| SCIN | 0.653 | 0.594 | −0.058 [−0.092, −0.024] |
| DermaCon-IN | 0.829 | 0.827 | −0.003 [−0.025, +0.021] |
| Fitzpatrick17k | 0.787 | 0.755 | −0.031 [−0.059, −0.005] |
| ↳ Fitzpatrick17k FST V–VI | 0.825 | 0.720 | **−0.105 [−0.188, −0.027]** |

The literature-backed fixes did not recover SCIN:
- balanced subsampling with worst-source checkpoint selection: −0.043 [−0.079, −0.008]
- last-layer retraining (DFR): −0.070 [−0.110, −0.029]

This matches Compton et al. (MLHC 2023) and Shen et al. (MLHC 2024). In this setting,
naively pooling datasets also made the model **less fair**: FST V–VI in Fitzpatrick17k
lost 0.105 AUC.

## 3. Architectures (one model per source, curated-model initialisation)

| Model | SCIN | DermaCon-IN | Fitzpatrick17k |
|---|---|---|---|
| ResNet18 (reference) | 0.65 | 0.83 | 0.79 |
| ShuffleNetV2-0.5x | 0.63 | 0.83 | 0.71 |
| ShuffleNetV2-1.0x | 0.65 | 0.85 | 0.73 |
| SqueezeNet1.1 | 0.63 | 0.84 | 0.70 |

These are the architectures that were backend-stable in the compression study
(`papers/edge-ai-lightweight-deployment/`), plus ResNet18 as the reference.

## 4. Compression audit (`scripts/audit_skintone_compression.py`)

The INT8 used here is real integer inference on ARM (qnnpack, with the ReLU6
channels-last fix), 256 calibration images. Pruning is one-shot global L1 magnitude
pruning. "dark−light" is ΔAUC(FST V–VI) − ΔAUC(lightest group: I–II, or III–IV for
DermaCon-IN). A negative value means compression costs darker skin more. `*` marks a 95%
case-bootstrap CI that excludes 0.

| Model | Source | FP32 AUC | INT8 ΔAUC | INT8 dark−light [95% CI] | INT8 flips | Prune50 ΔAUC | Prune50 dark−light [95% CI] |
|---|---|---|---|---|---|---|---|
| ResNet18 | SCIN | 0.661 | +0.003 | -0.013 [-0.038, +0.010] | 4.7% | -0.008 | -0.002 [-0.062, +0.056] |
| ResNet18 | DermaCon-IN | 0.830 | -0.001 | -0.010 [-0.019, -0.002] * | 3.2% | -0.002 | -0.010 [-0.029, +0.011] |
| ResNet18 | Fitzpatrick17k | 0.784 | -0.002 | -0.006 [-0.022, +0.008] | 3.2% | -0.017 | -0.031 [-0.073, +0.008] |
| ShuffleNetV2-0.5x | SCIN | 0.632 | -0.031 | +0.010 [-0.139, +0.156] | 24.8% | -0.016 | +0.060 [-0.045, +0.161] |
| ShuffleNetV2-0.5x | DermaCon-IN | 0.831 | -0.042 | -0.003 [-0.038, +0.033] | 21.5% | -0.052 | +0.003 [-0.034, +0.041] |
| ShuffleNetV2-0.5x | Fitzpatrick17k | 0.701 | -0.032 | -0.021 [-0.084, +0.040] | 26.2% | -0.059 | -0.009 [-0.079, +0.061] |
| ShuffleNetV2-1.0x | SCIN | 0.659 | -0.029 | -0.052 [-0.124, +0.010] | 16.6% | -0.035 | -0.069 [-0.176, +0.035] |
| ShuffleNetV2-1.0x | DermaCon-IN | 0.849 | -0.019 | -0.003 [-0.030, +0.020] | 7.9% | -0.039 | +0.011 [-0.026, +0.048] |
| ShuffleNetV2-1.0x | Fitzpatrick17k | 0.724 | -0.015 | +0.027 [-0.014, +0.068] | 13.4% | -0.037 | -0.018 [-0.084, +0.048] |
| SqueezeNet1.1 | SCIN | 0.625 | -0.001 | -0.003 [-0.018, +0.012] | 4.3% | -0.037 | -0.055 [-0.154, +0.031] |
| SqueezeNet1.1 | DermaCon-IN | 0.837 | -0.002 | +0.000 [-0.004, +0.005] | 1.8% | -0.034 | -0.000 [-0.031, +0.028] |
| SqueezeNet1.1 | Fitzpatrick17k | 0.704 | -0.001 | -0.003 [-0.016, +0.013] | 3.5% | -0.040 | +0.052 [-0.020, +0.122] |

The full results cover all 7 variants: INT8 with mixed / light-only / dark-only
calibration, prune 30/50/70%, and prune-30 followed by INT8. They are in
`experiments/skintone_audit/audit_summary.json`. The FP32 AUCs differ slightly from
Section 3 because the audit excludes images with no skin-type label and runs FP32 on the
CPU.

### Findings

1. **There is no consistent, practically meaningful dark-skin penalty from standard
   compression.**
   - Of 84 model × source × variant cells, 3 have a CI entirely below zero (darker skin
     worse) and 1 entirely above zero. A 5% false-positive rate alone would produce about
     4 such cells.
   - The largest significant disparity is ResNet18 + INT8 on DermaCon-IN: −0.010 AUC.
2. **There is a small directional tendency.** 56 of 84 cells are negative, and the mean
   dark−light is about −0.006 AUC for the three INT8 variants. The cells are not
   independent (same models and images across variants), so the naive sign-test
   p = 0.003 overstates the evidence. Report it as a weak pooled trend, not a per-setting
   harm.
3. **Architecture matters far more than skin tone.** ResNet18 and SqueezeNet1.1 are
   essentially lossless under INT8 (|ΔAUC| ≤ 0.003, 2–5% of predictions flip). Both
   ShuffleNetV2 widths lose 0.015–0.042 AUC under INT8 on qnnpack and flip 8–26% of
   predictions. That cost is in every skin-type group.
4. **The skin tone of the INT8 calibration images makes no measurable difference.**
   Light-only, dark-only and mixed calibration are within about 0.01 of each other in
   every cell.
5. **Pruning becomes harmful for everyone at 70%** (−0.03 to −0.24 AUC). The disparity
   CIs at 50–70% are wide and change sign between models.
6. The only large dark-skin gaps (ShuffleNetV2-1.0x and SqueezeNet on SCIN, −0.05 to
   −0.12) all come from **SCIN FST V–VI: 59 cases, where the FP32 model is already near
   chance** (AUC 0.53–0.58). They are too noisy to interpret.

### Mechanism checks (`scripts/explain_skintone_audit.py`)

- **Model uncertainty (Iofinova et al. 2023; Tran et al. NeurIPS 2022): not supported.**
  The FP32 decision margin is equal or **larger** for FST V–VI in every model × source.
  The models are not less certain on darker skin.
- **Representation (Hooker et al. 2019/2020): not distinguishable.** No consistent
  disparity exists to explain, and it doesn't track the V–VI share (SCIN 8%,
  Fitzpatrick17k 14%, DermaCon-IN 29%).

## 5. What this means for the paper

The headline is a mostly **reassuring, well-powered-where-possible null**:

> Standard deployment compression (INT8 on ARM, moderate pruning) of eczema classifiers
> does not measurably widen the skin-tone performance gap across three datasets and four
> architectures. Architecture choice dominates, and naive multi-dataset pooling harmed
> darker-skinned patients far more (−0.105 AUC) than any compression setting did.

**Limitations**
- The FST V–VI sample is small in SCIN (59 cases).
- DermaCon-IN has no FST I–II.
- Fitzpatrick17k skin types come from crowd annotators.
- FP32 performance on SCIN is modest (AUC about 0.63–0.66).
- Only qnnpack was tested (no x86 fbgemm on this machine).
- Everything so far is cross-validation. The test split still needs a pre-registered
  single evaluation.
