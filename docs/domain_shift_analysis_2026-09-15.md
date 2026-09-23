# Domain Shift Analysis: Internal vs. SCIN vs. SkinDisNet — 2026-09-15

## Why

`docs/external_validation_2026-09-15.md` established that the deployed Stage B model
(`curated_resnet18_balanced.pt`) fails to generalize zero-shot to SCIN and SkinDisNet.
`docs/external_generalization_improvement_2026-09-15.md` found that multi-source
fine-tuning helps SkinDisNet substantially (AUC 0.468→0.652) but SCIN barely at all
(AUC 0.554→0.579). This doc asks: can a formal, quantified measure of how different
these datasets are from Internal *explain* that asymmetry, rather than just restating
that it exists? This is the missing piece for treating the generalization findings as
a research contribution about domain shift, not only a negative result.

## Method

**Embeddings**: for every image in Internal (3,330, `manifest_curated_v3.csv`), SCIN
(976, `manifest_scin.csv`), and SkinDisNet (1,710, `manifest_skindisnet.csv`) — 6,016
images total — extracted the 512-dim penultimate-layer feature vector from the
**deployed** `curated_resnet18_balanced.pt` (no retraining; this is the exact
representation the model's own classifier head acts on). `scripts/extract_domain_embeddings.py`,
cached to `dataset/embeddings_internal_scin_skindisnet.npz`.

**Proxy A-distance** (Ben-David et al. 2010, standard domain-adaptation divergence
measure): for each dataset pair, trained a LightGBM binary classifier on the embeddings
to predict *which dataset an image came from* (70/30 stratified split, seed 42, out-of-sample
test accuracy only). A-distance = 2×(1 − 2×err). 0 = indistinguishable domains, 2 =
perfectly separable. 95% CI via bootstrap over the test set's per-image correctness
(1,000 resamples), same pattern as `eval_external_common.py`. `scripts/domain_shift_analysis.py`.

**3-way origin classifier**: same embeddings, single LightGBM model predicting
Internal/SCIN/SkinDisNet (chance ≈ 0.333), same train/test protocol, confusion matrix
reported so it's clear which pairs get confused with which.

**Image statistics**: extended the brightness-only comparison from
`docs/external_validation_2026-09-15.md` with saturation, native resolution, aspect
ratio, and a high-frequency-energy (Laplacian variance) proxy, computed on the exact
same 6,016-image population. `scripts/domain_image_stats.py`.

**Visualization**: 2D PCA of the same embeddings, colored by source. `scripts/domain_shift_analysis.py`
→ `docs/domain_shift_pca_2026-09-15.png`.

None of this touches the eczema-task train/val/test split — "labels" here are dataset
origin, an entirely different classification problem from Stage B's own task.

## Results

### Pairwise proxy A-distance

| Pair | Classifier test acc | Proxy A-distance (95% CI) |
|---|---|---|
| Internal vs. SCIN | 0.9481 | 1.793 (1.743–1.842) |
| Internal vs. SkinDisNet | 0.9729 | 1.892 (1.860–1.923) |
| SCIN vs. SkinDisNet | 0.9677 | 1.871 (1.816–1.916) |

All three pairs are close to the maximum possible value (2.0) — every dataset pair is
almost perfectly separable in this embedding space, not just "somewhat different."

### 3-way origin classifier

Accuracy 0.9474 (95% CI 0.9368–0.9573) against a chance baseline of 0.333.

Confusion matrix (rows = true, columns = predicted; order Internal/SCIN/SkinDisNet):

| | →Internal | →SCIN | →SkinDisNet | Recall |
|---|---|---|---|---|
| Internal (n=999) | 969 | 12 | 18 | 97.0% |
| SCIN (n=293) | 38 | 244 | 11 | 83.3% |
| SkinDisNet (n=513) | 12 | 4 | 497 | 96.9% |

SCIN is the least cleanly identified class (83.3% recall) and its errors go almost
entirely toward Internal (38 of 293), not SkinDisNet (11). SkinDisNet is identified
almost as cleanly as Internal itself.

### Image statistics

| Dataset | n | Brightness | Saturation | Median W×H | Aspect ratio | HF energy |
|---|---|---|---|---|---|---|
| Internal | 3,330 | 0.4267 | 0.3723 | 720×478 | 1.241 | 0.006588 |
| SCIN | 976 | 0.4915 | 0.3149 | 810×1080 | 0.877 | 0.005769 |
| SkinDisNet | 1,710 | 0.5462 | 0.2212 | 512×512 | 1.000 | 0.006318 |

Three datasets, three different acquisition conventions, visible in raw geometry alone:
Internal is landscape-oriented clinical macro photography (720×478, aspect 1.24), SCIN
is portrait-oriented consumer smartphone photography (810×1080, aspect 0.88 — phones
held vertically), SkinDisNet is uniformly square-cropped (512×512, aspect 1.00 exactly
— a preprocessing artifact of that dataset's release, not a photography convention).
Saturation decreases monotonically Internal > SCIN > SkinDisNet.

### PCA visualization

`docs/domain_shift_pca_2026-09-15.png` — PC1 explains 8.3% and PC2 4.8% of the 512-dim
variance (13.1% combined) — a small slice of a high-dimensional space, so the plot is
a partial, qualitative picture, not the basis for the A-distance numbers above (those
use the full 512 dimensions). Within that caveat, the three sources form visually
distinguishable, overlapping-at-the-edges clusters, consistent with the high classifier
accuracies above (near-perfect separability in 512-d rarely means zero overlap when
compressed into just 2 dimensions).

## Does this predict the fine-tuning asymmetry? — No, and the direction is backwards

This is the key check this analysis was for, and the honest answer is that **it does
not hold, and where it points, it points the wrong way.**

By raw pairwise A-distance, Internal-vs-SkinDisNet (1.892) is *more* separable than
Internal-vs-SCIN (1.793) — SkinDisNet looks like the more different, harder-to-bridge
domain. But `external_generalization_improvement_2026-09-15.md` found the opposite
in practice: fine-tuning closed much more of the gap for SkinDisNet (AUC 0.468→0.652)
than for SCIN (0.554→0.579, and that gain was mostly a threshold shift, not ranking
improvement per that doc's own analysis). The 3-way confusion matrix shows the same
pattern from a different angle: SCIN is the class most often confused *with Internal*
(38/293, recall only 83.3%), while SkinDisNet is barely confused with Internal at all
(12/513, recall 96.9%) — by this measure SCIN looks *closer* to Internal, not farther,
yet SCIN is the one that didn't respond to having its own training images added to the
fine-tuning mix.

**This is a real, reportable inconsistency, not a bug to explain away.** A plausible
(not proven) reading, supported by the image-statistics table: the embedding-space
separability measured here may be dominated by nuisance acquisition-pipeline
differences — SCIN's portrait aspect ratio and lower saturation, SkinDisNet's exact
512×512 crop — that a CNN trained on Internal's landscape clinical photos will latch
onto easily (making the origin classifier's job easy for all three pairs, hence all
three A-distances sitting near the 2.0 ceiling) without those differences necessarily
being the same shift that determines whether *task-relevant* (lesion-appearance)
features transfer under fine-tuning. If that reading is right, raw pooled-embedding
A-distance is measuring "how different do these photos look overall" rather than "how
hard is it to adapt the eczema-relevant part of the representation" — those can
diverge, and this result is evidence that they did here. This is a hypothesis this
analysis surfaces, not one it proves; distinguishing the two would need a
task-conditional divergence measure (e.g. computed on gradients or on a
task-fine-tuned subspace) rather than the plain penultimate-layer embedding used here.

## What this does and doesn't establish

- **Establishes**: all three datasets are highly, almost trivially separable in the
  deployed model's own feature space (A-distance 1.79–1.89 of a possible 2.0) — the
  cross-dataset generalization failure in `external_validation_2026-09-15.md` is
  consistent with a real, large, measurable representational gap, not a fluke of a
  couple of unlucky test images.
- **Does not establish**: that this raw separability measure predicts *which* external
  dataset will respond to fine-tuning — the one clean test of that (SkinDisNet vs. SCIN)
  came out backwards. Anyone citing proxy A-distance as a predictor of adaptability in
  this project's future work should know this counterexample exists.
- **Not tested here**: a task-conditional or fine-tuning-trajectory-based divergence
  measure, which might resolve the inconsistency above. Also not tested: whether the
  aspect-ratio/resolution difference is doing the work directly (e.g. by re-running the
  A-distance classifier on center-cropped, resolution-normalized images) — a concrete,
  cheap follow-up if this thread continues.

## Files

New: `scripts/extract_domain_embeddings.py`, `scripts/domain_image_stats.py`,
`scripts/domain_shift_analysis.py`, `dataset/embeddings_internal_scin_skindisnet.npz`
(6,016×512 cached embeddings, ~12MB), `dataset/domain_image_stats_summary.npz`,
`docs/domain_shift_pca_2026-09-15.png`, this file. Nothing existing was modified —
`curated_resnet18_balanced.pt`, all manifests, and `eval_external_common.py` were only
read, never written.
