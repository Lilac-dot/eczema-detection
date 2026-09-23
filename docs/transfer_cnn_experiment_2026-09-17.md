# EfficientNet-B0 vs. MobileNetV3-Small — Transfer Learning Comparison

## 1. Existing project inspection (done before any new code was written)

**Dataset location.** The existing, most-trustworthy Stage B image classifier
(`models/curated_resnet18_balanced.pt`, documented in
`docs/curated_eczema_vs_disease_2026-08-26.md`) is trained on
`SkinDisease/manifest_curated_v3_{train,val,test}.csv` — Eczema vs. 7 clinically similar
diseases (Psoriasis, Tinea, Candidiasis, Infestations_Bites, Lichen, DrugEruption, Rosacea),
merged from 3 separate Kaggle/DermNet-style photo archives
(`scripts/merge_all_eczema_sources.py`).

**Number of images / class distribution.** 3,330 total: 1,665 Eczema (label 1), 1,665
Other (label 0) — a deliberately rebalanced ~50/50 split (see that doc's "merging a third
Eczema source and rebalancing to 50/50" section). Full per-split, per-class breakdown is
recorded in `experiments/dataset_split_report.json`.

**Existing preprocessing.** 224×224 resize, ImageNet mean/std normalization. Training
augmentation: horizontal flip, ±15° rotation, mild color jitter. Validation/test:
deterministic resize + normalize only (`scripts/train_curated_cnn_balanced.py`).

**Existing train/val/test split.** Already exists: `manifest_curated_v3_{train,val,test}.csv`,
70/15/15, stratified by the 8-way `disease_class` column, fixed seed 42
(`scripts/merge_all_eczema_sources.py`). Counts: 2,327 train / 496 val / 507 test.

**Patient/subject IDs.** **Not available.** All three merged source datasets are
Kaggle/DermNet-style photo archives with per-image files only (`path`, `disease_class`,
`label` columns) — no patient or subject identifier anywhere in the pipeline. (Patient-level
IDs do exist elsewhere in this project — the separate SkinDisNet external-validation dataset
has 416 real patient IDs from a clinical hospital source — but SkinDisNet is not part of this
training set.) Per the protocol's own fallback rule, the existing split already uses "the
safest reproducible image-level split possible": deterministic seed, stratified by class.

**Existing model architecture.** ResNet18 (ImageNet-pretrained), `layer4` + `fc` unfrozen,
single-stage fine-tuning (not two-stage), plain `nn.Linear(512, 2)` head (2-class softmax,
not a single sigmoid unit), Adam optimizer with **no weight decay**, **no dropout** in the
head. Data augmentation + best-val-checkpoint selection are the only regularization
currently in place.

**Existing training/eval scripts.** `scripts/train_curated_cnn_balanced.py` (training),
`scripts/eval_curated_cnn_balanced.py` (evaluation — this script *does* compute test-set
metrics directly, since it was written for a prior, already-completed experiment that
predates this test-isolation protocol; it is left untouched, but the new experiment below
uses entirely separate code that never opens the test manifest).

**Decision: reuse the existing split, do not build a new one.** It already satisfies
everything this protocol asks for in the no-patient-ID case (deterministic seed, class-
stratified, image-level). Rebuilding it would only add a second, less-comparable split and
make this experiment not directly comparable to the existing ResNet18 baseline. Recorded
verbatim in `experiments/dataset_split_report.json`.

## 2. Class imbalance

Train: 1,165 Eczema / 1,162 Other. Val: 249 Eczema / 247 Other. Already effectively 50/50
in both splits that this experiment can see. **No class weighting or resampling applied** —
none is needed, and applying it to already-balanced data would be an unjustified extra
knob. (Test-split balance, 251/256, was **not** consulted in making this decision — this
was inspected in the existing doc from 2026-08-26, before the current test-isolation
protocol was adopted, but was not re-examined or used for anything in this new experiment.)

## 3. Test-set isolation

`scripts/train_transfer_cnn.py` opens exactly two manifest files:
`manifest_curated_v3_train.csv` and `manifest_curated_v3_val.csv`. The path to
`manifest_curated_v3_test.csv` does not appear anywhere in that file — there is no code
path, flag, or debug branch that could load it. No test accuracy, loss, F1, AUC, or
confusion matrix has been computed for either model **inside this script**.

**Correction, 2026-09-18**: this no longer describes the full picture. The test manifest
*was* subsequently opened outside this script, for a finalist comparison across the models
that had finished training at that point — see
`docs/transfer_cnn_test_set_access_2026-09-18.md` for what was accessed, why it happened
before it should have, and the guardrail now in place for the rest of this queue. Treat
this section as historical (accurate as of 2026-09-17, when only training had happened),
not as a current guarantee.

## 4. Architecture (both models, kept identical apart from the backbone)

```
Input -> preprocessing -> [moderate train-only augmentation] -> ImageNet-pretrained
backbone -> GlobalAveragePool (built into the backbone's own forward pass) ->
Linear(-> 128) -> ReLU -> Dropout(0.3) -> Linear(128 -> 1) -> Sigmoid
```

Trained with `BCEWithLogitsLoss` on the raw logit (numerically stable equivalent of
Sigmoid + BCELoss; sigmoid is applied separately wherever a probability is needed for
metrics/plots).

- **EfficientNet-B0**: `torchvision.models.efficientnet_b0`, ImageNet1K weights.
- **MobileNetV3-Small**: `torchvision.models.mobilenet_v3_small`, ImageNet1K weights.

## 5. Two-stage training strategy

**Stage 1 — frozen backbone.** All backbone parameters frozen; only the new head trains.
AdamW, lr=1e-4, weight_decay=1e-4. Frozen blocks' BatchNorm layers are explicitly kept in
`eval()` mode during training (otherwise their running stats keep drifting even though
their weights don't update — the same fix already used in `train_curated_cnn_balanced.py`).

**Stage 2 — partial fine-tuning.** Unfreezes the *last two feature blocks* of each
backbone — roughly the last ~15-20% of network depth for both architectures, picked to be
comparable across the two (EfficientNet-B0: blocks 7-8 of 9, its last MBConv stage + final
1×1 conv; MobileNetV3-Small: blocks 11-12 of 13, its last InvertedResidual + final conv).
Not the entire backbone — the earlier layers encode generic low-level features (edges,
textures) that transfer well from ImageNet and don't need to be re-learned on ~2,300
training images; unfreezing everything would add far more trainable parameters than this
dataset size can support without overfitting. AdamW, lr=1e-5, weight_decay=1e-4 (same
decay as stage 1).

Both stages: `ReduceLROnPlateau` (factor 0.5, patience 2, monitors val loss), early
stopping (patience 5 on val loss, best checkpoint restored), max 25 epochs/stage as an
upper bound (expected to stop earlier via early stopping in practice). A Windows free-RAM
gate (`scripts/mem_guard.py`, already used elsewhere in this project) checks available
memory before every epoch and halts cleanly rather than risk an unclean OOM kill on this
8GB machine.

## 6. Results

See `experiments/efficientnet_b0/config/config.json` and
`experiments/mobilenetv3_small/config/config.json` for full machine-readable results, and
`experiments/<model>/plots/` for training curves, validation confusion matrices and ROC
curves. Summary table and written analysis to follow once both training runs complete.

**TEST SET HAS NOT BEEN ACCESSED — as of when this section was written (2026-09-17).**
It was subsequently accessed, prematurely (before the rest of the model queue had
finished), on 2026-09-18. See `docs/transfer_cnn_test_set_access_2026-09-18.md` for the
full account, the results, and the guardrail adopted for the remaining candidates.
