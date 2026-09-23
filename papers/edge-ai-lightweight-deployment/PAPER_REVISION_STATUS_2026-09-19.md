# IEEE Paper Revision — Status and Handoff (2026-09-19, stopped mid-task by user request)

This documents exactly what was done, what's mid-flight, and what hasn't started, for the
requested publication-quality revision of `edge_ai_compression_paper_2026-09-19.docx`
("Compression Robustness Is Architecture-Dependent"). Work was stopped deliberately
(user: "document everything u did until now and stop") — nothing here is a failure, it's
a checkpoint. The background job was killed cleanly after its 3rd of 9 models finished
and saved; no data was lost, and no stray processes were left running (verified via
`Get-CimInstance Win32_Process`, not `tasklist`, per this project's own established
lesson).

## 1. What was actually completed

### 1.1 The "90 variants" counting error — diagnosed, not yet fixed in the docx
Found the exact bug: `scripts/build_edge_ai_paper_2026-09-19.py` line 359-360 says "The
90 pruning/quantization variants evaluated in this paper (9 architectures x 10 pruning
levels, plus 9 x 3 quantization variants)" — 9x10=90 and 9x3=27 were correctly computed
separately, but the combined total was mislabeled "90" instead of 90+27=**117**. Fix
identified and ready to apply; not yet edited into the script/docx.

### 1.2 Literature review — 17 real, individually verified citations found (see Section 4
below for the full list). Every one of these was checked via live search before being
accepted — author names, venue, year, and (where available) DOI or arXiv ID confirmed,
not taken from memory or invented. Covers all 8 requested categories (dermatology CNNs,
lightweight architectures, pruning, PTQ, QAT, SE/swish quantization sensitivity, edge-AI
benchmarking, clinical AI deployment/dataset-shift). **Not yet inserted into the paper
document** — citations are verified and ready to write in, but the docx itself hasn't
been edited yet.

### 1.3 Extended statistical analysis — 3 of 9 models complete, 6 remaining
New script: `scripts/edge_ai_extended_analysis.py`. For each model, on the VALIDATION
set only (test set never opened, per your explicit instruction), this computes:
- Point metrics + 2000-resample bootstrap 95% CIs for accuracy, precision, **sensitivity
  (recall), specificity, F1, AUROC** — the exact stats you asked to add.
- Class distribution and majority-class chance baseline (val: 249 Eczema / 247 Other,
  **majority-class accuracy = 50.20%** — computed and confirmed, ready to cite).
- Quantized operator/dtype coverage (what fraction of Conv2d/Linear layers actually
  became INT8 vs. stayed FP32 fallback) — implemented and verified working (SqueezeNet1.1
  test case: 27/27 layers quantized, 100% coverage, 0 FP32 fallbacks).
- A **combined pruning (30%) + static INT8** evaluation per model.
- Peak-memory measurement (process RSS delta during inference) per model/config.

**Completed and saved** (checkpointed to
`papers/edge-ai-lightweight-deployment/edge_ai_extended_analysis_2026-09-19.json`):
ResNet18, EfficientNet-B0, EfficientNet-Lite0 — all 4 sub-analyses each (FP32, pruned-at-
knee, static INT8, pruned30+INT8), full CIs and memory numbers.

**Not yet run**: MobileNetV3-Small, MobileNetV2-1.0x, ShuffleNetV2-0.5x,
ShuffleNetV2-1.0x, SqueezeNet1.1, RepGhostNet-0.5x (6 models), plus the SE/swish ablation
section (see 1.4) for all 3 target models.

One real result already in from this new analysis: at its 50% pruning knee,
EfficientNet-B0's accuracy dropped from 80.65% to 63.91% (point estimate; CI not yet
inspected in depth) — consistent with, and now more rigorously measured than, the
original paper's headline finding.

### 1.4 SE/swish causal ablation — designed and unit-tested, not yet run at scale
This directly answers the paper-revision request to "replace overly strong causal claims
... unless experimentally demonstrated." Implemented in the same script
(`run_se_swish_ablation()`): for each of the 3 SE/swish models, builds two modified
copies of the ALREADY-TRAINED checkpoint (no retraining) —
1. SE modules replaced with `nn.Identity()` (bypasses the excitation gate).
2. Swish-family activations (SiLU/Hardswish) replaced with ReLU/ReLU6.

Both are then statically quantized exactly as the original was, and the FP32-to-INT8
accuracy *gap* is compared against the original model's gap — isolating whether removing
the component specifically shrinks the quantization collapse, not just checking whether
the component matters for the network. Verified working correctly, standalone:
- Module detection confirmed on MobileNetV3-Small: 9 SE modules, 18
  SiLU/Hardswish activations found.
- A forward pass through an SE-ablated MobileNetV3-Small was confirmed to run without
  error (output shape correct) before this was wired into the full pipeline.

**Not yet run for real** — this was scheduled to run automatically as the last step of
`edge_ai_extended_analysis.py`, but the job was stopped before reaching it.

### 1.5 QAT (quantization-aware training) — script written, never launched
New script: `scripts/edge_ai_qat.py`. For the 3 target models (EfficientNet-B0,
MobileNetV3-Small, RepGhostNet-0.5x): loads the existing FP32 checkpoint, inserts
fake-quantization ops (`prepare_qat_fx`), fine-tunes 3 epochs at lr=1e-5 on the TRAIN
split, converts to a real INT8 model, evaluates on VALIDATION (never test), with the
same bootstrap-CI treatment as everything else. Written and code-reviewed by hand, but
**never executed even once** — this is real gradient-based training (much more expensive
than the inference-only work above) and was deliberately queued to run *after* the
extended-analysis job finished, to avoid two heavy jobs competing for this 8GB machine's
memory at once. Zero epochs have actually run.

### 1.6 Structured/channel pruning — dependency installed, no experiment run
`torch-pruning` (not previously installed in this project) was installed successfully
(`pip install torch_pruning`, version 1.6.0 confirmed importable). No structured-pruning
experiment was written or run — this was queued behind QAT and pruning+fine-tuning in
priority order and never reached.

## 2. What was not started at all

- **Pruning + fine-tuning at selected sparsity levels** (task 2, bullet 1). No script
  written. Planned scope was 2-3 sparsity levels (around the 40-60% knee region) on 2-3
  representative models (ShuffleNetV2-1.0x, MobileNetV3-Small, ResNet18), fine-tuning
  with the pruning mask held fixed.
- **Static INT8 PTQ vs. INT8 QAT comparison** — blocked on 1.5 (QAT never ran), so no
  comparison exists yet, even though the PTQ side of it is already on record from the
  original paper.
- **Pareto analysis** (accuracy vs. size/latency/memory, multi-axis). Not started —
  straightforward once 1.3's memory numbers exist for all 9 models, but no plotting code
  written yet.
- **Real Raspberry Pi / Android hardware benchmarking.** Not attempted. This project has
  no such hardware — confirmed earlier in this session (only a Raspberry Pi *product
  brief PDF* exists in `reference_papers/`, not an actual device). This should be
  reported to the user as genuinely infeasible in this project's current state, not
  merely "not gotten to."
- **The actual paper rewrite itself** — no changes have been made to
  `edge_ai_compression_paper_2026-09-19.docx` or its build script
  (`scripts/build_edge_ai_paper_2026-09-19.py`) yet. Every fix and new result above is
  ready to be written in, but the writing has not started. This includes:
  - The reframing around "compression-aware architecture selection for resource-
    constrained clinical AI" (task 4) — not yet drafted.
  - The rewritten Abstract, Introduction, Contributions, Related Work, Discussion,
    Limitations, Conclusion — not yet drafted.
  - The patient-ID-unavailable limitation discussion (task 1) — not yet added (this
    project's curated image archive genuinely has no patient identifier, already
    documented elsewhere in this project, e.g.
    `papers/architecture-selection-report/honors-paper-report.docx` Section 3 — just
    needs restating in this paper's own Limitations section).
  - Softening the SE/swish causal language (task 1) — not yet edited; the ablation data
    that would justify *how much* to soften it (1.4) hasn't been generated yet either.

## 3. Honest accounting: completed vs. not, by the user's own task list

| # | Task | Status |
|---|---|---|
| 1a | Fix "90 variants" error | Diagnosed, fix not yet applied |
| 1b | Soften SE/swish causal claims | Not done — waiting on 1.4's ablation data |
| 1c | Add F1/sensitivity/specificity/AUROC/CIs | **Done for 3/9 models**, script works |
| 1d | Report class distribution/majority baseline | **Done** (val: 50.20% majority-class) |
| 1e | Discuss missing patient IDs | Not yet added to the docx |
| 1f | Report quantized op/dtype coverage | **Done for 3/9 models**, script works |
| 1g | Distinguish unstructured sparsity from real speedup | Already partly in the original paper; not re-verified/strengthened this pass |
| 2a | Pruning + fine-tuning | Not started |
| 2b | Static PTQ vs. QAT | Script written, **0 of 3 models run** |
| 2c | Combined pruning + INT8 | **Done for 3/9 models** |
| 2d | Structured/channel pruning | Dependency installed, no experiment run |
| 2e | Memory/RAM measurement | **Done for 3/9 models** |
| 2f | Real Pi/edge benchmarking | **Not feasible** — no hardware in this project |
| 2g | SE/swish ablation | Designed + unit-tested, **not run at scale** |
| 2h | Pareto analysis | Not started |
| 3 | Literature review, verified citations | **17 real citations found and verified**, not yet inserted into the docx |
| 4 | Reframe around compression-aware architecture selection | Not started |
| 5 | Concise IEEE writing, new tables/figures, label new vs. old | Not started |

## 4. Citations verified so far (ready to use, IEEE-numbered when inserted)

All checked via live search — author list, venue, year, and DOI/arXiv ID confirmed
before acceptance; nothing here was taken from memory or invented.

1. Esteva, A. et al. "Dermatologist-level classification of skin cancer with deep neural
   networks." *Nature* 542, 115–118 (2017). DOI: 10.1038/nature21056
2. Daneshjou, R., Vodrahalli, K., Novoa, R.A., et al. "Disparities in dermatology AI
   performance on a diverse, curated clinical image set." *Science Advances* 8(32),
   eabq6147 (2022). DOI: 10.1126/sciadv.abq6147
3. Tan, M., Le, Q. "EfficientNet: Rethinking Model Scaling for Convolutional Neural
   Networks." *ICML* 2019.
4. Iandola, F.N. et al. "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
   and <0.5MB model size." arXiv:1602.07360 (2016).
5. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.-C. "MobileNetV2: Inverted
   Residuals and Linear Bottlenecks." *CVPR* 2018.
6. Howard, A. et al. "Searching for MobileNetV3." *ICCV* 2019.
7. Ma, N., Zhang, X., Zheng, H.-T., Sun, J. "ShuffleNet V2: Practical Guidelines for
   Efficient CNN Architecture Design." *ECCV* 2018.
8. Han, K., Wang, Y., Tian, Q., Guo, J., Xu, C., Xu, C. "GhostNet: More Features from
   Cheap Operations." *CVPR* 2020.
9. Chen, C., Guo, Z., Zeng, H., Xiong, P., Dong, J. "RepGhost: A Hardware-Efficient Ghost
   Module via Re-parameterization." arXiv:2211.06088 (2022).
10. Han, S., Pool, J., Tran, J., Dally, W. "Learning both Weights and Connections for
    Efficient Neural Networks." *NeurIPS* 2015.
11. Li, H., Kadav, A., Durdanovic, I., Samet, H., Graf, H.P. "Pruning Filters for
    Efficient ConvNets." *ICLR* 2017.
12. Jacob, B. et al. "Quantization and Training of Neural Networks for Efficient
    Integer-Arithmetic-Only Inference." *CVPR* 2018. (arXiv:1712.05877 — covers both PTQ
    methodology and QAT, cite for both categories d and e.)
13. Nagel, M., Fournarakis, M., Amjad, R.A., Bondarenko, Y., van Baalen, M., Blankevoort,
    T. "A White Paper on Neural Network Quantization." arXiv:2106.08295 (2021).
14. Gholami, A., Kim, S., Dong, Z., Yao, Z., Mahoney, M.W., Keutzer, K. "A Survey of
    Quantization Methods for Efficient Neural Network Inference." arXiv:2103.13630
    (2021).
15. Kim, T., Yoo, Y., Yang, J. "FrostNet: Towards Quantization-Aware Network
    Architecture Search." arXiv:2006.09679 (2020). (Directly on hard-swish/SE
    quantization difficulty — the strongest single match for category f.)
16. Banbury, C., Reddi, V.J., Torelli, P., et al. "MLPerf Tiny Benchmark."
    arXiv:2106.07597 (2021).
17. Zech, J.R., Badgeley, M.A., Liu, M., Costa, A.B., Titano, J.J., Oermann, E.K.
    "Variable generalization performance of a deep learning model to detect pneumonia in
    chest radiographs: A cross-sectional study." *PLOS Medicine* 15(11), e1002683
    (2018).

Already-cited in the original paper draft, kept: TensorFlow Blog, "Higher accuracy on
vision models with EfficientNet-Lite" (2020) — a primary source (Google's own
EfficientNet-Lite design writeup), not peer-reviewed; flagged as such if kept.

## 5. Files produced this session (all real, on disk)

- `scripts/edge_ai_extended_analysis.py` — extended stats/coverage/memory/ablation
  script (working, 3/9 models run).
- `scripts/edge_ai_qat.py` — QAT script (written, never executed).
- `papers/edge-ai-lightweight-deployment/edge_ai_extended_analysis_2026-09-19.json` —
  partial results (3 of 9 models).
- This file.

## 6. To resume later

1. Re-run `python scripts/edge_ai_extended_analysis.py` — it resumes automatically from
   the 3 completed models (checks `OUT_JSON` and skips finished work), continuing with
   MobileNetV3-Small next, then runs the SE/swish ablation once all 9 models are done.
2. Once that finishes, run `python scripts/edge_ai_qat.py` (do not run at the same time
   as step 1 — both are CPU/memory-heavy on this 8GB machine).
3. Write the pruning+fine-tuning script and Pareto-plot script (neither exists yet).
4. Only then start editing `scripts/build_edge_ai_paper_2026-09-19.py` — fix the 90/117
   error, insert the 17 verified citations as IEEE-numbered references, rewrite the
   framing/Abstract/Intro/Discussion/Limitations/Conclusion, add the new results tables,
   and clearly label which results are new (2026-09-19 revision) vs. original.
5. Re-run QA (markitdown placeholder check + `validate.py`) on the rebuilt docx, same as
   every other document this session.
