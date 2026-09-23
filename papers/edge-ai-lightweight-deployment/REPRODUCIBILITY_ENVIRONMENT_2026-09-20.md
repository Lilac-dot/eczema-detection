# Reproducibility / Environment Document (2026-09-20)

Companion to `eczema_compression_deployment_paper_2026-09-20.docx`, Section 4.9-4.10.
Documents exactly what changed between this project's original development machine and
the machine used for this paper's new experiments, and which results were produced on
which machine.

## 1. Machines involved

| | Original machine | This paper's new-experiment machine |
|---|---|---|
| OS | Windows | macOS (Darwin), Apple Silicon |
| CPU | x86_64 | Apple M5 (ARM64), 10 cores |
| RAM | 8 GB | 16 GB |
| Quantized backend available | `x86` (fbgemm) | `qnnpack` only (`torch.backends.quantized.supported_engines == ['qnnpack', 'none']`) |
| Python env at session start | already set up | **not set up at all** -- no Xcode Command Line Tools, `/usr/bin/python3` failed on `print("hi")` |

## 2. Environment fixes required before any new experiment could run

All fixes were necessary just to reach a working state -- none were optional hardening.

1. **Xcode Command Line Tools** -- macOS's stub Python requires this to function at all.
   User-run (`xcode-select --install`, GUI installer), not automatable from a sandboxed
   shell.
2. **File ownership** -- the entire project directory was owned by `root:wheel`
   (mode 755/644), not the user account, blocking every write. Fixed with
   `sudo chown -R tishy:staff /Users/tishy/Documents/Honors` (user-run, requires a
   password prompt an automated shell cannot supply).
3. **Python dependencies** -- installed via `python3 -m pip install --user pandas numpy
   lightgbm scikit-learn torch torchvision matplotlib pillow python-docx imagehash scipy
   openpyxl timm psutil torch_pruning`. Versions actually installed: torch 2.8.0,
   torchvision 0.23.0, timm 1.0.29, torch_pruning 1.6.0, psutil 7.2.2, scikit-learn 1.6.1.
4. **`scripts/mem_guard.py` was Windows-only.** It called
   `ctypes.windll.kernel32.GlobalMemoryStatusEx`, which does not exist on macOS/Linux.
   Replaced with `psutil.virtual_memory().available` (psutil was already a project
   dependency elsewhere, so this removed a platform-specific dependency rather than
   adding one).
5. **Dataset manifest CSVs contained hardcoded Windows absolute paths**
   (`C:\Users\tishy\Documents\Honors\...`) in their `path` column. Affects
   `SkinDisease/manifest_curated_v3_train.csv`, `manifest_curated_v3_val.csv`, and (found
   later, for the SkinDisNet external-validation extension)
   `SkinDisNet/manifest_skindisnet_clean.csv`. Fixed the same way each time: stripping
   the Windows prefix and converting remaining backslashes to forward slashes (both
   machines happen to use the same username, so the relative structure lines up
   exactly), with a `.winpath.bak` backup kept for each file before rewriting (no git in
   this project, so this was the only safety net). `CuratedDataset` in
   `scripts/edge_ai_extended_analysis.py` and `ExternalDataset` in
   `scripts/eval_external_common.py` were both patched to resolve relative paths against
   `paths.ROOT` explicitly, rather than relying on the process's current working
   directory, since that is fragile across different launch contexts.
6. **Quantization backend was hardcoded to `"x86"`.** `edge_ai_extended_analysis.py` and
   `edge_ai_qat.py` called `get_default_qconfig_mapping("x86")` /
   `get_default_qat_qconfig_mapping("x86")` and relied on PyTorch's default quantized
   engine, which is `fbgemm`-backed and simply does not exist on this ARM machine
   (`RuntimeError: Didn't find engine for operation quantized::conv2d_prepack NoQEngine`).
   Fixed by adding `QUANT_BACKEND = "x86" if "x86" in torch.backends.quantized.
   supported_engines else "qnnpack"` and using that everywhere a backend string was
   previously hardcoded, so the same code runs correctly on either machine. **This fix
   is the direct origin of the paper's central finding** (Section 5.5): once the code
   could run on `qnnpack` at all, its quantization behavior turned out to differ sharply
   from `x86/fbgemm` for several architectures.

Every fix above is a source-code or environment change, not a change to any trained
model, dataset split, or evaluation procedure -- no result computed on the original
machine needed to be, or was, altered to make this work.

## 3. Verification performed before trusting any new experiment

Before queuing any multi-hour run, each new script was smoke-tested at small scale
(one model, one epoch/ratio, or a short manual REPL run reproducing its core operations)
and checked against an already-known number where one existed (e.g., EfficientNet-B0's
pruned-at-knee clean accuracy reproduced exactly as 63.91%, matching the value already on
record from the original machine, before the full extended sweep was trusted to run
unattended).

## 4. Results provenance: exact file-by-file accounting

**Computed on the original (x86/Windows) machine, before this paper's new work began --
reused, not recomputed:**
- All 9 architectures' base training and checkpoints (`experiments/*/checkpoints/`,
  `models/curated_resnet18_balanced.pt`)
- `edge_simulation_all_architectures_2026-09-19.json` -- original PTQ/pruning sweep
- `edge_ai_extended_analysis_2026-09-19.json` -- bootstrap CIs, sensitivity/specificity/
  AUROC, majority-class baseline, quantized-op coverage, memory, SE/swish ablation,
  pruned30+INT8 combined, for all 9 architectures
- `edge_ai_pareto_analysis_2026-09-19.json` and its two plots
- `results/robustness_corruption_results_*.json` (8 architectures) and
  `results/robustness_cross_architecture_summary.json`
- `results/robustness_under_compression_shufflenet_v2_x1_0_x86backend_original.json`
  (backed up under this name during this session, before being overwritten by a qnnpack
  re-run of the same script)

**Computed during this session, on the ARM/macOS machine, for this paper:**
- `edge_ai_qat_2026-09-19.json` -- QAT for 3 SE/swish models (`scripts/edge_ai_qat.py`)
- `edge_ai_structured_pruning_2026-09-19.json` -- structured pruning, 7 architectures
  (`scripts/edge_ai_structured_pruning.py`, new script)
- `edge_ai_pruning_finetune_2026-09-19.json` -- pruning + fine-tune, 3 architectures x 3
  sparsity levels (`scripts/edge_ai_pruning_finetune.py`, new script)
- `results/robustness_under_compression_efficientnet_b0.json`,
  `results/robustness_under_compression_mobilenetv2_100.json` -- new architectures for
  the compression x corruption study (`scripts/robustness_under_compression.py`,
  generalized this session from a single-model script via `--model`)
- `results/robustness_under_compression_shufflenet_v2_x1_0.json` -- re-run of the
  original ShuffleNetV2-1.0x compression x corruption study on the qnnpack backend, for
  a backend-consistent 3-way comparison against the two new architectures above
- `edge_ai_qnnpack_backend_check_2026-09-19.json` -- the full 9-architecture x86-vs-
  qnnpack backend comparison (`scripts/edge_ai_qnnpack_backend_check.py`, new script) --
  this paper's central finding
- `experiment_matrix_2026-09-19.{json,md}` -- consolidated cross-experiment table
  (`scripts/build_experiment_matrix.py`, new script)
- A targeted literature review checking whether the x86/fbgemm-vs-ARM/qnnpack backend
  finding was already documented -- it was, at the PyTorch engineering level
  (`pytorch/pytorch` issue #44939 and related forum reports), which downgraded this
  paper's novelty claim on that point from "previously unreported" to "a known
  framework-level inconsistency, shown here to cause complete classifier collapse in an
  applied clinical task" (paper Sections 2.6-2.7). Also surfaced directly relevant
  prior AD/eczema wearable-deployment work (a MobileNet+Raspberry Pi AD-vs-psoriasis
  classifier, an EfficientNet mobile eczema/acne classifier, a 2025 systematic review),
  none of which tests compression -- cited in the paper's Related Work and References.
- `skindisnet_backend_collapse_check_2026-09-20.json` -- original 2-architecture
  external-validation check (`scripts/skindisnet_backend_collapse_check.py`), superseded
  by the all-9 version below but kept on disk as the first version of this check.
- `qnnpack_perchannel_diagnostic_2026-09-20.json` -- methodology-audit diagnostic
  (`scripts/qnnpack_perchannel_diagnostic.py`, new script), requested during
  publication-readiness review: re-quantizes the 5 qnnpack-affected architectures on the
  qnnpack engine with weight quantization forced to per-channel (qnnpack's own default
  is per-tensor), to isolate whether quantization granularity, not just "the backend,"
  is the cause. Result: rules out granularity for the 3 fully-collapsed architectures
  (no change), confirms it as a major factor for EfficientNet-B0, minor for
  MobileNetV3-Small (paper Section 5.5.1).
- `skindisnet_all9_external_validation_2026-09-20.json` -- external-validation check
  extended to all 9 architectures (`scripts/skindisnet_all9_external_validation.py`, new
  script), also requested during publication-readiness review. Verifies and logs the
  SkinDisNet label mapping programmatically before evaluation (Eczema + Atopic
  Dermatitis -> positive, four other classes -> negative). x86/fbgemm INT8 not
  attempted here -- flagged as infeasible on this machine (qnnpack is the only
  executable quantized engine), not fabricated. Result: all 3 internally-collapsing
  architectures reach the exact same 0%/100%/F1=0.0000 signature on SkinDisNet; the
  other 6 cannot be validated as quantization-robust externally, because their FP32
  zero-shot performance on SkinDisNet is already severely degraded before any
  quantization (a separate, pre-existing generalization-gap limitation) (paper Section
  5.5.2).
- `FROZEN_TEST_PROTOCOL_2026-09-20.md` -- the exact test-set evaluation protocol,
  written and saved to disk BEFORE `SkinDisease/manifest_curated_v3_test.csv` (507
  images) was opened for the first time in this compression study, or before
  `scripts/eval_final_test_frozen.py` was written.
- `final_test_frozen_results_2026-09-20.json` -- the held-out test-set evaluation
  itself (`scripts/eval_final_test_frozen.py`, new script), implementing exactly the
  frozen protocol above: FP32 and qnnpack INT8, all 9 architectures, calibration on the
  training split only. x86/fbgemm INT8 on test not attempted -- same infeasibility as
  SkinDisNet. Also required applying the same Windows-path fix (Section 2, item 5) to
  `manifest_curated_v3_test.csv`, backed up first as
  `manifest_curated_v3_test.csv.winpath.bak`. Result: the exact same 3-collapsed /
  2-degraded / 4-robust split found on validation reproduces on test,
  architecture-for-architecture, with no exceptions (paper Section 9).
- This document and `eczema_compression_deployment_paper_2026-09-20.docx`
  (`scripts/build_eczema_compression_paper_2026-09-20.py`, updated for
  publication-readiness review: 3-way collapsed/severely-degraded/robust terminology
  replacing an earlier binary collapsed/stable label, a methodology audit distinguishing
  what was and was not held constant between backends, an updated title, the full-9
  SkinDisNet results, and the new Section 9 test evaluation)

## 5. Known data-quality caveat

A subset of `edge_ai_pruning_finetune_2026-09-19.json`'s recorded epoch times
(MobileNetV3-Small at 50% sparsity's 3rd epoch, and all 3 epochs at 70% sparsity) are
30-50x longer than every other epoch measured in this project (thousands of seconds vs.
~100-130s), almost certainly reflecting the host machine throttling this long-running
background job (e.g. sleep/App Nap), not a real compute-time difference. The accuracy
and loss values from those epochs are used normally; the wall-clock times are not, and
no latency or timing claim anywhere in the paper is based on them.
