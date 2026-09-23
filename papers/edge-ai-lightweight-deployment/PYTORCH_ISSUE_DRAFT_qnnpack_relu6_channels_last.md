# Posted as https://github.com/pytorch/pytorch/issues/198373 (2026-09-23)

**Title:** [quantization][qnnpack] quantized relu6 / hardtanh / clamp return wrong values for channels_last inputs (silently breaks FX-quantized MobileNetV2 on ARM)

## 🐛 Describe the bug

With the `qnnpack` quantized engine, `F.relu6`, `F.hardtanh` and `torch.clamp` give wrong results when the
quantized input tensor is in `channels_last` memory format. They are exact for contiguous (NCHW) input.

This matters in practice because qnnpack's quantized `Conv2d` returns `channels_last` tensors, so any
ReLU6 applied after a quantized conv (not fused into it) hits this path. An FX-quantized
(`prepare_fx`/`convert_fx`, default qnnpack qconfig) torchvision `mobilenet_v2`-based classifier silently
predicts a single class for every input: no error or warning, and accuracy near 50% on a balanced binary task.
The same calibrated model converted with `convert_to_reference_fx` behaves normally.

Minimal reproduction:

```python
import torch
import torch.nn.functional as F

torch.backends.quantized.engine = "qnnpack"
torch.manual_seed(0)

x = torch.randn(2, 8, 5, 5) * 4
q = torch.quantize_per_tensor(x, scale=0.05, zero_point=128, dtype=torch.quint8)
expected = torch.clamp(q.dequantize(), 0, 6)

for name, t in [("contiguous", q), ("channels_last", q.contiguous(memory_format=torch.channels_last))]:
    for op_name, op in [("F.relu6", lambda t: F.relu6(t)),
                        ("F.hardtanh", lambda t: F.hardtanh(t, 0.0, 6.0)),
                        ("torch.clamp", lambda t: torch.clamp(t, 0, 6))]:
        err = (op(t).dequantize() - expected).abs().max().item()
        print(f"{name:14s} {op_name:12s} max abs error = {err:.3f}")

conv = torch.ao.nn.quantized.Conv2d(8, 8, 3, padding=1)
print("quantized conv output is channels_last:", conv(q).is_contiguous(memory_format=torch.channels_last))
```

Output:

```
contiguous     F.relu6      max abs error = 0.000
contiguous     F.hardtanh   max abs error = 0.000
contiguous     torch.clamp  max abs error = 0.000
channels_last  F.relu6      max abs error = 6.000
channels_last  F.hardtanh   max abs error = 6.000
channels_last  torch.clamp  max abs error = 6.000
quantized conv output is channels_last: True
```

Expected: identical results for both memory formats (as for `F.relu`, `F.hardsigmoid`, `F.hardswish`,
`torch.sigmoid`, `F.max_pool2d` and `F.adaptive_avg_pool2d`, which we checked and which match on
channels_last input). In our tests the output of the failing ops is also reported as NCHW-contiguous
even though the input was channels_last, which suggests the result is written with the wrong layout.

Model-level impact (binary image classifier, 496-image balanced validation set, default qnnpack qconfig,
256 calibration images):

| model | FP32 | reference-quantized (`convert_to_reference_fx`) | qnnpack (`convert_fx`) | qnnpack + `.contiguous()` before each relu6 |
|---|---|---|---|---|
| torchvision mobilenet_v2 backbone | 78.8% | 71.2% | 49.8% (constant output) | 73.0% |
| timm efficientnet_lite0 backbone | 76.4% | n/a | 49.8% (constant output) | 67.7% |

Root cause (as far as we can tell from the source): in `aten/src/ATen/native/quantized/cpu/qclamp.cpp`,
`qnnpack_clamp` keeps the input in its suggested memory format
(`input.contiguous(input.suggest_memory_format())`) and runs the QNNPACK clamp over the raw buffer, but
allocates the output with

```cpp
Tensor qy = at::_empty_affine_quantized(
    input_contig.sizes(),
    input_contig.options(),
    input_contig.q_scale(),
    input_contig.q_zero_point());
```

i.e. without a memory format, so it defaults to contiguous NCHW. The NHWC-ordered result is then
interpreted as NCHW. With a traceable input (values -5..6, shape 1x3x2x2) the output exactly equals
`clamp(raw NHWC buffer)` read back in NCHW order. The sibling kernels `qnnpack_sigmoid`
(`qsigmoid.cpp`) and `qnnpack_hardsigmoid` (`qhardsigmoid.cpp`) pass
`input_contig.suggest_memory_format()` to `_empty_affine_quantized`, and their channels_last results are
correct. A likely one-line fix is to pass `input_contig.suggest_memory_format()` here as well. The code
is unchanged on `main` as of 2026-09-23.

Workaround: insert `x.contiguous()` before every relu6/hardtanh/clamp node in the converted GraphModule.

## Versions

- Reproduced identically (channels_last max error 6.000, NCHW 0.000) on PyTorch 2.4.1, 2.6.0, 2.8.0 and
  2.14.0 (pip wheels, macOS arm64).
- PyTorch 2.8.0 (pip), `torch.backends.quantized.supported_engines == ['qnnpack', 'none']`
- torchvision 0.23.0, timm 1.0.29
- macOS 26.6 (build 25G72), Apple M5 (arm64)
- Python 3.9.6

(Run `python -m torch.utils.collect_env` and paste its output here before filing.)

## Notes before posting

- Search the tracker for "qnnpack hardtanh channels_last" / "quantized clamp channels_last" first. Searches on 2026-09-23 (relu6 / hardtanh / clamp / channels_last / qnnpack_clamp) found no
  existing report, and qclamp.cpp's commit history shows no fix.
- The FX quantization APIs are deprecated in favour of torchao's pt2e flow; maintainers may ask whether it
  reproduces there. The standalone-op reproduction above doesn't depend on FX.
