# -*- coding: utf-8 -*-
"""Declared protocol deviation (FROZEN_TEST_PROTOCOL_DEVIATION_2026-09-23.md): re-run the frozen
test-split qnnpack INT8 evaluation for all 9 architectures, as_is vs. with the relu6/hardtanh/clamp
channels_last fix. Same checkpoints, quantize_static, training-split calibration and seed as
eval_final_test_frozen.py."""
import json, sys, warnings
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT, SKINDISEASE_DIR  # noqa: E402
import edge_ai_extended_analysis as ext  # noqa: E402
from qnnpack_relu6_layout_fix_2026_09_23 import fix_layout  # noqa: E402

torch.backends.quantized.engine = "qnnpack"
TEST_MANIFEST = SKINDISEASE_DIR / "manifest_curated_v3_test.csv"
OUT = ROOT / "papers" / "edge-ai-lightweight-deployment" / "final_test_relu6_fix_2026-09-23.json"
ALL = ["resnet18", "efficientnet_b0", "efficientnet_lite0", "mobilenetv3_small", "mobilenetv2_100",
       "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "squeezenet1_1", "repghostnet_050"]


def cm(labels, probs):
    p = (np.array(probs) >= 0.5).astype(int); l = np.array(labels)
    return {"tp": int(((p == 1) & (l == 1)).sum()), "fp": int(((p == 1) & (l == 0)).sum()),
            "tn": int(((p == 0) & (l == 0)).sum()), "fn": int(((p == 0) & (l == 1)).sum())}


def main():
    import os
    os.chdir(ROOT)
    tf = ext.make_eval_transform()
    test_loader = DataLoader(ext.CuratedDataset(TEST_MANIFEST, tf), batch_size=32, shuffle=False)
    train_loader = DataLoader(ext.CuratedDataset(ext.TRAIN_MANIFEST, tf), batch_size=32, shuffle=True)
    out = {}
    for name in ALL:
        torch.manual_seed(42); np.random.seed(42)
        q = ext.quantize_static(ext.load_model(name), train_loader)
        qf, n = fix_layout(q)
        res = {"n_clamp_nodes_fixed": n}
        for tag, m in (("as_is", q), ("fixed", qf)):
            labels, probs = ext.get_predictions(m, test_loader, name == "resnet18")
            res[tag] = {"point": {k: float(v) for k, v in ext.point_metrics(labels, probs).items()},
                        "confusion_matrix": cm(labels, probs)}
        out[name] = res
        a, f = res["as_is"]["point"], res["fixed"]["point"]
        print(f"{name:20s} nodes={n:3d} as_is acc={a['accuracy']:.3f} | fixed acc={f['accuracy']:.3f} "
              f"se={f['sensitivity_recall']:.3f} sp={f['specificity']:.3f} f1={f['f1']:.3f}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
