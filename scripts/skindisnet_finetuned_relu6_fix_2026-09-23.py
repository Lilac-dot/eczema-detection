# -*- coding: utf-8 -*-
"""Re-run the SkinDisNet fine-tuned qnnpack INT8 check (skindisnet_finetuned_backend_check.py,
2026-09-21) as_is vs. with the relu6/hardtanh/clamp channels_last fix. Same fine-tuned checkpoints,
same deterministic calibration (first 8 batches of the fine-tune training split, unshuffled), same
298-image fine-tune validation split."""
import json, sys, warnings
from pathlib import Path
import torch
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT, SKINDISNET_DIR, MODELS_DIR  # noqa: E402
from edge_ai_extended_analysis import CuratedDataset, make_eval_transform, get_predictions, point_metrics, quantize_static  # noqa: E402
from edge_simulation_all_architectures import build_candidate, build_resnet18_baseline  # noqa: E402
from qnnpack_relu6_layout_fix_2026_09_23 import fix_layout  # noqa: E402

torch.backends.quantized.engine = "qnnpack"
OUT = ROOT / "papers" / "edge-ai-lightweight-deployment" / "skindisnet_finetuned_relu6_fix_2026-09-23.json"
ORIG = json.load(open(ROOT / "papers" / "edge-ai-lightweight-deployment" / "skindisnet_finetuned_backend_check_2026-09-21.json"))["per_model"]
ALL = ["resnet18", "efficientnet_b0", "efficientnet_lite0", "mobilenetv3_small", "mobilenetv2_100",
       "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "squeezenet1_1", "repghostnet_050"]


def main():
    import os
    os.chdir(ROOT)
    tf = make_eval_transform()
    calib = DataLoader(CuratedDataset(SKINDISNET_DIR / "manifest_skindisnet_finetune_train.csv", tf), batch_size=32, shuffle=False)
    val = DataLoader(CuratedDataset(SKINDISNET_DIR / "manifest_skindisnet_finetune_val.csv", tf), batch_size=32, shuffle=False)
    out = {}
    for name in ALL:
        m = build_resnet18_baseline() if name == "resnet18" else build_candidate(name)
        m.load_state_dict(torch.load(MODELS_DIR / f"{name}_skindisnet_finetuned.pt", map_location="cpu"))
        m.eval()
        q = quantize_static(m, calib, n_calib_batches=8)
        qf, n = fix_layout(q)
        res = {"n_clamp_nodes_fixed": n}
        for tag, mm in (("as_is", q), ("fixed", qf)):
            labels, probs = get_predictions(mm, val, name == "resnet18")
            res[tag] = {k: float(v) for k, v in point_metrics(labels, probs).items()}
        o = ORIG[name]["finetuned_qnnpack_int8"]["point"]
        out[name] = res
        print(f"{name:20s} nodes={n:3d} orig acc={o['accuracy']:.3f} f1={o['f1']:.3f} | as_is acc={res['as_is']['accuracy']:.3f} "
              f"f1={res['as_is']['f1']:.3f} | fixed acc={res['fixed']['accuracy']:.3f} f1={res['fixed']['f1']:.3f} "
              f"se={res['fixed']['sensitivity_recall']:.3f} sp={res['fixed']['specificity']:.3f}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
