"""
Mechanism checks for the skin-tone compression audit (audit_skintone_compression.py),
using only its saved out-of-fold predictions -- no retraining.

1. Representation (Hooker et al. 2019/2020: compression hurts under-represented
   examples first): each source's share of FST V-VI in the development set, next to that
   source's dark-minus-light delta-AUC. If the disparity tracks representation (SCIN
   8% V-VI vs DermaCon-IN 29%), representation -- not skin tone itself -- is the likelier
   driver.
2. Model uncertainty (Iofinova et al. CVPR 2023: samples the dense model is unsure about
   are the ones compression changes): the FP32 decision margin |p - 0.5| by Fitzpatrick
   group, and a logistic regression of "prediction flipped" on skin-tone group AND
   margin. If the group effect vanishes once margin is controlled for, darker skin is
   affected because the model is less certain there, not because of skin tone per se.

Writes experiments/skintone_audit/mechanisms.json.
"""
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from paths import ROOT
from train_skintone_cv import load_dev

AUDIT_DIR = ROOT / "experiments" / "skintone_audit"
KEY_VARIANTS = ["int8", "prune30", "prune30_int8", "prune50"]


def main():
    preds = pd.read_csv(AUDIT_DIR / "predictions.csv")
    summary = json.load(open(AUDIT_DIR / "audit_summary.json"))
    dev = load_dev()
    share = dev.groupby("source")["fst_group"].apply(lambda s: round(float((s == "V-VI").mean()), 3))
    out = {"v_vi_share_of_dev": share.to_dict(), "by_model": {}}

    for (model, source), d in preds.groupby(["model", "source"]):
        wide = d.pivot_table(index=["path", "split_group", "label", "fst_group"], columns="variant",
                             values="prob").reset_index()
        wide["margin"] = (wide["fp32"] - 0.5).abs()
        res = {
            "fp32_margin_median_by_fst": wide.groupby("fst_group")["margin"].median().round(4).to_dict(),
            "dark_minus_light_delta_auc": {
                v: summary[model][source]["variants"][v]["delta_auc_dark_minus_light"]
                for v in KEY_VARIANTS if v in summary[model][source]["variants"]},
            "flip_models": {},
        }
        for v in KEY_VARIANTS:
            if v not in wide:
                continue
            wide["flip"] = ((wide["fp32"] >= 0.5) != (wide[v] >= 0.5)).astype(int)
            if wide["flip"].sum() < 10:
                res["flip_models"][v] = "too few flips to model"
                continue
            ref = "III-IV" if source == "DermaCon-IN" else "I-II"
            fits = {}
            for name, formula in [("group_only", f"flip ~ C(fst_group, Treatment('{ref}'))"),
                                  ("group_plus_margin", f"flip ~ C(fst_group, Treatment('{ref}')) + margin")]:
                try:
                    m = smf.logit(formula, data=wide).fit(disp=0, cov_type="cluster",
                                                          cov_kwds={"groups": pd.factorize(wide["split_group"])[0]})
                    fits[name] = {k: dict(odds_ratio=round(float(np.exp(m.params[k])), 3),
                                          p=round(float(m.pvalues[k]), 4))
                                  for k in m.params.index if k != "Intercept"}
                except Exception as e:  # separation etc.
                    fits[name] = f"fit failed: {e}"
            res["flip_models"][v] = fits
        out["by_model"].setdefault(model, {})[source] = res

    with open(AUDIT_DIR / "mechanisms.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
