"""
Train/val/test split for the pooled skin-tone dataset (dataset/skintone/manifest_clean.csv).

  - Group-aware: every image sharing a split_group (same SCIN case, same DermaCon-IN
    subject, or near-duplicates merged by clean_skintone_manifest.py) lands in exactly
    one split.
  - Stratified on source x label x Fitzpatrick group, so each split keeps every
    source's eczema/other and skin-tone mix -- the subgroup audit needs V-VI examples
    in test from every source that has them. (Sex was dropped as an audit axis: 46% of
    SCIN cases don't record it.)
  - 70 / 15 / 15 by group. The test split is written once and is meant to stay frozen
    (same practice as papers/edge-ai-lightweight-deployment/FROZEN_TEST_PROTOCOL).

Writes dataset/skintone/manifest_{train,val,test}.csv and split_report.json.
"""
import json

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from paths import DATASET_DIR

OUT_DIR = DATASET_DIR / "skintone"
SEED = 42


def take_fold(df, n_splits, seed):
    """One StratifiedGroupKFold fold (~1/n_splits of groups) and the remainder."""
    strata = df["source"] + "|" + df["label"].astype(str) + "|" + df["fst_group"].fillna("unk")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, fold_idx = next(sgkf.split(df, strata, groups=df["split_group"]))
    return df.iloc[rest_idx], df.iloc[fold_idx]


def main():
    df = pd.read_csv(OUT_DIR / "manifest_clean.csv")
    rest, test = take_fold(df, 7, SEED)            # ~15% test
    train, val = take_fold(rest, 6, SEED)          # ~15% of total as val
    assert not set(train["split_group"]) & set(test["split_group"])
    assert not set(train["split_group"]) & set(val["split_group"])
    assert not set(val["split_group"]) & set(test["split_group"])

    report = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(OUT_DIR / f"manifest_{name}.csv", index=False)
        report[name] = {
            "n": len(part), "groups": int(part["split_group"].nunique()),
            "by_source_label_fst": pd.crosstab(
                [part["source"], part["label"]], part["fst_group"].fillna("unknown")
            ).reset_index().to_dict("records"),
        }
        print(f"{name}: {len(part)} images, {part['split_group'].nunique()} groups")
        print(pd.crosstab([part["source"], part["label"]], part["fst_group"].fillna("unknown")).to_string())
    with open(OUT_DIR / "split_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    main()
