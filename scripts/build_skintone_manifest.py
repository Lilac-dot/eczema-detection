"""
Builds the pooled manifest for the skin-tone compression audit from three sources:
  - SCIN          (dataset/skintone/scin_rows.csv, written by download_scin_skintone.py)
  - Fitzpatrick17k-C (Abhishek et al. 2025 cleaned release; Zenodo record 11101338,
                   images from Fitzpatrick17k_CategorizedAbbrvs.zip, extracted under
                   dataset/Fitzpatrick17k/images/)
  - DermaCon-IN   (Harvard Dataverse doi:10.7910/DVN/W7OUZM; DATASET_0/1.zip extracted
                   under dataset/DermaConIN/images/)

Labels are harmonized onto Eczema vs. the 7 Stage B look-alike classes by
scripts/skintone_labels.py. Output columns are the same for every source:
  path, source, group_id, orig_label, disease_class, label, fst, fst_source, fst_self,
  mst, fst_group, shot_type, label_weight

group_id is the unit that must never be split across train/val/test: SCIN case_id,
DermaCon-IN Subject_ID. Fitzpatrick17k has no patient identifier, so each image is its
own group there -- near-duplicate images of the same patient are instead caught by the
phash screen in clean_skintone_manifest.py.

Writes dataset/skintone/manifest_raw.csv (before cleaning/dedup).

Usage:
    python build_skintone_manifest.py                 # all three sources
    python build_skintone_manifest.py --sources SCIN  # SCIN only (first-pass study)
"""
import argparse

import pandas as pd

from paths import DATASET_DIR, ROOT
from skintone_labels import F17K_MAP, DERMACON_MAP, fst_group

OUT_DIR = DATASET_DIR / "skintone"
F17K_DIR = DATASET_DIR / "Fitzpatrick17k"
DERMACON_DIR = DATASET_DIR / "DermaConIN"


def index_images(root):
    """filename -> path for every image under root (archives nest folders differently)."""
    idx = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            idx[p.name] = p
    return idx


def f17k_rows():
    meta = pd.read_csv(F17K_DIR / "Fitzpatrick17k-C.csv")
    meta["disease_class"] = meta["label"].map(F17K_MAP)
    meta = meta.dropna(subset=["disease_class"])
    idx = index_images(F17K_DIR / "images")
    rows, missing = [], 0
    for r in meta.itertuples():
        p = idx.get(r.new_img_name) or idx.get(r.filepath.split("/")[-1])
        if p is None:
            missing += 1
            continue
        fst = int(r.fitzpatrick) if r.fitzpatrick in range(1, 7) else None
        rows.append(dict(
            path=str(p.relative_to(ROOT)), source="Fitzpatrick17k", group_id=f"F17K_{r.md5hash}",
            orig_label=r.label, disease_class=r.disease_class,
            label=int(r.disease_class == "Eczema"), fst=fst,
            fst_source="crowd_annotator" if fst else None, fst_self=None, mst=None,
            fst_group=fst_group(fst), shot_type=None, label_weight=None,
        ))
    print(f"Fitzpatrick17k-C: {len(rows)} rows, {missing} images not found on disk")
    return rows


def dermacon_rows():
    meta = pd.read_csv(DERMACON_DIR / "Skin_Metadata.csv")
    meta["disease_class"] = meta["Disease_label"].map(DERMACON_MAP)
    meta = meta.dropna(subset=["disease_class"])
    # Keep only images the annotating dermatologist marked sufficient for diagnosis.
    meta = meta[meta["Quality"].str.startswith("Yes", na=False)]
    idx = index_images(DERMACON_DIR / "images")
    rows, missing = [], 0
    for r in meta.itertuples():
        p = idx.get(r.Image_name)
        if p is None:
            missing += 1
            continue
        fst = int(r.Fitzpatrick.split()[-1]) if isinstance(r.Fitzpatrick, str) and \
            r.Fitzpatrick.startswith("FST") else None
        mst = int(r.Monk_skin_tone.split()[-1]) if isinstance(r.Monk_skin_tone, str) and \
            r.Monk_skin_tone.startswith("MST") else None
        rows.append(dict(
            path=str(p.relative_to(ROOT)), source="DermaCon-IN", group_id=f"DCIN_{r.Subject_ID}",
            orig_label=r.Disease_label, disease_class=r.disease_class,
            label=int(r.disease_class == "Eczema"), fst=fst,
            fst_source="dermatologist" if fst else None, fst_self=None, mst=mst,
            fst_group=fst_group(fst), shot_type=None, label_weight=r.Confidence,
            sex=r.Sex.upper() if r.Sex in ("Male", "Female") else None, age_group=r.Age,
        ))
    print(f"DermaCon-IN: {len(rows)} rows, {missing} images not found on disk")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["SCIN", "Fitzpatrick17k", "DermaCon-IN"],
                    choices=["SCIN", "Fitzpatrick17k", "DermaCon-IN"])
    args = ap.parse_args()
    parts = []
    if "SCIN" in args.sources:
        scin = pd.read_csv(OUT_DIR / "scin_rows.csv")
        print(f"SCIN: {len(scin)} rows")
        parts.append(scin)
    if "Fitzpatrick17k" in args.sources:
        parts.append(pd.DataFrame(f17k_rows()))
    if "DermaCon-IN" in args.sources:
        parts.append(pd.DataFrame(dermacon_rows()))
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(OUT_DIR / "manifest_raw.csv", index=False)
    print(f"\nWrote {len(df)} rows -> {OUT_DIR / 'manifest_raw.csv'}\n")
    print(pd.crosstab([df["source"], df["disease_class"]], df["fst_group"].fillna("unknown"),
                      margins=True).to_string())


if __name__ == "__main__":
    main()
