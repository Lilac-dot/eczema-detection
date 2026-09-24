"""
Fetches only the Fitzpatrick17k-C and DermaCon-IN images that map onto the Stage B
classes (scripts/skintone_labels.py), straight out of the remote zips via
fetch_zip_members.py -- the full archives are 1.4 GB + 3.6 GB, most of it diagnoses
this project never uses. Re-runnable: already-extracted files are skipped.

Writes images to dataset/Fitzpatrick17k/images/ and dataset/DermaConIN/images/, which
build_skintone_manifest.py then indexes.
"""
import sys

import pandas as pd

from paths import DATASET_DIR
from skintone_labels import F17K_MAP, DERMACON_MAP
from fetch_zip_members import fetch_members

F17K_DIR = DATASET_DIR / "Fitzpatrick17k"
DERMACON_DIR = DATASET_DIR / "DermaConIN"

F17K_ZIP = "https://zenodo.org/api/records/11101338/files/Fitzpatrick17k_CategorizedAbbrvs.zip/content"
DERMACON_ZIPS = {
    "DATASET_0.zip": "https://dataverse.harvard.edu/api/access/datafile/11377389",
    "DATASET_1.zip": "https://dataverse.harvard.edu/api/access/datafile/11377566",
}


def main():
    # Optional argument "f17k" or "dermacon" fetches one source only, so the two (different
    # servers, each throttled separately) can run in parallel processes.
    only = sys.argv[1] if len(sys.argv) > 1 else None
    if only != "dermacon":
        fetch_f17k()
    if only != "f17k":
        fetch_dermacon()


def fetch_f17k():
    f17k = pd.read_csv(F17K_DIR / "Fitzpatrick17k-C.csv")
    wanted = set(f17k.loc[f17k["label"].isin(F17K_MAP), "new_img_name"])
    print(f"Fitzpatrick17k-C: {len(wanted)} images wanted")
    fetch_members(F17K_ZIP, wanted, F17K_DIR / "images",
                  partial_local=F17K_DIR / "Fitzpatrick17k_CategorizedAbbrvs.zip")


def fetch_dermacon():
    meta = pd.read_csv(DERMACON_DIR / "Skin_Metadata.csv")
    keep = meta["Disease_label"].isin(DERMACON_MAP) & meta["Quality"].str.startswith("Yes", na=False)
    wanted = set(meta.loc[keep, "Image_name"])
    print(f"DermaCon-IN: {len(wanted)} images wanted")
    for name, url in DERMACON_ZIPS.items():
        fetch_members(url, wanted, DERMACON_DIR / "images", partial_local=DERMACON_DIR / name)


if __name__ == "__main__":
    main()
