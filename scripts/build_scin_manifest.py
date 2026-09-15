"""
External-validation manifest for the SCIN dataset (Ward et al., JAMA Network Open 2024;
Google/Stanford, crowd-sourced consumer dermatology photos). Used to test whether the
deployed Stage B model (models/curated_resnet18_balanced.pt) generalizes to a genuinely
independent photographic source -- SCIN images come from US Google Search users
photographing their own skin, nothing like the DermNet-style clinical-atlas archive the
curated dataset is built from.

scin_cases.csv / scin_labels.csv downloaded directly from the public GCS bucket
(no auth needed):
  https://storage.googleapis.com/dx-scin-public-data/dataset/scin_cases.csv
  https://storage.googleapis.com/dx-scin-public-data/dataset/scin_labels.csv

weighted_skin_condition_label is a per-case dict of {condition_name: weight}, already
mapping clinical synonyms (e.g. "Nummular eczema", "Eczematous dermatitis") onto a single
"Eczema" label (see dataset_schema.md from google-research-datasets/scin). This script
takes each case's highest-weighted condition as its single label -- label=1 if that's
"Eczema", label=0 for a random sample of other-condition cases of equal size, so the
external test set is roughly balanced like the internal one.

Only image_1_path is used per case (one image per case_id) to avoid the multi-image
same-case leakage risk the report already flags for the internal dataset (no reliable
patient/case grouping there at all -- SCIN's case_id lets this script actually avoid the
problem, so it does).

Writes:
  dataset/SCIN/images/<case_id>.png  (downloaded)
  dataset/SCIN/manifest_scin.csv     (path, disease_class, label)
"""
import ast
import csv
from concurrent.futures import ThreadPoolExecutor
import random

import urllib.request

import pandas as pd

from paths import SCIN_DIR

BUCKET = "https://storage.googleapis.com/dx-scin-public-data/"
N_PER_CLASS = 488  # = number of Eczema-top-condition cases available; matches negative pool 1:1
SEED = 42


def parse_weighted(s):
    if pd.isna(s):
        return {}
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}


def top_condition(d):
    if not d:
        return None
    return max(d.items(), key=lambda kv: kv[1])[0]


def download_one(args):
    case_id, image_path, label, out_dir = args
    out_path = out_dir / f"{case_id}.png"
    if out_path.exists():
        return case_id, str(out_path), label, True
    try:
        with urllib.request.urlopen(BUCKET + image_path, timeout=30) as resp:
            out_path.write_bytes(resp.read())
        return case_id, str(out_path), label, True
    except Exception as e:
        print(f"  FAILED {case_id}: {e}")
        return case_id, str(out_path), label, False


def main():
    cases = pd.read_csv(SCIN_DIR / "scin_cases.csv", dtype={"case_id": str})
    labels = pd.read_csv(SCIN_DIR / "scin_labels.csv", dtype={"case_id": str})
    labels["weighted_parsed"] = labels["weighted_skin_condition_label"].apply(parse_weighted)
    labels["top_condition"] = labels["weighted_parsed"].apply(top_condition)

    df = pd.merge(cases[["case_id", "image_1_path"]], labels[["case_id", "top_condition"]], on="case_id")
    df = df.dropna(subset=["image_1_path", "top_condition"])

    eczema = df[df["top_condition"] == "Eczema"]
    other = df[df["top_condition"] != "Eczema"]
    print(f"Eczema-top-condition cases: {len(eczema)}")
    print(f"Other-top-condition cases (pool): {len(other)}")

    rng = random.Random(SEED)
    n_pos = min(N_PER_CLASS, len(eczema))
    n_neg = min(N_PER_CLASS, len(other))
    eczema_sample = eczema.sample(n=n_pos, random_state=SEED)
    other_sample = other.sample(n=n_neg, random_state=SEED)

    out_dir = SCIN_DIR / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = (
        [(r.case_id, r.image_1_path, 1, out_dir) for r in eczema_sample.itertuples()]
        + [(r.case_id, r.image_1_path, 0, out_dir) for r in other_sample.itertuples()]
    )
    print(f"Downloading {len(jobs)} images...")
    rows = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, (case_id, path, label, ok) in enumerate(ex.map(download_one, jobs)):
            if ok:
                cls = "Eczema" if label == 1 else "Other"
                rows.append({"path": path, "disease_class": cls, "label": label, "case_id": case_id})
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)} done")

    manifest_path = SCIN_DIR / "manifest_scin.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "disease_class", "label", "case_id"])
        w.writeheader()
        w.writerows(rows)

    n1 = sum(1 for r in rows if r["label"] == 1)
    n0 = sum(1 for r in rows if r["label"] == 0)
    print(f"\nManifest written: {manifest_path}")
    print(f"Eczema={n1}, Other={n0}, total={len(rows)}")


if __name__ == "__main__":
    main()
