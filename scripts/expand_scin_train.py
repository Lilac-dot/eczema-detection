"""
Experiment 2c (fine-tune-more, continuation of Experiment 2): the 488 Eczema-top-condition
SCIN cases used throughout Experiment 2 were already the entire available top-condition
pool (build_scin_manifest.py's own N_PER_CLASS=488 comment says so) -- there is no way to
grow SCIN-train's positive class further without either touching the frozen val/test split
or loosening the label criterion. This does the latter, deliberately conservatively: SCIN's
weighted_skin_condition_label field gives 568 additional cases where Eczema is present but
NOT the single top-weighted condition; of those, 56 have Eczema weight >= 0.40 (close to or
tied with the top condition -- a real, if secondary, dermatologist-assigned diagnosis
confidence, not a token mention). This script adds those 56 plus 56 freshly-sampled
non-Eczema cases (to keep the added slice balanced) into SCIN-train ONLY.

manifest_scin_val.csv and manifest_scin_test.csv (from split_external_manifest.py) are
untouched -- read only to build the exclusion set, never written. This keeps the "before vs
after" comparison on those files valid.

Writes:
  dataset/SCIN/manifest_scin_train_expanded.csv (original 488 train rows + 112 new rows = 600)
"""
import ast
import csv
from concurrent.futures import ThreadPoolExecutor
import random

import urllib.request
import pandas as pd

from paths import SCIN_DIR

BUCKET = "https://storage.googleapis.com/dx-scin-public-data/"
ECZEMA_WEIGHT_THRESHOLD = 0.40
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


def load_used_case_ids():
    used = set()
    for name in ["manifest_scin_train.csv", "manifest_scin_val.csv", "manifest_scin_test.csv"]:
        path = SCIN_DIR / name
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                used.add(row["case_id"])
    return used


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
    labels["eczema_weight"] = labels["weighted_parsed"].apply(lambda d: d.get("Eczema", 0.0))

    df = pd.merge(cases[["case_id", "image_1_path"]], labels[["case_id", "top_condition", "eczema_weight"]], on="case_id")
    df = df.dropna(subset=["image_1_path", "top_condition"])

    used = load_used_case_ids()
    print(f"Already-used case_ids (train+val+test, excluded from new sampling): {len(used)}")

    # New positives: Eczema present with weight >= threshold but not the single top condition
    # (those ARE already used -- they're the original 488), and not already used.
    secondary_eczema = df[
        (df["eczema_weight"] >= ECZEMA_WEIGHT_THRESHOLD)
        & (df["top_condition"] != "Eczema")
        & (~df["case_id"].isin(used))
    ]
    print(f"New secondary-Eczema candidates (weight >= {ECZEMA_WEIGHT_THRESHOLD}, not top): {len(secondary_eczema)}")

    # New negatives: any non-Eczema-top case not already used, matched 1:1 to the new positives
    other_pool = df[(df["top_condition"] != "Eczema") & (~df["case_id"].isin(used))
                     & (~df["case_id"].isin(secondary_eczema["case_id"]))]
    n_new = len(secondary_eczema)
    other_sample = other_pool.sample(n=min(n_new, len(other_pool)), random_state=SEED)
    print(f"New negative sample (matched 1:1): {len(other_sample)}")

    out_dir = SCIN_DIR / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = (
        [(r.case_id, r.image_1_path, 1, out_dir) for r in secondary_eczema.itertuples()]
        + [(r.case_id, r.image_1_path, 0, out_dir) for r in other_sample.itertuples()]
    )
    print(f"Downloading {len(jobs)} new images...")
    new_rows = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, (case_id, path, label, ok) in enumerate(ex.map(download_one, jobs)):
            if ok:
                cls = "Eczema" if label == 1 else "Other"
                new_rows.append({"path": path, "disease_class": cls, "label": label, "case_id": case_id})
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} done")

    # Append to the existing train split (not val/test) -> manifest_scin_train_expanded.csv
    with open(SCIN_DIR / "manifest_scin_train.csv", newline="", encoding="utf-8") as f:
        existing_train = list(csv.DictReader(f))

    combined = existing_train + new_rows
    out_path = SCIN_DIR / "manifest_scin_train_expanded.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "disease_class", "label", "case_id"])
        w.writeheader()
        w.writerows(combined)

    n1 = sum(1 for r in combined if int(r["label"]) == 1)
    n0 = sum(1 for r in combined if int(r["label"]) == 0)
    print(f"\n{out_path}: {len(combined)} total (Eczema={n1}, Other={n0}) "
          f"-- was {len(existing_train)} ({sum(1 for r in existing_train if int(r['label'])==1)}/"
          f"{sum(1 for r in existing_train if int(r['label'])==0)})")
    print("manifest_scin_val.csv and manifest_scin_test.csv untouched.")


if __name__ == "__main__":
    main()
