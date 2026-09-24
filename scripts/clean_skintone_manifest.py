"""
Cleans dataset/skintone/manifest_raw.csv (build_skintone_manifest.py) before any
training, following the same md5 + phash pattern as check_external_duplicates.py:

  1. Unreadable / truncated images                          -> dropped
  2. Images under MIN_SIDE px on their short side           -> dropped
  3. Exact duplicates (md5)                                 -> one copy kept
  4. Near-duplicates (phash Hamming <= HAMMING_THRESHOLD):
       within the pooled set -> NOT dropped; their group_ids are merged (union-find) so
                                every near-copy lands on the same side of the split.
                                This matters most for Fitzpatrick17k, which has no
                                patient id of its own.
       against the curated Stage B set (SkinDisease/manifest_curated_v3.csv, all splits)
                             -> dropped: the fine-tuning starting checkpoint was trained
                                on those images, so keeping them would leak into the
                                new evaluation.
  5. Rows with no skin-tone label are KEPT (usable for training) but flagged; the
     subgroup audit only ever evaluates rows with fst_group set.

Also records simple per-image statistics (resolution, mean RGB, brightness) for
check_skintone_merge.py.

Writes dataset/skintone/manifest_clean.csv and dataset/skintone/clean_report.json.
"""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from paths import ROOT, DATASET_DIR

OUT_DIR = DATASET_DIR / "skintone"
CURATED_MANIFEST = ROOT / "SkinDisease" / "manifest_curated_v3.csv"
HAMMING_THRESHOLD = 5
MIN_SIDE = 128
WIN_PREFIX = "C:\\Users\\tishy\\Documents\\Honors\\"


def local_path(p):
    """Curated manifests were written on the old Windows machine with absolute paths."""
    if p.startswith(WIN_PREFIX):
        p = p[len(WIN_PREFIX):].replace("\\", "/")
    return ROOT / p


def inspect(path):
    try:
        with open(path, "rb") as fh:
            md5 = hashlib.md5(fh.read()).hexdigest()
        with Image.open(path) as im:
            im.load()
            rgb = im.convert("RGB")
            w, h = rgb.size
            ph = int(str(imagehash.phash(rgb)), 16)
            small = np.asarray(rgb.resize((64, 64)), dtype=np.float32)
        mean = small.reshape(-1, 3).mean(0)
        return dict(ok=True, md5=md5, phash=ph, width=w, height=h,
                    mean_r=mean[0], mean_g=mean[1], mean_b=mean[2],
                    brightness=float(small.mean()))
    except Exception as e:
        return dict(ok=False, error=str(e))


def hamming_pairs(a, b, same, threshold):
    """All (i, j) with popcount(a[i] ^ b[j]) <= threshold, vectorized in row chunks."""
    a = np.array(a, dtype=np.uint64)
    b = np.array(b, dtype=np.uint64)
    popcount8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    pairs = []
    for start in range(0, len(a), 256):
        x = a[start:start + 256, None] ^ b[None, :]
        bits = popcount8[x.view(np.uint8).reshape(x.shape + (8,))].sum(-1)
        ii, jj = np.nonzero(bits <= threshold)
        ii = ii + start
        keep = ii < jj if same else np.ones_like(ii, dtype=bool)
        pairs.extend(zip(ii[keep].tolist(), jj[keep].tolist()))
    return pairs


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def main():
    df = pd.read_csv(OUT_DIR / "manifest_raw.csv")
    report = {"n_raw": len(df), "by_source_raw": df["source"].value_counts().to_dict()}

    with ProcessPoolExecutor() as ex:
        info = list(ex.map(inspect, [str(ROOT / p) for p in df["path"]], chunksize=64))
    info = pd.DataFrame(info)
    df = pd.concat([df.reset_index(drop=True), info], axis=1)

    bad = ~df["ok"]
    report["dropped_unreadable"] = int(bad.sum())
    df = df[~bad]
    small = df[["width", "height"]].min(axis=1) < MIN_SIDE
    report["dropped_too_small"] = int(small.sum())
    df = df[~small]

    dup = df.duplicated("md5", keep="first")
    report["dropped_exact_duplicates"] = int(dup.sum())
    report["exact_duplicates_cross_source"] = int(
        df[df["md5"].isin(df.loc[dup, "md5"])].groupby("md5")["source"].nunique().gt(1).sum())
    df = df[~dup].reset_index(drop=True)

    # Overlap with the curated Stage B set the starting checkpoint was trained on.
    cur = pd.read_csv(CURATED_MANIFEST)
    with ProcessPoolExecutor() as ex:
        cur_info = list(ex.map(inspect, [str(local_path(p)) for p in cur["path"]], chunksize=64))
    cur_hashes = [c["phash"] for c in cur_info if c["ok"]]
    report["curated_hashed"] = len(cur_hashes)
    leak = {i for i, _ in hamming_pairs(df["phash"].tolist(), cur_hashes, False, HAMMING_THRESHOLD)}
    report["dropped_overlap_with_curated"] = {
        s: int(n) for s, n in df.loc[sorted(leak), "source"].value_counts().items()}
    df = df.drop(index=sorted(leak)).reset_index(drop=True)

    # Near-duplicates inside the pooled set: merge groups instead of dropping.
    pairs = hamming_pairs(df["phash"].tolist(), df["phash"].tolist(), True, HAMMING_THRESHOLD)
    uf = UnionFind(df["group_id"].unique())
    cross_source = conflicting = 0
    for i, j in pairs:
        uf.union(df.at[i, "group_id"], df.at[j, "group_id"])
        cross_source += df.at[i, "source"] != df.at[j, "source"]
        conflicting += df.at[i, "disease_class"] != df.at[j, "disease_class"]
    df["split_group"] = df["group_id"].map(uf.find)
    report["near_duplicate_pairs"] = len(pairs)
    report["near_duplicate_pairs_cross_source"] = int(cross_source)
    report["near_duplicate_pairs_label_conflict"] = int(conflicting)

    # Near-duplicate images that disagree on diagnosis can't both be right -> drop the merged
    # group. Only groups CREATED by a near-duplicate merge are checked: one DermaCon-IN
    # subject photographed for two different conditions is legitimate and is kept (its
    # images just stay on the same side of the split).
    merged = df.groupby("split_group")["group_id"].transform("nunique") > 1
    n_cls = df.groupby("split_group")["disease_class"].transform("nunique")
    conflict = merged & (n_cls > 1)
    report["dropped_label_conflict_images"] = int(conflict.sum())
    report["groups_with_multiple_diagnoses_kept"] = int(
        df.loc[~merged & (n_cls > 1), "split_group"].nunique())
    df = df[~conflict]

    df = df.drop(columns=["ok", "error"], errors="ignore")
    df.to_csv(OUT_DIR / "manifest_clean.csv", index=False)
    report["n_clean"] = len(df)
    report["by_source_clean"] = df["source"].value_counts().to_dict()
    report["no_skin_tone_label"] = df["fst_group"].isna().groupby(df["source"]).sum().astype(int).to_dict()
    with open(OUT_DIR / "clean_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
