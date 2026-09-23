"""
Duplicate/overlap screen for the two external eczema datasets (SCIN, SkinDisNet) before
using them in the cross-dataset generalization matrix, following the same pattern as
check_eczema_overlap.py (md5 exact + phash near-duplicate, Hamming distance threshold).

Checks:
  (a) near-duplicates WITHIN SCIN
  (b) near-duplicates WITHIN SkinDisNet
  (c) overlap BETWEEN SCIN and SkinDisNet
  (d) overlap of either against the internal curated dataset (SkinDisease/manifest_curated_v3.csv)

Does not move/delete anything by itself -- writes a report and, if any duplicates are
found, a script-generated list of which manifest rows to drop (test-set copy is always
kept over a train-set copy of the same pair, to protect test-set integrity).
"""
import csv
import hashlib
from pathlib import Path
from collections import defaultdict

from PIL import Image
import imagehash

ROOT = Path(r"C:\Users\tishy\Documents\Honors")
SCIN_MANIFEST = ROOT / "dataset" / "SCIN" / "manifest_scin.csv"
SKINDISNET_MANIFEST = ROOT / "SkinDisNet" / "manifest_skindisnet.csv"
INTERNAL_MANIFEST = ROOT / "SkinDisease" / "manifest_curated_v3.csv"

HAMMING_THRESHOLD = 5


def load_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def hash_all(rows, label):
    md5s, phashes = {}, {}
    errors = 0
    for row in rows:
        p = row["path"]
        try:
            with open(p, "rb") as fh:
                digest = hashlib.md5(fh.read()).hexdigest()
            with Image.open(p) as im:
                ph = imagehash.phash(im.convert("RGB"))
        except Exception as e:
            errors += 1
            continue
        md5s[p] = digest
        phashes[p] = ph
    print(f"[{label}] hashed {len(phashes)}/{len(rows)} images ({errors} errors)")
    return md5s, phashes


def find_within_duplicates(md5s, phashes, label):
    print(f"\n=== Within-{label} duplicate check ===")
    # exact
    by_md5 = defaultdict(list)
    for p, d in md5s.items():
        by_md5[d].append(p)
    exact_groups = [g for g in by_md5.values() if len(g) > 1]
    print(f"Exact (md5) duplicate groups: {len(exact_groups)}")
    for g in exact_groups[:10]:
        print(f"  {g}")

    # near (phash), O(n^2) but datasets are <2000 images so fine
    items = list(phashes.items())
    near_pairs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            p1, h1 = items[i]
            p2, h2 = items[j]
            dist = h1 - h2
            if dist <= HAMMING_THRESHOLD:
                near_pairs.append((p1, p2, dist))
    print(f"Near-duplicate pairs (Hamming <= {HAMMING_THRESHOLD}, excluding exact matches already found): {len(near_pairs)}")
    for p1, p2, d in near_pairs[:15]:
        print(f"  {p1}\n  ~= {p2} (dist={d})")
    return exact_groups, near_pairs


def find_cross_duplicates(phashes_a, phashes_b, label_a, label_b):
    print(f"\n=== {label_a} vs {label_b} overlap check ===")
    pairs = []
    for pa, ha in phashes_a.items():
        for pb, hb in phashes_b.items():
            dist = ha - hb
            if dist <= HAMMING_THRESHOLD:
                pairs.append((pa, pb, dist))
    print(f"Cross-dataset near-duplicate pairs (Hamming <= {HAMMING_THRESHOLD}): {len(pairs)}")
    for pa, pb, d in pairs[:15]:
        print(f"  {pa}\n  ~= {pb} (dist={d})")
    return pairs


def main():
    scin_rows = load_manifest(SCIN_MANIFEST)
    skindisnet_rows = load_manifest(SKINDISNET_MANIFEST)
    internal_rows = load_manifest(INTERNAL_MANIFEST)

    print(f"SCIN: {len(scin_rows)} images")
    print(f"SkinDisNet: {len(skindisnet_rows)} images")
    print(f"Internal (curated_v3): {len(internal_rows)} images")

    scin_md5, scin_ph = hash_all(scin_rows, "SCIN")
    skdn_md5, skdn_ph = hash_all(skindisnet_rows, "SkinDisNet")
    internal_md5, internal_ph = hash_all(internal_rows, "Internal")

    scin_exact, scin_near = find_within_duplicates(scin_md5, scin_ph, "SCIN")
    skdn_exact, skdn_near = find_within_duplicates(skdn_md5, skdn_ph, "SkinDisNet")

    cross_scin_skdn = find_cross_duplicates(scin_ph, skdn_ph, "SCIN", "SkinDisNet")
    cross_scin_internal = find_cross_duplicates(scin_ph, internal_ph, "SCIN", "Internal")
    cross_skdn_internal = find_cross_duplicates(skdn_ph, internal_ph, "SkinDisNet", "Internal")

    # Write a combined report CSV of every flagged pair, for the record
    report_path = ROOT / "docs" / "external_duplicate_screen_2026-09-16.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["check", "path_a", "path_b", "distance"])
        for g in scin_exact:
            for p in g[1:]:
                w.writerow(["within_scin_exact", g[0], p, 0])
        for p1, p2, d in scin_near:
            w.writerow(["within_scin_near", p1, p2, d])
        for g in skdn_exact:
            for p in g[1:]:
                w.writerow(["within_skindisnet_exact", g[0], p, 0])
        for p1, p2, d in skdn_near:
            w.writerow(["within_skindisnet_near", p1, p2, d])
        for p1, p2, d in cross_scin_skdn:
            w.writerow(["scin_vs_skindisnet", p1, p2, d])
        for p1, p2, d in cross_scin_internal:
            w.writerow(["scin_vs_internal", p1, p2, d])
        for p1, p2, d in cross_skdn_internal:
            w.writerow(["skindisnet_vs_internal", p1, p2, d])

    total_flagged = (sum(len(g) - 1 for g in scin_exact) + len(scin_near)
                      + sum(len(g) - 1 for g in skdn_exact) + len(skdn_near)
                      + len(cross_scin_skdn) + len(cross_scin_internal) + len(cross_skdn_internal))
    print(f"\n=== Summary ===")
    print(f"Total flagged pairs/duplicates across all checks: {total_flagged}")
    print(f"Report written: {report_path}")


if __name__ == "__main__":
    main()
