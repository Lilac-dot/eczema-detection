"""
Builds a deduplicated, skin-content-filtered SkinDisNet manifest for the full-set
zero-shot external-validation comparison (not the fine-tuning train/val/test splits,
which already have their own separate _clean variants -- this is the plain, whole-dataset
manifest used to zero-shot-test the deployed image model and any new candidate against it).

Two filters, both already computed and sitting in docs/ unused until now:
  1. Near-duplicate removal: docs/external_duplicate_screen_2026-09-16.csv found 37
     within-SkinDisNet near-duplicate pairs (perceptual hash, Hamming distance <=4). For
     each pair, keeps path_a and drops path_b -- consistent with this project's established
     "keep one, drop the other" convention (check_eczema_overlap.py, check_external_duplicates.py).
     One pair (EC (118) vs SC (287)/(290)) crosses disease labels -- the same photo appears
     under both Eczema and Scabies in the source dataset. Flagged in the printed output,
     not silently resolved by a label vote, since there's no ground truth to arbitrate it.
  2. Low skin-content removal: docs/external_skin_content_check_2026-09-16.csv computed a
     skin-pixel fraction per image (same check that found the SkinDisease dataset's "Normal"
     class was contaminated with non-skin photos, docs/skindisease_dataset notes). Drops
     rows with skin_fraction < 0.3, the same threshold already used as this check's own
     flag point.

Writes: SkinDisNet/manifest_skindisnet_clean.csv
"""
import csv

from paths import SKINDISNET_DIR, ROOT

DUP_SCREEN = ROOT / "docs" / "external_duplicate_screen_2026-09-16.csv"
SKIN_CHECK = ROOT / "docs" / "external_skin_content_check_2026-09-16.csv"
SKIN_FRACTION_THRESHOLD = 0.3


def main():
    manifest_path = SKINDISNET_DIR / "manifest_skindisnet.csv"
    rows = list(csv.DictReader(open(manifest_path, newline="", encoding="utf-8")))
    print(f"Starting manifest: {len(rows)} rows")

    # 1. drop the "b" side of each near-duplicate pair
    drop_paths = set()
    cross_label_pairs = []
    with open(DUP_SCREEN, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["check"] != "within_skindisnet_near":
                continue
            drop_paths.add(r["path_b"])
            a_cls = next((row["disease_class"] for row in rows if row["path"] == r["path_a"]), None)
            b_cls = next((row["disease_class"] for row in rows if row["path"] == r["path_b"]), None)
            if a_cls != b_cls:
                cross_label_pairs.append((r["path_a"], a_cls, r["path_b"], b_cls))

    if cross_label_pairs:
        print(f"\nWARNING: {len(cross_label_pairs)} near-duplicate pair(s) cross disease labels "
              f"(same/near-identical photo, different class in the source dataset) -- kept the "
              f"path_a side's label as-is, not arbitrated:")
        for a, ac, b, bc in cross_label_pairs:
            print(f"  KEPT {ac}: {a}\n  DROPPED {bc}: {b}")

    # 2. drop low-skin-content rows
    low_skin_paths = set()
    with open(SKIN_CHECK, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] != "SkinDisNet":
                continue
            if r["skin_fraction"] and float(r["skin_fraction"]) < SKIN_FRACTION_THRESHOLD:
                low_skin_paths.add(r["path"])

    kept = [r for r in rows if r["path"] not in drop_paths and r["path"] not in low_skin_paths]

    n_dup_dropped = len(rows) - len([r for r in rows if r["path"] not in drop_paths])
    n_skin_dropped = len([r for r in rows if r["path"] not in drop_paths]) - len(kept)
    print(f"\nDropped {n_dup_dropped} near-duplicate rows, {n_skin_dropped} low-skin-content rows "
          f"(skin_fraction < {SKIN_FRACTION_THRESHOLD})")
    print(f"Remaining: {len(kept)} rows")

    n1 = sum(1 for r in kept if r["label"] == "1")
    n0 = sum(1 for r in kept if r["label"] == "0")
    print(f"Eczema(+AD)={n1}, Other={n0}, total={len(kept)}")

    out_path = SKINDISNET_DIR / "manifest_skindisnet_clean.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "disease_class", "label"])
        w.writeheader()
        w.writerows(kept)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
