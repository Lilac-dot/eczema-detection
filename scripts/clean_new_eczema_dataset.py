"""
Clean the new "Eczema/" dataset (17 subtype subfolders) added by the user, same
methodology as clean_dataset.py / clean_skindisease.py: verify each file genuinely opens
as an image, flag corrupt/tiny files, and detect exact/near duplicates WITHIN this
dataset. Does not check against the other two datasets yet -- that's a separate script
(merge_all_eczema_sources.py) since it needs to compare against the already-merged pool.

Splits the 17 subfolders into two groups based on whether they're a recognized
eczema/atopic-dermatitis subtype, or a distinct (even if related-looking) diagnosis that
would mislabel the Eczema class if included -- see the reasoning given to the user before
running this:

  ECZEMA subtypes: Atopic dermatitis childhood phase, Atopic dermatitis feet,
    Eczema areola, Eczema asteatotic, Eczema chronic, Eczema fingertips, Eczema foot,
    Eczema hand, Dyshidrosis, Pompholyx, Stasis dermatitis

  EXCLUDED (distinct diagnoses): Ichthyosis, Keratolysis exfoliativa,
    Keratosis pilaris, Lichen simplex chronicus (conflicts with the existing "Lichen"
    other-disease class), Neurotic excoriations, Prurigo nodularis

Writes:
  Eczema/manifest_clean.csv (path, subtype, is_eczema, width, height) -- survivors only
  Eczema/clean_report.csv -- flagged/quarantined files + reason
  Quarantined files moved to Eczema/_quarantine/<subtype>/<reason>/
"""
import hashlib
import csv
import shutil
from pathlib import Path
from PIL import Image
import imagehash

ROOT = Path(r"C:\Users\tishy\Documents\Honors\Eczema")
QUARANTINE = ROOT / "_quarantine"
MIN_DIM = 64

ECZEMA_SUBTYPES = {
    "Atopic dermatitis childhood phase", "Atopic dermatitis feet", "Eczema areola",
    "Eczema asteatotic", "Eczema chronic", "Eczema fingertips", "Eczema foot",
    "Eczema hand", "Dyshidrosis", "Pompholyx", "Stasis dermatitis",
}
EXCLUDED_SUBTYPES = {
    "Ichthyosis", "Keratolysis exfoliativa", "Keratosis pilaris",
    "Lichen simplex chronicus", "Neurotic excoriations", "Prurigo nodularis",
}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def quarantine(path: Path, subtype: str, reason: str):
    dest_dir = QUARANTINE / subtype / reason
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    if dest.exists():
        dest = dest_dir / f"{path.stem}_{md5_of(path)[:8]}{path.suffix}"
    shutil.move(str(path), str(dest))


def main():
    subtypes = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and p.name != "_quarantine")
    unknown = set(subtypes) - ECZEMA_SUBTYPES - EXCLUDED_SUBTYPES
    if unknown:
        raise ValueError(f"Unclassified subfolder(s), update the script: {unknown}")

    report_rows = []
    manifest_rows = []
    seen_md5 = {}
    seen_phash = {}

    for subtype in subtypes:
        cls_dir = ROOT / subtype
        files = sorted(p for p in cls_dir.iterdir() if p.is_file())
        print(f"[{subtype}] scanning {len(files)} files...")

        for path in files:
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    ph = imagehash.phash(im)
            except Exception as e:
                report_rows.append([str(path), subtype, "corrupt_or_unreadable", str(e)])
                quarantine(path, subtype, "corrupt_or_unreadable")
                continue

            if w < MIN_DIM or h < MIN_DIM:
                report_rows.append([str(path), subtype, "too_small", f"{w}x{h}"])
                quarantine(path, subtype, "too_small")
                continue

            digest = md5_of(path)
            if digest in seen_md5:
                report_rows.append([str(path), subtype, "exact_duplicate", f"dup_of={seen_md5[digest]}"])
                quarantine(path, subtype, "exact_duplicate")
                continue
            seen_md5[digest] = str(path)

            is_near_dup = False
            for existing_hash, existing_path in seen_phash.items():
                if ph - existing_hash <= 4:
                    report_rows.append([str(path), subtype, "near_duplicate", f"similar_to={existing_path}"])
                    quarantine(path, subtype, "near_duplicate")
                    is_near_dup = True
                    break
            if is_near_dup:
                continue
            seen_phash[ph] = str(path)

            is_eczema = subtype in ECZEMA_SUBTYPES
            manifest_rows.append([str(path), subtype, int(is_eczema), w, h])

    with open(ROOT / "clean_report.csv", "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow(["path", "subtype", "reason", "detail"])
        w_.writerows(report_rows)

    with open(ROOT / "manifest_clean.csv", "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow(["path", "subtype", "is_eczema", "width", "height"])
        w_.writerows(manifest_rows)

    from collections import Counter
    reasons = Counter(r[2] for r in report_rows)
    eczema_kept = sum(1 for r in manifest_rows if r[2] == 1)
    excluded_kept = sum(1 for r in manifest_rows if r[2] == 0)
    print("\n=== Cleaning summary ===")
    for reason, n in reasons.items():
        print(f"  {reason}: {n}")
    print(f"Total quarantined: {len(report_rows)}")
    print(f"Kept as Eczema: {eczema_kept}")
    print(f"Kept as excluded (not eczema): {excluded_kept}")


if __name__ == "__main__":
    main()
