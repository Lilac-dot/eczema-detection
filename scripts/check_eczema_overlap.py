"""
Check for exact/near-duplicate overlap between the two Eczema image pools before
merging them:
  - dataset/manifest_clean.csv (original Eczema/Normal dataset), class == 'Eczema'
  - SkinDisease/manifest_clean.csv (new disease dataset), class == 'Eczema'

Both datasets' Eczema images look like they come from the same underlying DermNet-style
source (matching filename conventions), so this checks whether merging them would just
be adding duplicates rather than genuinely new images, and specifically whether any
duplicates would end up split across train/test if merged carelessly.
"""
import csv
import hashlib
from pathlib import Path
from PIL import Image
import imagehash

ORIG_MANIFEST = Path(r"C:\Users\tishy\Documents\Honors\dataset\manifest_clean.csv")
NEW_MANIFEST = Path(r"C:\Users\tishy\Documents\Honors\SkinDisease\manifest_clean.csv")


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_eczema_paths(manifest, class_col="class"):
    with open(manifest, newline="", encoding="utf-8") as f:
        return [row["path"] for row in csv.DictReader(f) if row[class_col] == "Eczema"]


def main():
    orig_paths = load_eczema_paths(ORIG_MANIFEST)
    new_paths = load_eczema_paths(NEW_MANIFEST)
    print(f"Original dataset Eczema images: {len(orig_paths)}")
    print(f"SkinDisease dataset Eczema images: {len(new_paths)}")

    print("\nHashing original dataset's Eczema images...")
    orig_md5 = {}
    orig_phash = {}
    for p in orig_paths:
        try:
            with Image.open(p) as im:
                orig_md5[md5_of(p)] = p
                orig_phash[imagehash.phash(im.convert("RGB"))] = p
        except Exception as e:
            print(f"  skip {p}: {e}")

    print("Hashing SkinDisease's Eczema images and checking against original...")
    exact_dupes = []
    near_dupes = []
    for p in new_paths:
        try:
            digest = md5_of(p)
            with Image.open(p) as im:
                ph = imagehash.phash(im.convert("RGB"))
        except Exception as e:
            print(f"  skip {p}: {e}")
            continue

        if digest in orig_md5:
            exact_dupes.append((p, orig_md5[digest]))
            continue

        for existing_hash, orig_p in orig_phash.items():
            if ph - existing_hash <= 4:
                near_dupes.append((p, orig_p, ph - existing_hash))
                break

    print(f"\nExact duplicates (SkinDisease Eczema == original dataset Eczema): {len(exact_dupes)}")
    for p, o in exact_dupes[:10]:
        print(f"  {p}\n    == {o}")
    print(f"\nNear-duplicates (hamming distance <= 4): {len(near_dupes)}")
    for p, o, d in near_dupes[:10]:
        print(f"  {p}\n    ~= {o} (dist={d})")

    total_overlap = len(exact_dupes) + len(near_dupes)
    print(f"\nTotal overlap: {total_overlap} of {len(new_paths)} SkinDisease Eczema images "
          f"({100*total_overlap/len(new_paths):.1f}%)")
    print(f"If merged and overlap removed: {len(orig_paths) + len(new_paths) - total_overlap} unique Eczema images")


if __name__ == "__main__":
    main()
