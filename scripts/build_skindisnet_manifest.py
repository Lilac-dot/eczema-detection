"""
External-validation manifest for SkinDisNet (Sultana et al., Data in Brief 2025;
DOI 10.1016/j.dib.2025.112239; Mendeley DOI 10.17632/yj3md44hxg). Clinical smartphone
photos from two Bangladesh hospitals -- genuinely independent of the DermNet-style
curated archive the deployed Stage B model was trained on.

Only the "Preprocessed" folder is used (1,710 real clinical images); the "Augmented"
folder (11,970 images) is synthetic data-augmentation of the same 1,710 photos and would
just inflate the external test set with near-duplicates, not add genuine new evidence.

Harmonization: "Eczema" and "Atopic Dermatitis" are merged into label=1. The report
(Section on SkinDisNet review) already notes this split is clinically unexplained in the
source paper, so treating them as one positive class is the defensible choice rather than
guessing which is "real" eczema. "Contact Dermatitis", "Scabies", "Seborrheic Dermatitis",
and "Tinea Corporis" become label=0.

Writes: SkinDisNet/manifest_skindisnet.csv (path, disease_class, label)
"""
import csv

from paths import SKINDISNET_DIR

POSITIVE_CLASSES = ["Eczema (EC)", "Atopic Dermatitis (AD)"]
NEGATIVE_CLASSES = ["Contact Dermatitis (CD)", "Scabies (SC)", "Seborrheic Dermatitis (SD)",
                    "Tinea Corporis (TC)"]


def main():
    preprocessed = SKINDISNET_DIR / "Preprocessed"
    rows = []
    for cls in POSITIVE_CLASSES + NEGATIVE_CLASSES:
        cls_dir = preprocessed / cls
        files = sorted(p for p in cls_dir.iterdir() if p.is_file())
        label = 1 if cls in POSITIVE_CLASSES else 0
        for p in files:
            rows.append({"path": str(p), "disease_class": cls, "label": label})
        print(f"{cls}: {len(files)} images (label={label})")

    manifest_path = SKINDISNET_DIR / "manifest_skindisnet.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "disease_class", "label"])
        w.writeheader()
        w.writerows(rows)

    n1 = sum(1 for r in rows if r["label"] == 1)
    n0 = sum(1 for r in rows if r["label"] == 0)
    print(f"\nManifest written: {manifest_path}")
    print(f"Eczema(+AD)={n1}, Other={n0}, total={len(rows)}")


if __name__ == "__main__":
    main()
