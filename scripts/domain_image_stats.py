"""Extended image-statistic comparison across Internal / SCIN / SkinDisNet, building on
the brightness-only comparison already done in docs/external_validation_2026-09-15.md
(diagnose_external_shortcut.py). Adds saturation, native resolution/aspect ratio, and a
high-frequency-energy (Laplacian variance) proxy for compression/sharpness differences --
simple hand-crafted signals to check whether any one of them alone tracks the CNN-embedding
domain gap computed in domain_shift_analysis.py, or whether the gap is subtler than that.

Reads the same image population cached by extract_domain_embeddings.py (same paths/sources
arrays), so results are computed on an identical population to the A-distance analysis.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import laplace

from paths import ROOT

EMBEDDINGS_PATH = ROOT / "dataset" / "embeddings_internal_scin_skindisnet.npz"
RESIZE_TO = (128, 128)


def load_rgb01(path):
    img = Image.open(path)
    native_w, native_h = img.size
    img = img.convert("RGB")
    resized = img.resize(RESIZE_TO)
    arr = np.asarray(resized, dtype=np.float64) / 255.0
    return arr[..., 0], arr[..., 1], arr[..., 2], native_w, native_h


def rgb_to_hsv_saturation(R, G, B):
    maxc = np.maximum(np.maximum(R, G), B)
    minc = np.minimum(np.minimum(R, G), B)
    delta = maxc - minc
    sat = np.where(maxc == 0, 0.0, delta / np.where(maxc == 0, 1.0, maxc))
    return sat


def image_stats(path):
    R, G, B, w, h = load_rgb01(path)
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    brightness = float(gray.mean())
    saturation = float(rgb_to_hsv_saturation(R, G, B).mean())
    aspect_ratio = float(w / h) if h else float("nan")
    hf_energy = float(np.var(laplace(gray)))
    return dict(brightness=brightness, saturation=saturation, width=w, height=h,
                aspect_ratio=aspect_ratio, hf_energy=hf_energy)


def main():
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    paths = data["paths"]
    sources = data["sources"]

    results = {s: {"brightness": [], "saturation": [], "width": [], "height": [],
                    "aspect_ratio": [], "hf_energy": []} for s in np.unique(sources)}

    n_failed = 0
    for i, (path, source) in enumerate(zip(paths, sources)):
        try:
            stats = image_stats(str(path))
        except Exception:
            n_failed += 1
            continue
        for k, v in stats.items():
            results[source][k].append(v)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(paths)}")

    print(f"\nFailed to read: {n_failed}/{len(paths)}")

    print("\n=== Image statistics by dataset ===")
    print(f"{'dataset':>12} {'n':>6} {'brightness':>11} {'saturation':>11} "
          f"{'median_w':>9} {'median_h':>9} {'aspect':>8} {'hf_energy':>11}")
    summary = {}
    for source in results:
        r = results[source]
        n = len(r["brightness"])
        row = dict(
            n=n,
            brightness_mean=float(np.mean(r["brightness"])),
            brightness_std=float(np.std(r["brightness"])),
            saturation_mean=float(np.mean(r["saturation"])),
            saturation_std=float(np.std(r["saturation"])),
            width_median=float(np.median(r["width"])),
            height_median=float(np.median(r["height"])),
            aspect_ratio_mean=float(np.mean(r["aspect_ratio"])),
            hf_energy_mean=float(np.mean(r["hf_energy"])),
            hf_energy_std=float(np.std(r["hf_energy"])),
        )
        summary[source] = row
        print(f"{source:>12} {n:>6} {row['brightness_mean']:>11.4f} {row['saturation_mean']:>11.4f} "
              f"{row['width_median']:>9.0f} {row['height_median']:>9.0f} "
              f"{row['aspect_ratio_mean']:>8.3f} {row['hf_energy_mean']:>11.6f}")

    np.savez(ROOT / "dataset" / "domain_image_stats_summary.npz", **{
        f"{src}__{k}": v for src, row in summary.items() for k, v in row.items()
    })
    return summary


if __name__ == "__main__":
    main()
