# Checking the New "Human Skin Diseases (Image)" Dataset

Source: Kaggle, `youssefmohmmed/human-skin-diseases-image`, downloaded to
`SkinDisease/{train,valid,test}/<class>/`. 20 classes, own pre-made train/valid/test
split, 17,266 images total (13,114 train / 2,556 valid / 1,596 test).

## Headline finding: the disease classes look good, but "Normal" is badly contaminated — exclude it

### Disease classes: consistent, single-source, usable

Filenames follow the same convention across Eczema, Psoriasis, Rosacea, Lichen, etc.
(e.g. `eczema-fingertips-49.jpeg`, `psoriasis-scalp-25.jpeg`, `rosacea-22.jpeg`) — the
same pattern as the *original* Eczema/Normal dataset's DermNet-sourced files. Spot-checked
5 classes directly: all clinical macro close-ups on dark backgrounds, several with the
same `©Dermnet.com` watermark seen before. This is genuinely good news — it means
**Eczema vs. other-disease classes in this dataset are plausibly a same-source
comparison**, which is exactly the fix the shortcut-learning problem needed.

Quantitative check across all 20 classes (`scripts/check_normal_class_content.py`, a
skin-colour-fraction screen applied identically to every class): every disease class has
a mean skin-pixel fraction of 0.58–0.90 and fewer than 8.4% of images below a
"probably not a skin photo" threshold (2% skin coverage). Consistent with close-up
lesion photography across the board.

### "Normal" class: contaminated with unrelated stock photos, and internally inconsistent across its own splits

Spot-checking 20 "Normal" images turned up an unrelated stock/product photo for the
**majority** of them: a drill kit, a dish rack, a frying pan, a vacuum cleaner, a stack of
towels, a train, a cat, an elephant, cherries on a branch. Only a handful were actually
skin photos (a face selfie, a palm, fingernails, a leg with a hair-removal device).

The quantitative check confirms this isn't cherry-picked bad luck: **Normal's mean
skin-pixel fraction is 0.389 and its median is 0.286 — roughly half of every disease
class's median (0.73–0.99), and by far the lowest of all 21 classes.** 12.0% of Normal
images have essentially no skin-coloured pixels at all (vs. 0.2–7.2% for every disease
class), and 16.6% are below a slightly looser threshold (vs. 0.4–8.4% elsewhere).

| Class | n | mean skin-fraction | median | % below 2% skin | % below 5% skin |
|---|---|---|---|---|---|
| **Normal** | 1,732 | **0.389** | **0.286** | **12.0%** | **16.6%** |
| Eczema | 1,064 | 0.804 | 0.835 | 0.6% | 0.7% |
| Vitiligo (lowest disease class — genuinely depigmented skin, expected) | 796 | 0.581 | 0.562 | 0.6% | 1.0% |
| Benign_tumors (2nd-highest contamination among disease classes) | 1,231 | 0.750 | 0.968 | 7.2% | 8.4% |
| *(all other 17 disease classes fall between these)* | | | | ≤5.6% | ≤7.4% |

**Worse: "Normal" is a different sub-source in each split**, which breaks it even
further as a usable class:
- `train`/`valid`: filenames like `Image365.jpeg` — the mixed junk-photo content above.
- `test`: filenames like `0_0_anhu_0216.jpeg`, `0_0_baobeier_0020.jpeg` — a completely
  different source, low-resolution cropped face thumbnails (apparently from a face
  dataset, judging by the pinyin-style filenames), not stock-photo junk, but also not at
  all the same style/resolution/framing as train's images.

A model trained on train's "Normal" (mostly random objects with occasional real skin) and
evaluated on test's "Normal" (low-res cropped faces) would not even be evaluating the
same distribution it was trained on — this isn't just a stylistic confound like the
original dataset's Eczema-vs-stock-photo problem, it's a fundamentally broken class
regardless of what comparison it's used in.

## Recommendation

**Drop this dataset's "Normal" class entirely.** Don't use it as a stand-in for
healthy/normal skin in any model. Build the Eczema-vs-other-skin-condition model from
this dataset's 20 disease classes only (or a curated subset of visually-similar ones, per
the earlier discussion) — that comparison has real same-source support. If a genuine
normal-skin class is still wanted later, it needs its own properly curated,
single-source data (the original dataset's `Normal/` folder remains a documented,
already-flagged alternative, or a new source entirely) — not a patch job filtering this
class down to its "good" ~80%, since the remaining images would still span at least two
inconsistent sub-sources (train/valid's incidental real-skin photos vs. test's low-res
face crops).

## Status

- `scripts/check_normal_class_content.py` — the skin-fraction screen above, run and
  complete. Output: `SkinDisease/skin_content_check.csv` (per-image skin fraction).
- `scripts/clean_skindisease.py` — corruption/duplicate/near-duplicate check (same
  methodology as the original dataset's `clean_dataset.py`, extended to also check for
  cross-split duplicates specifically, since a train/test leak here would be a real
  evaluation-validity problem). Still running at the time of writing (17,266 images is
  ~6x the original dataset; the near-duplicate check scales worse than linearly). Will
  update this doc with final counts once it completes.
- Not yet done: deciding the exact "other-disease" class subset (all 19 non-Eczema,
  non-Normal classes vs. a curated visually-similar subset), and building the actual
  Eczema-vs-other-disease classifier.
