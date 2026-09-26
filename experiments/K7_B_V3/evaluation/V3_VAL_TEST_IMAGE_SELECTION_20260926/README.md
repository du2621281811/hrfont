# K7-B V3 VAL / TEST image-selection package

This package contains the completed V3 validation and test predictions for four models: `K7-B`, `A1`, `A2`, and `A3`. It is for side-by-side visual review and sample selection.

## Open the gallery

Open `index.html` in a browser. Choose VAL or TEST, optionally narrow the batch, script, font, or character, and review the four model outputs for each target side by side. Select sample cards and export the selected sample list as CSV or JSON. The page shows 24 samples at a time so it does not load the entire image collection into the browser at once.

The gallery and `image_index.csv` use relative paths and contain no machine-specific source paths. The original prediction PNGs are preserved unchanged. Click an image to open its full-size file.

## Coverage

| Split | Batch | Samples per model | Models | Prediction images |
| --- | --- | ---: | ---: | ---: |
| VAL | 0913 | 4,123 | 4 | 16,492 |
| VAL | 0917 | 3,208 | 4 | 12,832 |
| TEST | 0913 | 3,880 | 4 | 15,520 |
| TEST | 0917 | 3,208 | 4 | 12,832 |
| **Total** |  | **14,419** | **4** | **57,676** |

Each sample has one prediction from each model. Matching font/codepoint filenames make the four outputs directly comparable. VAL and TEST remain in separate directories and are never merged by the gallery filter.

## Files

- `index.html` — self-contained, filterable visual-selection page; works from a local checkout without a server or external dependencies.
- `image_index.csv` — one row per font/character sample, with split, batch, script, and relative prediction paths for all four models.
- `gallery_template.html` and `build_gallery.py` — page template and standard-library rebuild script; run `python3 build_gallery.py` from this directory after editing the CSV.
- `SHA256SUMS` — SHA-256 checksums for every package file except this checksum list.
- `VAL/` and `TEST/` — unchanged prediction PNGs, organized as `<split>/<batch>/<model>/`.

To verify the package from this directory, run `shasum -a 256 -c SHA256SUMS`.
