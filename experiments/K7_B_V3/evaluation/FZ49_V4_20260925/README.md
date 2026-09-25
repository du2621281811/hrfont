# FZ49 V4 evaluation package

This package records the final V4 comparison on the FZ49 test set. It contains metric tables and provenance only; prediction images, datasets, model weights, and feature caches are intentionally not included.

## Test set and aggregation

The V4 FZ49 evaluation panel contains 8,638 font--character samples across 47 fonts and 284 characters: 3,542 Latin, 4,726 kana, and 370 bopomofo samples.

The final comparison contains 15 methods, each evaluated on every sample. The selected FontDiffuser comparison is **FontDiffuser 45K** (`fontdiffuser_45k`). FontDiffuser P2 remains a separately identified result row and is not merged into the 45K result. Summary values first average over characters within each font, then average fonts with equal weight (`font-macro`). Rankings use competition rank; lower is better for L1, LPIPS, and DINO distance, while higher is better for SSIM and RSC-8.

## Metric provenance and caveats

- L1 was recomputed for all 129,570 final method/sample predictions against the shared K7-B target, using grayscale 96×96 images, values normalized to [0,1], and mean absolute pixel error. `all_methods_l1_audit.csv.gz` records original-versus-recomputed parity for the 14 methods with pre-existing L1 inputs; FontDiffuser 45K has newly computed L1 values in the complete per-sample table.
- FontDiffuser 45K predictions were sourced from collaborator commit `b7078a875e0bb28df206d864890233ae4452cfb1` on `data/v0921-texiao-supplement`. The commit contains 400 regular PNG files; 378 correspond to V4 samples and complement the regular release images. The regular release and supplemental images together cover all 8,638 V4 samples. FontDiffuser P2 remains a distinct result row. Per-file hashes for the supplemental PNGs are recorded in `fontdiffuser_release_supplement_sha256.csv`.
- Read-only V100 hash checks found that some `shot_1`, `shot_2`, and `shot_4` target snapshots differ from the shared K7-B targets. Target-dependent values for affected V4 rows were recomputed against the shared target (L1 for all rows; SSIM/LPIPS and DINO for the affected rows). RSC-8 is target-independent. The row-level evidence is in `v4_target_alignment_audit.csv`, `v4_metric_recompute_parity.json`, and `v4_provenance.json`.
- LPIPS and DINO use the existing scoring pipeline. The collaborator's private scoring scripts/configuration were not available for independent end-to-end reproduction, so the package reports local recomputations only for target-mismatch corrections and includes a matched-row parity check; it does not claim a complete independent audit of the collaborator's private protocol.
- `v4_per_sample_metrics.csv.gz` and `all_methods_l1_audit.csv.gz` omit machine-specific prediction/target paths. Sample identifiers and metric values are preserved. The target-alignment audit retains mismatch/recomputation evidence but omits absolute paths. No image assets are bundled.

## Files

| File | Contents |
| --- | --- |
| `V4_REVIEW.md` | Human-readable comparison, all overall metrics/ranks, domain leaders, and interpretation boundaries. |
| `v4_per_sample_metrics.csv.gz` | Per-sample V4 metrics for all 15 methods, with original input values retained in `*_input` columns where available. |
| `v4_metric_rankings.csv` | Scores and ranks by metric and scope. |
| `v4_font_macro_summary.csv` | Font-macro metric means by method and scope. |
| `v4_sample_manifest.csv.gz` | Portable V4 sample/font/character/script/dataset manifest. |
| `v4_target_alignment_audit.csv` | Per-sample target mismatch flags and metrics recomputed for alignment. |
| `all_methods_l1_audit.csv.gz` | Full-panel L1 input-versus-recompute audit for the 14 methods with pre-existing input scores, with machine paths removed. |
| `fontdiffuser_release_supplement_sha256.csv` | SHA-256 inventory of the 400 regular PNGs supplied by the collaborator commit. |
| `v4_metric_recompute_parity.json` | Matched-row parity check for target-dependent metric recomputation. |
| `v4_provenance.json` | Evaluation, aggregation, ranking, L1 protocol, and target-alignment provenance. |
| `SHA256SUMS` | SHA-256 checksums of all package files except this checksum list. |

All counts, averages, and ranks in this package describe this fixed V4 panel. They are descriptive comparisons, not significance tests or proof that a sample/ground truth is invalid. Shot settings with different reference sets should not be read as one causal shot-count curve; cross-method ranking does not replace matched ablation attribution or visual review.
