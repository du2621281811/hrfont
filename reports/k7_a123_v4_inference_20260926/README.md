# K7 A1/A2/A3 V4 full inference results

This archive contains the complete 4k and 10k inference PNGs for A1, A2, and A3 on the matched V4 evaluation set: 6 conditions, 8,638 samples per condition (51,828 images total), spanning 47 fonts. The separate 20k rows in the metric summary are existing comparison results; this upload does not claim to include a new 20k inference image batch.

## Contents

- `bundle.tar.gz.part-*`: split compressed archive with the A1/A2/A3 prediction folders, per-condition metrics/protocol/DONE records, full per-sample metrics, aggregate metric summary, manifests, and run specification/completion record.
- `SHA256SUMS`: checksums for every archive part.

## Restore

From this directory, join and unpack the parts:

```sh
cat bundle.tar.gz.part-* > bundle.tar.gz
tar -xzf bundle.tar.gz
```

The archive expands to the original run layout (`A1/`, `A2/`, `A3/`, `metrics/`, and `manifests/`).

## Evaluation protocol

Matched V4, 8-shot, per-sample fixed references and seed 3407, DPM++ 20 steps/order 2. Primary aggregate is font-macro: average within each font, then equal-weight across 47 fonts. Metrics: L1, SSIM, LPIPS, DINO distance, and RSC-8/E12-C. Lower is better for L1/LPIPS/DINO distance; higher is better for SSIM/RSC-8. No model checkpoints are included.
