# E12-C V0916 inference package

This directory contains the selected E12-C inference checkpoint pair for run `E12C-V0916-S3407`:

- `A_best.pt`: Stage-A encoder selected on the fixed internal validation split.
- `B_best.pt`: Stage-B scoring head selected on the fixed internal validation split and bound to the encoder SHA below.
- `DONE.json`, `config.json`: run completion, weights, data-manifest, preprocessing, and training-code provenance.

The inference implementation is in the repository at `scripts/score_e12c.py`. It uses `scripts/e12c_model.py`, `scripts/train_e12c.py` (the `Images` preprocessing implementation), and `scripts/new_data_inventory.py` (SHA verification). From the repository root:

```bash
python scripts/score_e12c.py \
  --checkpoint artifacts/e12c_v0916_s3407 \
  --query /path/to/generated_glyph.png \
  --refs /path/to/chinese_ref_1.png /path/to/chinese_ref_2.png \
  --device cpu
```

Supply 1–8 actual Chinese reference glyph images. The output includes the primary compatibility `logit`, the diagnostic `cosine_diagnostic`, and the shot count. The logit is not a calibrated probability or an overall glyph-quality score.

SHA-256:

- `A_best.pt`: `1f68845826e67c893bcb77af8f8363f0115cfee154daef7cb9b83f8bd3b44c4d`
- `B_best.pt`: `1d598833064d86216f9144c323b59f2127df46214c8f5339921d1748f065c268`
- `scripts/e12c_model.py` at training: `79a866528913bc7111220df3e71b598b9ce65d68a54c48a94f00e3d07c45df12`
- `scripts/train_e12c.py` at training: `46b6227af72aa6a2f78dac17041f169a1f804300124d3018cf99d2f86aa5cfd7`
- `scripts/e12c_data.py` at training: `56c01b87739c8974233e3c2408d805c128637c2d01d5b45c627f35604440709c`
- `scripts/e12c_metrics.py` at training: `35343f242eb1b3ace20878fbd1d138fce593ce028e21f6979f62fd76a65d9b3c`
