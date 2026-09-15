# HR-Font editable Bezier variant

Status: implemented MVP architecture; not trained and not a result claim.

## Contract

The raster FontDiffuser/HR-Font conditioning backbone remains frozen during
Stage V1. A vector condition bridge consumes the same cached style/content and
retrieval mean-Delta features as the current F1/F2/F3 path. A Transformer
deforms the neutral font's ordered cubic control points and predicts normalized
`[advance_width, lsb, rsb]`.

In addition to global condition tokens, every control point bilinearly samples
the multi-scale Content and structure/Delta fields at its own canvas position.
This preserves the spatial role of Set-Delta instead of collapsing it to a
single global vector before outline decoding.

The first version is deliberately fixed-topology. It can learn width, weight,
slant, curvature, terminal and serif-like deformation, but cannot honestly claim
single/double-storey conversion, hole creation, or contour deletion. Those need a
separately controlled topology-edit head.

## Sidecar layout

```text
<vector_root>/<split>/ContentOutline/<cp>.npz
<vector_root>/<split>/TargetOutline/<font>/<font>+<cp>.npz
```

Every NPZ contains:

- `points [N,2]`: canvas-normalized, y-up coordinates;
- `point_types [N]`: 0 endpoint, 1 cubic off-curve control;
- `contour_ids [N]`;
- `segments [S,4]`: point indices `[start,c1,c2,end]`;
- `segment_contours [S]` and `adjacent_segments [A,2]`;
- `metrics [3]`: advance width, LSB and RSB normalized by UPM;
- `render_transform` and `upm` for provenance/export.

Quadratic TrueType curves are exactly converted to cubic curves. Composite glyphs
are decomposed by the fontTools glyph set. Missing or empty glyphs fail closed.

## Training

1. Extract ContentOutline from the exact Noto Content font and frozen A-protocol
   font size. Extract every TargetOutline from its source TTF/OTF and corresponding
   A-protocol font size.
2. Run `python scripts/test_bezier_variant.py`.
3. Start `train_bezier.py` with a compatible raster checkpoint and its verified
   Es/Ec caches. Stage V1 freezes the raster model and trains the condition
   bridge, per-point multi-scale samplers and vector decoder. `--condition-mode
   online` is only a no-Delta smoke-test fallback.
4. Validate rendered identity, point count, self-intersections, holes and metrics
   before considering selective joint fine-tuning.

Example (artifact paths are placeholders):

```bash
python code/variants/hrfont_bezier/FontDiffuser/train_bezier.py \
  --data-root data/fontdiffuser-p253-t295-s338-cn2west-v2 \
  --vector-root data/fontdiffuser-p260-bezier-v1 \
  --split-manifest manifests/split_v3_228_16_16.json \
  --raster-ckpt runs/F2-DELTARSI-A-S3407/best \
  --es-cache-path artifacts/f0/es_spatial_f0 \
  --ec-cache-path artifacts/f0/ec_multiscale_f0 \
  --output-dir runs/V1-BEZIER-S3407
```

Editable inference defaults to the same cached mean-Delta path. Supply one
`--style-chars` item for each `--style-images` item so retrieval can compare
same-character style embeddings. The command writes SVG, GLIF and NPZ outputs.

## Before formal training

- perform a lineage-grouped split audit;
- bind TTF hashes, UPM, cmap, A font size and extraction code SHA;
- compare every sidecar soft render with its frozen A96 PNG and reject alignment
  outliers;
- add verified SDF/boundary and self-intersection losses;
- register new dataset/run IDs in provenance;
- do not describe the MVP as topology-generative.
