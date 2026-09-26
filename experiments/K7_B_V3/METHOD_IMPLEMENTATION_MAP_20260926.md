# Current Method ↔ K7-B implementation map

This map is anchored to the user-provided Method text (SHA-256 `adab5377e56a2ef145b0a7733e89be17a534705e90148095b776e693b79bc6b4`). The code references below were checked against the isolated source tree on `sitonholy` at `/root/projects/hrfont_k7_v3_20260922`.

The source handoff is a composition of the files under [`source/`](source/) plus the single current-server override under [`source_overlays/sitonholy_method_20260926/`](source_overlays/sitonholy_method_20260926/README.md). The original source snapshot and its identity manifest are preserved unchanged.

## Executed K7-B composition

`train_k6.py` selects `K7-B` and delegates model construction to `k6_runtime.py`. That runtime builds the `K5-B` model through `k5_runtime.py`; K5-B adds the parallel EC/ES structural branches to the K4-C base. The base construction reaches `scripts/k_runtime.py` → `scripts/hrfont_k.py` → `scripts/hrfont_i.py`, while `scripts/h_runtime.py` loads the FontDiffuser variant under `code/variants/cn2west_f123_rsi/`. This composition is why the Method's local reader, RMS calibration, dual CRP branches, and identity-safe skip update live in different source files.

## Component map

| Current Method component | Implementation and correspondence |
| --- | --- |
| Reference-conditioned candidate set and prior \(\alpha\) | [`k4_runtime.py`](source/experiments/K6/implementation/r3/scripts/k4_runtime.py): `Library.select` and `DataContext.conditions`; [`k4_family.py`](source/scripts/k4_family.py): family/weight exclusions; [`hrfont_delta_v2.py`](source/scripts/hrfont_delta_v2.py) and the selected FontDiffuser `train.py`: per-reference pooled descriptors and cosine-based `compute_alpha`. The selected fonts and prior are built before sampling and then reused. |
| EC/ES feature differences and two structural branches | [`k5_runtime.py`](source/experiments/K6/implementation/r3/scripts/k5_runtime.py), with its current-server copy in the overlay: `EsTargetFeatures`, `EsOffset`, and `DualOffset`; [`hrfont_i.py`](source/scripts/hrfont_i.py): `SetOffset`. K5-B creates separate routers and offset estimators, adapts the ES feature with a bias-free 1×1 layer, and combines the branch offsets with a learned tanh gate. The router adds the log prior to location- and timestep-conditioned cosine scores. |
| Identity-safe deformable skip update | [`build.py`](source/code/variants/cn2west_f123_rsi/FontDiffuser/src/build.py) selects `StyleRSIUpBlockIdentitySafe`; [`unet_blocks.py`](source/code/variants/cn2west_f123_rsi/FontDiffuser/src/modules/unet_blocks.py) applies DCN to the skip, passes the difference through a zero-initialized 1×1 projection, adds it back to the skip, then concatenates it with the decoder feature. |
| Local appearance reader, \(M\), and \(T\) | [`hrfont_i.py`](source/scripts/hrfont_i.py): `LocalMemory` and `IModel`. It uses the trainable local encoder copy, shallow/deep reference tokens, target-content queries plus learned slots, head-wise Q/K normalization, a residual MLP, and separate projections for the VGG-supervised readout and decoder tokens. |
| Global/local decoder attention and \(\kappa\) | [`hrfont_k.py`](source/scripts/hrfont_k.py): `KModel.forced_attention`; [`hrfont_h.py`](source/scripts/hrfont_h.py): shared `attention_once`. The same attention module computes global and local outputs in separate calls/softmaxes. The detached ratio is `RMS(global)/RMS(local)`, clipped to `[0.25, 4]`. The fixed coefficient is hard-coded as `0.8 * gate`; the training gate is supplied by `train_k6.py` and is one during inference. The config's `beta` string is not the executable implementation of this term. |
| Diffusion, perceptual, completion, and detail objectives | [`train_k6.py`](source/experiments/K6/implementation/r3/scripts/train_k6.py): objective assembly and target VGG teacher; [`k_components.py`](source/experiments/K6/implementation/r3/scripts/k_components.py) and [`i56_components.py`](source/experiments/K6/implementation/r3/scripts/i56_components.py): completion and detail calculations. The K7-B branch uses zero offset-loss weight, the fixed VGG perceptual coefficient, and the ramped completion/detail terms described in Method. |
| Inference-time reuse vs. per-denoising-step updates | [`eval_k6_protocol.py`](source/experiments/K6/implementation/r3/scripts/eval_k6_protocol.py): its sampler builds data conditions and model context before the solver loop, then calls `model.denoise` for each solver evaluation. Thus the candidate set, prior, and TC context are reused, while CRP routing and RMS calibration run inside the denoiser. |
| Inherited FontDiffuser components | [`unet.py`](source/code/variants/cn2west_f123_rsi/FontDiffuser/src/modules/unet.py), its `unet_blocks.py`, and the variant's MCA modules retain the inherited denoiser, MCA, and global-style path. These are not presented as new K7-B modules. |

The canonical imported local-reader file is `source/scripts/hrfont_i.py`. A similarly named copy under `source/experiments/K6/implementation/r3/scripts/` is not the module imported by `scripts/hrfont_k.py` in this launch path.

## Source and run provenance boundary

The base handoff contains 438 source files. A read-only audit against the sitonholy identity found one current-tree change: `k5_runtime.py`; that live file is preserved as a one-file overlay with its SHA-256 and delta summary. Its changes are in collision-safe cache writes and evaluation-reference selection; the EC/ES `DualOffset` method path is unchanged from the base snapshot.

This is a Method-aligned source snapshot, **not** a claim of byte-identical recovery of the 32k training process. The frozen run config records different hashes for `train_k6.py` (`4e278516…`) and `k5_runtime.py` (`ef268d67…`); the available current trainer is `4bdee200…`, and the current server overlay runtime is `5814b485…`. Keep this distinction when describing training provenance or claiming exact restartability.
