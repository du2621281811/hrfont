# K7-B V3: collaborator handoff / 协作交接

This is a **source-and-provenance snapshot**, copied from the V100 K7-B working tree on 2026-09-25. It is not a checkpoint release or a claim that a fresh machine can reproduce the training run bit-for-bit. Start with `python3 experiments/K7_B_V3/verify_package.py`; it is read-only and needs no GPU.

## What is here

| Path | Meaning |
| --- | --- |
| `source/` | 438 files named by `K7_CODE_IDENTITY.json`, byte-verified against the V100 source snapshot; includes K7 queue/cache scripts and the transitive K6/FontDiffuser runtime. |
| `source/experiments/K4/`, `source/experiments/K6/AUTHORIZATION_K7_B*.json` | Runtime policy inputs absent from the 438-file code identity. |
| `contracts/v2/`, `contracts/v0917/`, `contracts/v0921/` | Exact split/donor/pair/style manifest files, not images. V0921 is supplemental **train-only, non-bank** data. |
| `contracts/CALIBRATION.json`, `contracts/teacher/` | Calibration and frozen teacher normalization/manifest. |
| `ops/` | 50k continuation overlay and supervisor, separately copied from the V100 operations tree. |
| `provenance/` | Frozen run configs, 20k and 32k checkpoint metadata, and 20k→50k continuation manifest. |
| `verify_package.py` | SHA-256 audit and historical-code-drift disclosure. |

The snapshot does **not** include image datasets, encoder/model weights, feature caches, optimizer/EMA checkpoints, logs, or generated evaluation images. These are large external artifacts. Do not mistake the JSON checkpoint metadata for checkpoint tensors. Paths embedded in configs are V100 absolute paths, not portable settings. `source/scripts/hrfont_h.py` and the K6 runtime use `/root/projects/hrfont` and its `artifacts/h_20260915` layout; deployment requires deliberately staging external assets at matching paths or adapting them and revalidating hashes.

## Architecture and training path

The model inherits FontDiffuser's U-Net, Ec/Es conditioning, and MCA-style encoder/middle fusion. K7-B's additional paths are **TC/local appearance memory** and **CRP/structural candidate routing**; do not label inherited MCA as a K7-B invention. The authoritative implementations are `source/experiments/K6/implementation/r3/scripts/hrfont_i.py` (`LocalMemory`, `SetOffset`, `IModel`), `k5_runtime.py` (base dual-encoder construction), `k_components.py` (extra losses and detail distances), `train_k6.py` (training orchestration), and `k4_runtime.py` (dataset/bank selection).

TC: shallow and deeper reference-encoder maps produce 24²+12² = 720 tokens **per valid reference image**. The frozen target-content feature supplies a 12×12 query, augmented with 144 learned position slots. Four-head attention reads all valid reference tokens into 144 local 256-D vectors; an MLP residual refines them. Separate linear projections yield `T` (144×1024 local U-Net cross-attention tokens) and `M` (144×128 appearance readout). Right/up-block cross-attention adds a gated local-attention output to the inherited global-style attention output (numeric sum, not concatenation). `M` is supervised by detached, normalized, 12×12 pooled second-block VGG16 features of the target glyph. These 128 dimensions come from that VGG feature stage, not the U-Net.

CRP: the structure bank remains based on frozen V2 donors; V0921 samples do not become donors. Ec/Es structural candidates and a neutral candidate feed `SetOffset`, which computes target/location/time-conditioned logits plus log prior weights, masks unavailable donors, softmaxes over candidates, and passes the mixed style to the existing offset estimator. Its `[B,18,H,W]` offsets act through the RSI/DCN skip pathway. TC and CRP affect selected U-Net up blocks; MCA remains in the inherited backbone. The `up_blocks`/`sc_interpreter_offsets` code is the final authority for exact layer placement.

The training objective in the frozen configs is epsilon MSE plus `0.01 ×` VGG perceptual feature MSE, plus ramped `0.01 ×` completion Smooth-L1 (`M` vs detached VGG teacher) and `0.05 ×` region/detail loss on reconstructed `x̂₀` and ground-truth glyph. The detail code compares pixels/edges/high-frequency structure at native 96 and downsampled 48 scales, with target-derived region weighting. Offset loss weight is **0**; the logged offset value is not an active training term. Pure-noise auxiliary training is disabled. See `train_k6.py` and `k_components.py` for the exact masking, timestep gating, and reductions. The perceptual and detail terms both use reconstructed output but constrain different representations; neither is the `M` supervision.

## Data, lineage, and evidence

The 20k run is K7-B-V3-K0-S3407, 8 V100 ranks, global batch 64, from the frozen F0 10k parent. Its config and completion metadata are in `provenance/`. The authorized 50k continuation starts from 20k; the supplied continuation manifest records its overlay, authorization, and trainer hashes. The latest **captured complete checkpoint metadata here is 32k**, not 50k completion. This package does not certify a current live process or final model quality. Keep training-health, fixed-protocol metrics, and visual judgment distinct.

The V2 manifests fix train/VAL/test and donor identity. V0921 adds training pairs only: `role=train_only_nonbank`, `bank_inclusion=false`. `v0917` is retained as evaluation/data context. Ec/Es cache identities in `config_*.json` matter: do not reuse a legacy K4 cache when encoder hashes differ. Before running, restore the exact external dataset roots, F0 parent weights, teacher artifacts, VGG weights, and encoder caches, then compare all hashes and bank policy against the frozen config. Never silently replace a missing asset with a convenient newer one.

One naming trap: `data_identity.family_groups_sha256` in the run config hashes `source/experiments/K4/weight_groups.json` (as `k4_runtime.py` shows), **not** the separate `family_groups.json` file. Both files are included.

## Critical reproducibility limitation

The present 438-file `K7_CODE_IDENTITY.json` matches the copied V100 source **exactly**, but its historical hash list embedded in both frozen 20k/50k run configs differs for two files:

| File | Frozen run SHA-256 | Current source SHA-256 |
| --- | --- | --- |
| `experiments/K6/implementation/r3/scripts/train_k6.py` | `4e278516f586f6e7c0b8c4cf6746bbbad772c711472c2615c1d918300db14d2e` | `4bdee2008dab18edfab47eb63f09cbdcb052f578f0d4e371d7d357ea9c8cc679` |
| `experiments/K6/implementation/r3/scripts/k5_runtime.py` | `ef268d67e7e420209beee63178ccefd2ce345baf16a7f901fc9b1cf07fc6968a` | `bf283144e97da1f3a42895935752504c51450d1fef385e03a6a486b1c6abd9df` |

The original bytes of these two historical files were not found in the inspected V100 tree. The continuation manifest explicitly pins the **frozen** trainer hash. Thus the checked-in current source is valuable for implementation review, but **must not be represented as the exact train-time source or used to claim bit-exact restart**. In particular, the 50k overlay may expect old trainer text and should not be run against this snapshot without a reviewed compatibility patch. Recovering those historical bytes, plus external weights/data/checkpoints, is required for strict reproduction. `verify_package.py` deliberately reports this drift separately from copy corruption.

## Suggested review order

1. Run `python3 experiments/K7_B_V3/verify_package.py` from the repository root.
2. Read `source/experiments/K6/implementation/r3/scripts/hrfont_i.py`, `k_components.py`, `train_k6.py`, and `k4_runtime.py` alongside `provenance/config_20k.json`.
3. Inspect `contracts/v2/` and `contracts/v0921/` before any comparison or bank change; check the `data_identity` hashes in the frozen config.
4. Treat `ops/` as operational provenance, **not** a ready-to-run recipe. Obtain historical two-file source and external tensors before restarting or claiming exact reproduction.

No script in this handoff launches training when merely imported or when the integrity audit is run.

## FZ49 V4 evaluation

The reviewed V4 FZ49 metrics, sample filter, target-alignment audit, per-sample tables, rankings, and provenance are in [`evaluation/FZ49_V4_20260925/`](evaluation/FZ49_V4_20260925/README.md). This is a metrics/provenance package only; prediction images and model/data assets remain external.
