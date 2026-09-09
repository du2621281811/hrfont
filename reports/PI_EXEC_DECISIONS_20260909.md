# PI execution decisions — 2026-09-09

Locked for implementation on this machine:

1. **F3b support chars**: preset CN subset + stroke-type bucket sampling (not topology top-16).
2. **F3b support features**: **own-font Ec only** (option B). Cross-bank train ≠ own-font infer.
3. **E12**: may **fit on main train228** StyleImage; narrative = same-domain protocol scorer (not open-world perfect reasonableness). **No CoAtNet**.
4. **F1**: resume from stopped_step → 80k (started).
5. **F3b run id**: `f3b_joint_crossbank_s3407` (name kept; features are own-font).
6. **F3b adapter**: standard random init (not zero-init).
7. **Schedule**: F3b launched with **80k horizon** from F0 (evaluate at 20k; can continue without reschedule).

Active GPUs (intent):
- GPU2: F1
- GPU0: F3b
- GPU1: E12 phi/membership on `artifacts/e12/cache_train228_v51`

Live status board: `reports/TRAINING_STATUS_F1_F3B_E12.md`.
