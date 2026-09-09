# PI execution decisions — 2026-09-09

Locked for implementation on this machine:

1. **F3b support chars**: topology ranking top-16 (pinned). Stroke-bucket bank is a control only.
2. **F3b support features**: **own-font Ec only** (option B). Cross-bank train ≠ own-font infer.
3. **E12**: may **fit on main train228** StyleImage; narrative = same-domain protocol scorer (not open-world perfect reasonableness). **No CoAtNet**.
4. **F1**: resume from stopped_step → 80k (running; do not interrupt).
5. **F3b run ids**:
   - **stopped:** `f3b_joint_crossbank_s3407` (stroke-bucket, own-font) at `stopped_step` 10883.
   - **queued after F1:** `f3b_topology_ownfont_s3407` from F0, bank `artifacts/f0/support_bank_f3b_topology.json`.
6. **F3b adapter**: standard random init (not zero-init). Injection = up-path style attention concat, not RSI.
7. **Schedule**: new F3b launches with **80k horizon** from F0 after F1 completes (evaluate at 20k; can continue without reschedule).

Active GPUs (intent):
- GPU2: F1
- GPU0: F3b
- GPU1: E12 phi/membership on `artifacts/e12/cache_train228_v51`

Live status board: `reports/TRAINING_STATUS_F1_F3B_E12.md`.
