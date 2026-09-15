# Paper experiments — I series, 2026-09-15

Authoritative implementation: [I specification](../../reports/I_EXECUTION_20260915.md). [Cross-series tracker](../../reports/EXPERIMENT_TRACKER.md).

| Run | Initialization | Role | Budget / status |
|---|---|---|---|
| I0 | G0b V0913 step10000 | Parent reference | 0 new updates; matched inference pending |
| I1 | I0 | Dynamic Delta + online spatial TC | 10k successful updates; running on 8 V100 |
| I2 | I0, independent of I1 | Same model + Chinese auxiliary | 10k planned; formal launch HOLD for rendering review |

I1: global64, inherited LR2e-5/new LR1e-4, FP16, 500 warmup, plateau to5k then cosine to10k. I2 adds CN32 at loss weight0.5. Both use spatial TC supervision weight0.01 ramped over1k.

I1 continues training/inference. Do not remove the queue STOP before PI approves [render review](../../reports/review_20260915/I2_RENDER_REVIEW.md). Existing queue executes the I2 20-step preflight before checking STOP; preflight does not authorize formal I2. No running model/trainer is hot-edited.

The 192-image milestone and 4096-image final panels are **validation**, not test: fixed reference manifests, EMA, DPM++20, CFG7.5, paired noise. Final test evaluation is a separate run after selection.

## Targeted contribution studies after full-model quality review

1. Dynamic versus fixed Mean-Delta over identical candidates.
2. Spatial completion objective versus generation loss only, same reader.
3. Trainable versus frozen local encoder.
4. Content-plus-slot versus slot-only queries.
5. Shared-parent structural-only / appearance-only / full model if main gains warrant the budget.
6. Same-bank absolute-feature control for anchor-relative conditioning.

These are proposed, not launched. Assign IDs after approval. I0 versus I1 alone is not compute-matched: I1 receives extra training. Historical G2/G2-RL/H images provide context, not substitutes for current shared-parent controls.

## Evaluation readiness

- Archive all PNGs/GT/sample keys/checkpoint hashes under I; include ordinary, decorative and difficult matched examples.
- E12-b is trained. Resolve [review findings](../../reports/review_20260915/E12_REVIEW.md) before final evaluator tables; cosine is not a calibrated probability.
- I scoring must read actual per-output refs and use shared coverage. Latin validation does not establish kana/phonetic validity.
- Correct tied-rank statistics and extend the existing39-item/single-rater pilot; current association is near zero.
- Record extra CN compute, GPU-hours, latency and selected artifact hashes. Keep missing result cells blank.
