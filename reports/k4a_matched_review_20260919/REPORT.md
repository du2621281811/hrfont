# K4-A matched-bank visual review

All 30080 default/cross-checkpoint predictions and 288 intervention predictions completed. Local visual review: outputs/K4A_VISUAL_REVIEW/index.html, backed by sibling K4A_MATCHED_BANK_20260918 and K4A_INTERVENTION_MATCHED_20260918 assets. Browser tested with Chrome: no JavaScript exceptions or broken sampled images after decoding. Do not describe this sampled browser test as exhaustive human image acceptance.

A uses training v0917 donor bank with weight-only exclusions. K1 step0 uses original0913 bank and original exclusions. Thus cross-lineage K1-to-A comparison includes bank/data history changes; intra-A comparisons keep bank fixed. G0 is1shot only, not matched-shot against4shot. All inference uses repaired attention, with earlier training precision differences retained in provenance. Old v2-bank diagnostics remain historical, not the authoritative result.

The A-character visual inspection shows two different behaviors: FZBangSKLTJW loses the GT serif contour as A training progresses; FZPANGPBJW changes from conventional A toward the rounded outline but retains broken/incorrect connections. This establishes sample-specific mixed changes, not uniform regression over the dataset.

Local-path intervention changes generated glyph geometry. For FZPANGPBJW A, local-off changes the form and full reference swap produces the solid ordinary-font appearance. The path is used; this does not establish reader compression as the root cause. Local-only swaps retain global/Delta conditions, full swap changes the entire reference-conditioned input. off is local-style-off, not Delta-off.

Loss tables are baseline-condition GT-noised diagnostic forward passes at125/375/625/875, not losses of generated PNGs, not per-intervention losses and not historical training loss. L1 is image error. Frozen detail annotation is not an objective difficulty scale. Full human review remains with the user.

User subsequently authorized K5-A then K5-B on V2/original K1 recipe, retaining Top10/alpha; donor/representation diagnostics are deferred until both finish. No causal bank or encoder conclusions have been claimed from the current review.
