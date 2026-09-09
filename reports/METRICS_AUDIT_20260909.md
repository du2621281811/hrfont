# Metrics audit — 2026-09-09

## Verdict
Implementations are **correct for their stated roles**. One display bug fixed (E1 `latin_ext` bucket). Best cells bolded on Glyph Board / PI briefing; Portal already marks `.best`.

## Pixel diagnostics (vs GT)
| Metric | Formula | Check |
|---|---|---|
| L1↓ | mean \|gray(pred)−gray(gt)\| /255 | Hand recompute 1 sample: **exact match** |
| SSIM↑ | **Global** gray SSIM (not sliding-window) | Aggregates match items; treat as diagnostic only |
| LPIPS↓ | Alex net, RGB in [-1,1] | Summary means match items (752/method) |

- All methods share identical 752 (font,char) keys.
- Directions: L1/LPIPS lower better; SSIM higher better.
- **Best (generators):** L1=E1 `0.0685`; SSIM=F3 `0.6536`; LPIPS=F3 `0.1437`.

## E12 v5.1 (style)
| Metric | Formula | Check |
|---|---|---|
| φ SC-R↑ | cos(φ(query), mean(φ(ref8))) | CPU rescore vs saved: **~1e-4** |
| mem_prob↑ | σ(logit / T), T≈2.249 | Same |
| mem_prob L+D↑ | subset Latin+digit (n=512) | Primary claim column |

- Protocol: query=pred\|GT, refs=target-font StyleImage ref8; test16 ∉ E12 train228.
- Not mixed into a total with L1.
- **Best generators:** φ=F2 `0.702`; mem / L+D mem=F3 `0.595` / `0.610`. GT still highest on mem.
- φ(F2) > φ(GT) is observed and allowed (StyleImage refs ≠ Target query space); ranking among generators still valid.

## Fairness caveats (unchanged)
- 1-shot (P1/E1/F0) vs 8-ref (F2/F3) are different conditioning; use Mode A/B.
- ckpt steps: E1/F0@100k, F2@75k, F3@80k.
- E1 by_bucket previously mislabeled 80 `latin_ext` as `latin_lower` → **fixed**; overall means unchanged.

## Files
- `scripts/score_f03_e12_v51.py`, `scripts/rebuild_glyph_board_fair_axes.py`
- `reports/f03_test16_strat/{metrics_summary,e12_v51_scores,index}.html`
