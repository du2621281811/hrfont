# E12 Cross-Script Style Evaluator — paper tables (teacher protocol)

Primary metric: **φ cosine** (see `cosine_protocol.md`).  
Contribution framing: **Cross-script / Style-aware / GT-free** — not a font classifier.  
Membership verifier: appendix only (`controls_summary.json`, `scores_test16_summary.json`).

Encoder: `e12_phi_s2_b_s3407` · cache `cache_v0913_b` · seed 3407 · ref8 永和书风骨韵天地.

---

## Table 1 — Evaluator self-validation (required)

| Experiment | Result | Source |
|------------|--------|--------|
| Cross-script font retrieval **R@1** | **0.364** | `retrieval_summary.json` |
| Cross-script font retrieval **R@5** | **0.784** | same · n_queries=1768 · gallery=34 test families |
| MRR | 0.547 | same |
| Same-font vs different-font **AUC** (cosine) | **0.947** | `auc_cosine_summary.json` |
| Order acc (same > cross) | 0.939 | same · gap cos 0.425 |
| Human Spearman (mean rating vs cosine) | **pending** | fill `human_ratings_template.csv` → `e12_spearman_human.py` → `human_spearman_summary.json` |

---

## Table 2 — Application Style Score on generator test16

Metric: \(S_{01}=(\cos+1)/2\) · latin+digit · refs = font StyleImage ref8.

### Full dirty coverage (n=512)

| Method | Style Score \(S_{01}\) ↑ | Cosine |
|--------|-------------------------:|-------:|
| GT | 0.892 | 0.785 |
| F2 @80k | 0.897 | 0.794 |
| F2 @40k | 0.896 | 0.792 |
| F0 @100k | 0.869 | 0.737 |
| GT + wrong-family refs | 0.723 | 0.446 |

### Matched overlap with clean preds (n=144)

| Method | Style Score \(S_{01}\) ↑ |
|--------|-------------------------:|
| F2-CLEAN @80k | **0.904** |
| F2-CLEAN @40k | 0.902 |
| GT | 0.894 |
| F2 (dirty) @80k | 0.894 |
| F0-CLEAN @100k | 0.889 |
| F0 (dirty) @100k | 0.865 |
| GT + wrong-family | 0.720 |

Source: `scores_test16_cosine_summary.json`, `scores_test16_cosine_matched.json`.

---

## Table 3 — Main generation results (pixel + Ours)

Pixel diagnostics from `reports/f03_test16_strat/metrics_summary.json` (overall).  
**FID**: not in this pack yet — leave column blank or compute later.  
**Ours** = Style Score \(S_{01}\) (cosine), GT-free for cross-script style.

| Method | SSIM ↑ | LPIPS ↓ | FID ↓ | Ours Style ↑ |
|--------|-------:|--------:|------:|-------------:|
| F0 @100k | 0.644 | 0.157 | — | 0.869 |
| F2 @80k | 0.653 | 0.145 | — | 0.897 |
| F0-CLEAN @100k | *(recompute metrics when board metrics include F0C)* | | — | 0.889 (matched) |
| F2-CLEAN @80k | *(same)* | | — | 0.904 (matched) |

Caveat (paper): L1/SSIM/LPIPS need target-script GT; Ours does not.

---

## Scripts

```bash
PY=/root/miniforge3/envs/boogu/bin/python
$PY scripts/e12_eval_retrieval_cosine.py --device cuda:0
$PY scripts/e12_validate_auc_cosine.py --device cuda:0
$PY scripts/score_preds_e12_cosine.py --device cuda:0
# after human ratings:
$PY scripts/e12_spearman_human.py --ratings reports/e12_paper/human_ratings_template.csv
```

Human: `HUMAN_PROTOCOL.md` · board `human_board/index.html` (serve from repo root on :8793).
