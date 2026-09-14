# Evaluator validation tables (teacher layout)

Source numbers: `teacher_tables_1to3.json`  
Protocol: unseen **test** fonts (34), zh ref8 mean-pool proto, cosine Style Score.  
Baselines: CLIP ViT-B/32 · DINOv2-base · LPIPS-Alex trunk features (same embed→cosine protocol).

**Paper body (recommended):** Table 1 + Table 2 + Table 4(human).  
**Appendix:** Table 3 (score separation).

---

## Table 1. Cross-Script Retrieval

*Cross-script font retrieval on unseen fonts. Given Chinese reference glyphs, the task is to retrieve the corresponding Latin font from the candidate set.*

| Method | R@1 ↑ | R@5 ↑ | MRR ↑ |
|--------|------:|------:|------:|
| CLIP | 0.093 | 0.252 | 0.198 |
| DINOv2 | 0.265 | 0.535 | 0.396 |
| LPIPS-Alex | 0.169 | 0.495 | 0.330 |
| **Ours** | **0.364** | **0.784** | **0.547** |

Gallery size = 34 (chance R@1 ≈ 0.029). n_queries = 1768.

---

## Table 2. Cross-Script Style Verification

*Cross-script style verification on unseen fonts. We evaluate whether each metric can distinguish matched Chinese–Latin font pairs from mismatched pairs.*

| Method | ROC-AUC ↑ | Pairwise Order Acc. ↑ |
|--------|----------:|----------------------:|
| CLIP | 0.543 | 0.564 |
| DINOv2 | 0.825 | 0.849 |
| LPIPS-Alex | 0.781 | 0.886 |
| **Ours** | **0.947** | **0.939** |

n_pairs = 544. Order Acc. = P(score(matched) > score(mismatched)) for the same Chinese reference.

**Put this table in the main paper.**

---

## Table 3. Style Score Separation (appendix)

*Ours Style Score \(S_{01}=(\cos+1)/2\) on GT Latin/digit queries; mean ± std.*

| Pair Type | Mean ↑ | Std | n |
|-----------|-------:|----:|--:|
| Matched font | **0.901** | 0.066 | 272 |
| Hard negative (same weight class) | 0.792 | 0.112 | 160 |
| Random negative | 0.733 | 0.112 | 272 |
| Same coarse category, different font | 0.792 | 0.112 | 160 |

Observed order: \(S_{\mathrm{matched}} > S_{\mathrm{hard}} > S_{\mathrm{random}}\).  
Hard = different test font with the same parsed weight class (light/regular/bold/italic); unknown-class stems excluded from the hard pool.

---

## Table 4. Human Agreement (pending ratings)

*Correlation with human judgments of cross-script font-style consistency.*

| Metric | Spearman ρ ↑ | Kendall τ ↑ |
|--------|-------------:|------------:|
| LPIPS | — | — |
| CLIP | — | — |
| DINOv2 | — | — |
| **Ours** | **pending** | **pending** |

Fill `human_ratings_template.csv` via `human_board/`, then:

```bash
/root/miniforge3/envs/boogu/bin/python scripts/e12_spearman_human.py \
  --ratings reports/e12_paper/human_ratings_template.csv
```

Baseline human correlations: re-score the same items with CLIP/DINOv2/LPIPS embeddings (extend after ratings land).

---

## Reproduce

```bash
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniforge3/envs/boogu/bin/python
$PY scripts/e12_teacher_tables_baselines.py --device cuda:0
```
