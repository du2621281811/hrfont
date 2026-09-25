# Human questionnaire metrics and automatic-metric correlation (FD45k)

This package reports the complete 28-participant, 30-question human ranking and its association with per-sample automatic scores. The FontDiffuser candidate is mapped only to FontDiffuser 45K (`fontdiffuser_45k`); FontDiffuser P2 is excluded from this questionnaire analysis.

## Human questionnaire summary

| Candidate | Mean rank (lower is better) | Top-2 rate |
|---|---:|---:|
| Ground truth | 1.823 | 80.5% |
| HR-Font | 2.111 | 77.5% |
| FCA-GAN | 4.757 | 2.4% |
| FTransGAN | 4.782 | 3.5% |
| GAR-Font | 3.927 | 13.7% |
| FontDiffuser (FD45k) | 3.600 | 22.5% |

Valid raters: **28**; formal samples: **30** (all extra Latin); complete ballots: **840**; Kendall's W: **0.636**. No annotator was excluded by the attention-check rule. Twelve repeated formal rank rows were resolved by keeping the final response for the affected annotator/sample, consistent with the official analyzer.

## Human–automatic correlation

We correlated per-sample, per-method human mean ranks with the corresponding automatic metric values using Kendall tau-b. The point estimate uses 30 samples × 5 generated candidates = 150 paired observations. Scores were not averaged across samples or methods before calculating tau. Confidence intervals resample the 30 samples (1,000 replicates, seed 3407). Lower-is-better metrics are direction-reversed before computing tau.

| Metric | Kendall τ-b | 95% bootstrap CI |
|---|---:|---:|
| l1 | -0.023 | [-0.101, 0.050] |
| ssim | -0.016 | [-0.092, 0.057] |
| lpips | 0.246 | [0.159, 0.356] |
| dino_distance | 0.281 | [0.199, 0.370] |
| rsc8 | 0.281 | [0.184, 0.384] |

## Coverage and files

All 30 FontDiffuser questionnaire samples were covered by FD45k predictions: 29 regular-release PNGs and one verified Git-supplement PNG. The 23 samples already in the V4 FD45k metric table reuse those scores; the seven questionnaire samples outside the V4 panel were newly scored against the shared K7-B targets with the same L1, SSIM, LPIPS, DINOv2, and RSC-8 pipeline. Every candidate/sample metric join is complete (150 rows, no duplicate keys).

- `questionnaire_summary.json`: aggregate-only participant, response-quality, rank, top-2, pairwise, agreement, and bootstrap metrics.
- `questionnaire_per_sample.csv`: per-item mean ranks and human pairwise-preference rates; no annotator identifiers.
- `automatic_scores_per_sample.csv`: the five generated candidates’ per-sample automatic scores.
- `correlation_metrics.csv` and `.json`: estimates, directions, protocol, and confidence intervals.
- `provenance.json`: source commits, input hashes, metric definitions, coverage, and privacy boundary.

These are exploratory associations on this fixed 30-item Latin-character questionnaire. They do not establish causality or broad generalization; no multiple-testing correction was applied.
