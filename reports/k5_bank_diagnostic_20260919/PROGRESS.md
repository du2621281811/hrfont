# K5 progress snapshot

Snapshot UTC: 2026-09-19T16:02:55.383777+00:00

- K5-A and K5-B: 10,000 successful updates each, zero skipped updates; complete-state checkpoints and final EMA accepted.
- Standard evaluation: 59,980 images per model, 119,960 total. Paired metrics, shot/script/GT-derived difficulty strata and browser-tested local review board delivered. See ../k5_metrics_20260919/REPORT.md.
- Bank diagnostic: A 288/288; B 268/288. Generation still running at snapshot; this is not final diagnostic acceptance.
- Local board: outputs/K5_EVALUATION_20260919/index.html (workspace root; large image assets are not committed).

## Controlled bank experiment

Frozen independent A/B final weights, matched V2 training bank and weight-only exclusion. Six conditions: Top10 original alpha, Top10 uniform, Top50 original alpha, all legal candidates original alpha, all uniform, Delta-off. 24 original cases, two fixed seeds, four references. Ordinary control cases repeat across hollow-font comparisons; these are not 24 independent font identities.

All six conditions use fresh full-FP32 generation. Standard evaluation used AMP, so its images are not reused as the diagnostic baseline. Full-bank routing uses chunked global softmax without candidate truncation. FP32 synthetic chunk equivalence and six real generation parity checks passed; see JSON evidence. Initial AMP parity was insufficient and was retained as historical evidence. Frozen configuration mutation was repaired with dataclasses.replace; the initial failed log is retained remotely.

## Outstanding work

Complete and validate all 576 results; deliver six-condition visual board and paired summaries, assess hollow-style recovery and ordinary-font degradation. Complete router/representation audits before attribution or retrieval changes. Current metrics alone establish neither a winning model nor correct donor retrieval. Do not restart K4 or retrain K5. No retrieval/scientific recipe change is authorized without user review.

## Provenance

Remote source: /root/projects/hrfont_k5_20260919_r2
Remote control: /root/data1/hrfont_k5_20260919/control
Remote output: /root/data1/hrfont_k5_20260919/bank_diagnostic
Runner log: bank_diagnostic_r2.log
Execution scripts archived under experiments/K5/diagnostics; their absolute paths intentionally document the executed environment. DECISIONS files preserve prior repairs and storage decisions. RESULTS_SNAPSHOT is a partial snapshot, not a final aggregate.

## Completion update 2026-09-19 16:11 UTC follow-up

Generation now complete: A 288/288, B 288/288. All 576 local prediction hashes match recorded hashes. SUMMARY.csv contains per-model/font/mode means and differences from matched Top10. Six-condition local board outputs/K5_BANK_DIAGNOSTIC_20260919/index.html includes GT and fixed references; browser QA passed with zero broken images and no JavaScript errors. Screenshot spot-check completed. RESULTS_SNAPSHOT.json above remains the preserved partial snapshot. Full router/representation audits and final causal interpretation are still pending; do not equate completed PNGs with complete diagnosis.
