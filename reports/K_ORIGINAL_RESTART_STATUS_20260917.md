# K original restart status — 2026-09-17

The original K1 is running as `K1-ORIGINAL-V0917-S3407` on `sitonholy`.

Checked at 2026-09-17T03:47:12.011094+00:00; latest logged successful update 10, attempt 10; global batch 64 = 8 ranks x 8; DDP spread 0; AMP skips 0. This is startup/engineering evidence, not a generation-quality claim.

## Execution provenance

- Core training/inference commit: `43916c3ea91b00e99aa64372354bd46b81fd4452`.
- Additional frozen panels and final validation review: `1df6cec53`.
- Isolated code: `/root/projects/hrfont_k_20260917`.
- Existing historical source and runs under `/root/projects/hrfont` were not reset or overwritten.
- Old `K1-FORCED-STYLE-COMPLETION-V0916-S3407` was safely stopped and retained; it is an exploratory global512/legacy-TC run, not the original K1 protocol.
- New K1 starts fresh from the specified K0/G0b@10k.
- Checkpoints: `/root/data1/hrfont_k_20260917/K1-ORIGINAL-V0917-S3407`, symlinked into the run. Two rolling complete states plus milestone EMA are retained.

## Implementation restored

The runner restores I LocalMemory (144 target-aligned tokens, VGG enc2 readout), Dynamic SetOffset, beta-ramped RMS-scaled TC with fixed global coefficient 1, raw-x0 change-focused output loss, hard per-update quotas, actual joint CFG/source residual gating, full-state checkpoints and EMA inference. No historical I/G/H implementation was edited. Formal K1 does not add Chinese auxiliary training or west-to-Chinese training. K2 is not pre-scheduled, following the original specification.

## Verification

- Single-process 20-update global64 smoke passed.
- 1000 quota episodes and real-model condition/mask/gradient assertions passed.
- Eight-GPU 120-update reference run passed, with step100 saved; zero AMP skips, rank parameter spread 0, nonzero Reader/online encoder/router gradients.
- Resume from step100 to120 passed full-state continuation checks. First two resumed updates match the reference loss and gradient values exactly. Later FP16/DDP trajectory is not bit-identical: step120 loss absolute difference 2.0355e-5; max parameter difference 5.5801e-5; relative model L2 difference 1.2035e-4 including spectral-normalization buffers. This numerical boundary is explicitly retained in the preflight evidence.
- Stable preflight update median 0.965 s, rank0 allocated GPU peak 8663 MiB. Rough pure-update 10k duration is 2.7 h plus checkpoint and inference time. Old global512 step times are not a matched throughput comparison.
- Three-script K0/K1 EMA DPM20 inference and full matched HTML review generation smoke passed.

## Completed and scheduled outputs

- K0 frozen VAL192: 192/192 complete.
- K0 frozen TEST: 2816/2816 complete, 704 clean pairs x 1/2/4/8-shot.
- K1 training: running; at2k evaluate frozen VAL192; at10k evaluate full TEST with EMA.
- Final supplementary full VAL192 EMA review follows primary completion to retain outline-font qualitative cases. Validation and primary test metrics remain separate.
- Frozen manifests: [VAL192](../experiments/K/K_VAL192.json), [TEST](../experiments/K/K_TEST.json), [qualitative cases](../experiments/K/K_QUALITATIVE_PANEL.json). Cases were selected from GT before formal training, not from K1 outcomes.

Control/status paths on the execution host:

```text
reports/k_original_queue_20260917/status.json
reports/k_original_queue_20260917/final_val192.status.json
runs/K1-ORIGINAL-V0917-S3407/heartbeat.json
reports/k1_original_step2000_review/index.html
reports/k1_final_review/index.html
reports/k1_final_val192_review/index.html
reports/k_preflight_20260917/PREFLIGHT_REVIEWED.json
```

Core queue uses `scripts/queue_k_20260917.py`; supplementary post-review uses `scripts/k_finalize_review_20260917.py`. Both are detached processes. A failed child stage stops the queue and records FAILED; it does not silently report completion or alter hyperparameters. To resume an interrupted core queue, inspect the failure first, then rerun the core queue: it reuses completed baselines and the last complete training state with unchanged source/configuration.

## West-to-Chinese follow-up assessment

A reverse auxiliary task may improve reading western reference styles and transferring them to complex Chinese glyphs, but adds no western-output GT diversity. If western reference glyphs do not expose a Chinese-specific design choice, reverse supervision cannot make that choice identifiable.

A read-only exact-pixel audit found 223 training fonts with 89 clean western glyphs each, 217 distinct complete western sets and six duplicate pairs. Those six pairs also share the fixed Chinese ref8 pixels. This rules out neither near-duplicate western styles nor weak local-style diversity. The existing reviewed manifest labels only eight training western fonts as confirmed complex; other fonts are unconfirmed, not automatically simple.

Next experiment proposal, not launched: keep the same Chinese-to-western episode count; compare Chinese-to-Chinese auxiliary against an equal auxiliary budget partly replaced by western-to-Chinese. Judge the original western task using matched GT-consistent generation, reference swaps and local visual detail. First complete the unmodified original K1. Western references require a correctly partitioned western Es/cache path; do not treat the existing Chinese StyleImage cache as interchangeable.
