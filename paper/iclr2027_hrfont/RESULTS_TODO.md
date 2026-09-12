# Result placeholder ledger

Execution order and Go/No-Go dependencies are specified in [`EXPERIMENT_EXECUTION_PLAN.md`](EXPERIMENT_EXECUTION_PLAN.md). A table cell may be filled only after its matching release condition below and the frozen protocol are both satisfied.

| Paper location | Required artifact | Release condition |
|---|---|---|
| Abstract | headline primary result and identity margin | formal main comparison complete |
| Table 2 | main baseline comparison | frozen test opened once; clustered CIs computed |
| Table 3 | D/R ablations | each retained claim has its direct matched control |
| Figure 4 | qualitative grid | preregistered selection rule; same noise/sampler/reference |
| Appendix | n-shot robustness | fixed checkpoint, n in 1/2/4/8 |
| Appendix | leakage probes | donor/font/style probes and wrong-condition interventions |
| Appendix | retrieval diagnostics | top-K overlap, alpha entropy, and effective donor count |
| Appendix | efficiency | parameters, FLOPs, peak memory, p50/p95 step time, GPU hours |
| Ethics | asset licenses and human study | license audit and study protocol complete |

## Claim ledger

| Proposed sentence | Minimum evidence | If negative |
|---|---|---|
| Canonicalized geometry Delta reduces leakage relative to legacy feature Delta | D2 vs D1 plus leakage probe | retain D1 as the historical baseline and remove the package-level gain claim |
| Candidate-axis preservation is useful | D3 vs D2 | simplify to geometry mean-Delta |
| Anchor-relative residual is necessary | D3 vs D4 | call method retrieved same-character set, not residual prior |
| Delta uses the requested character | correct vs D6 wrong-character | remove target-aware claim |
| Local reference evidence improves brush/style | R2 vs R1 on ref-observable attributes | keep per-ref global only; run layer probe |
| Graphics keys provide cross-script correspondence | R3 vs R2, R4, equal-budget FSFont-style baseline | remove graphics keys and retain learned local attention |
| Dual path is beneficial | D7 warp-only/value-only/dual | retain the simplest passing path |
