# K5 execution and review specification — 2026-09-19

## Authorization and ordering

Latest user authorization: finish old matched-bank review delivery, then K5-A followed by K5-B. Both train on frozen V2 with original K1 recipe, independent K0/G0b initialization, original Top10 and alpha, and 10000 successful updates each. K5-B follows the dual-branch design below and is now authorized. K4 STOP files remain intact; no K4 restarts. Successful training means valid updates and full checkpoints, not guaranteed visual improvement.

Order: OLD_REVIEW_DELIVERY -> K5A_PREFLIGHT -> K5A_TRAIN -> K5B_PREFLIGHT -> K5B_TRAIN -> BOTH_EVALUATION_REPORTS -> POST_K5_BANK_DONOR_DIAGNOSIS -> USER_REVIEW_SELECTION_CHANGES.

Bank/donor and comparative representation diagnostics are deferred until BOTH K5-A and K5-B complete; they are no longer a pretraining gate. Preflight still validates shapes, gradients, numerics, caches, leakage rules and resume integrity. Do not start the previously scheduled K4-A six-arm donor study ahead of K5. Preserve its plan as historical context.

## Post-K5 diagnostic plan (deferred)

After both models complete, apply the six inference interventions separately to frozen K5-A and K5-B, each with its training V2 bank and weight-only exclusions. Freeze a common sample/reference/seed manifest before evaluation; retain ordinary-font controls. Do not reuse K4-A outputs as matched K5 baselines. Original Top10 alpha; Top10 uniform; Top50 alpha; all eligible alpha; all eligible uniform; Delta-off. Candidate changes are interventions, not matched-training deployment baselines. Chunk full-bank computation and compare against unchunked results before running.

Add frozen Ec/Es spatial feature audit on same-character/different-font legal images, at matched spatial scales, with centered per-character variation, normalized effect sizes, redundancy/effective rank and sensitivity to outline filling, boundary thickness, local terminal removal versus translation controls. These controlled image operations are diagnostic perturbations, not new GT. Validate perturbations visually and do not interpret holes intrinsic to glyph identity as outline style. Compare candidate coverage in the target character domain, not just Chinese reference similarity. No evaluation GT may be used for selecting production donors. Freeze audit sample manifest before computing results.

Representation distances and rank are descriptive; they do not prove recoverable geometry or successful generation. An optional equal-capacity shallow reconstruction probe can test boundary/interior recoverability with both font and character holdouts and simple pixel/edge baselines. Label this probe training explicitly; it must not update the generator or become inference conditioning. Prefer frozen audit first; use the probe only if needed to resolve ambiguous results, record exact budget/splits before training.

After both trainings and evaluations, deliver images, stratified metrics, routing/prior stats, feature audit and provenance. Decide whether to recommend bank/donor changes from matched final-image effects, not feature distances alone. Changes to formal selection remain subject to user review.

## K5-A: Es spatial Delta adapter (authorized)

Dataset and donor bank: frozen V2 train; preserve 0913 membership and all legal-GT rules; self and explicit weight-variant exclusion. Inference uses exactly the same bank, encoder hashes, exclusion and selection as training. Seed3407. Original K1 initialization: G0b/K0 step10000, NOT K1/K4 fine-tuning. Copy all original K1 hyperparameters and sampler semantics, adapting only V2 manifests and Es-Delta interface. Original K1 budget is 10000 successful updates (not the separate K4 extension to 20000).

- Global batch64; eight ranks microbatch8; equivalent accumulation if hardware changes, documented.
- Frozen content and style encoders; repaired FP32 attention logits/softmax/value accumulation retained.
- AdamW: backbone2e-5, new reader/router/Es adapter1e-4; betas .9/.999, eps1e-8; decay .01 for multidimensional parameters, zero otherwise.
- Warmup500, flat through5000, cosine to .1 peak at10000; TC .8*min(update/1000,1); joint CFG drop .02, source drop .05.
- Original loss coefficients: epsilon1, VGG .01, offset .25, completion .01, detail .05. Preserve exact original gate/time weighting implementation. No K3 pair loss or new representation loss.
- Original EMA min(.999,1-1/(update+1)); full optimizer/scaler/RNG/sampler state checkpoints, successful-update accounting and AMP-skip audit.

For target character c and eligible donor f, use frozen Es intermediate spatial maps at48x48 and24x24, not pooled1024 vector or upsampled3x3 final map. Verify actual shapes from a real forward before implementing cache contracts. Compute d_s = Es_l(x_f,c)-Es_l(x_neutral,c); adapter A_l maps to existing64/128-channel Delta interfaces using bias-free1x1 projection (add complexity only if documented necessity). Content branch remains Ec(neutral). Retain K1 SetOffset/TC and offset injection structure; replace only donor Delta features. Normalization must preserve zero difference, with any scale statistics derived from train only and frozen for inference; no independent image normalization that destroys difference magnitude.

Retain original Top10 Chinese-reference alpha for primary K5-A to isolate representation from selection. New bank selection policy is an independent subsequent factor, not silently folded into K5-A after diagnostics. Build new target-glyph and neutral Es spatial caches keyed by exact encoder/image/layer/preprocessing hashes. Existing Chinese pooled Es retrieval cache is not sufficient.

Preflight: complete donor keys, split/weight exclusion, actual train/inference bank equality; zero-difference preservation; finite adapter/router/backbone gradients across ranks after startup gates; real-input FP32/AMP forward/backward; full-state save/resume; training/eval feature equality; disk capacity estimate for new caches/checkpoints. New run and control directories with lock/PID/source identity. Never remove K4 STOP. Diagnose failures and resume complete state; do not skip data, silently change recipe, fake completion, or endlessly retry deterministic failures.

Evaluate fixed train/val/test1/2/4/8shot and confirmed-detail examples at checkpoints, compare with matched V2 K1-recipe baseline if available. K4-C stopped2191 is not a matched full-budget baseline; K4-A differs in initialization/data history. Clearly label unavailable matched controls. Retain1k/2k/4k/6k/8k/10k EMA milestones plus rolling full states, subject to storage preflight without deleting unrelated artifacts.

## K5-B: dual geometric and style residual (AUTHORIZED after K5-A)

Same K5-A dataset, initialization, budget, sampler, donor selection and losses. For each donor and character form Ec spatial difference d_c and adapted Es spatial difference d_s at the same48/24 resolutions. Keep branches separate through routing and offset prediction:

w_c = softmax(route_c(hidden,Ec(neutral),ref,t,d_c)+log alpha)
w_s = softmax(route_s(hidden,Ec(neutral),ref,t,d_s)+log alpha)
o_c = Offset_c(hidden,sum w_c*d_c)
o_s = Offset_s(hidden,sum w_s*d_s)
o = o_c + tanh(g_l)*o_s

Then use the existing single deformable-convolution residual injection with combined o. Initialize g_l=0 and the Es branch nonzero so the gate receives gradient; verify later Es branch gradient rather than requiring it at the exactly zero gate. No double-zero initialization. Gates are per layer, logged; apply original offset penalty to actual combined offsets, with branch norms monitored. This is a conservative dual structural-conditioning proposal; it does not solve appearance-only texture representation by assumption.

Controls: Ec-only V2 K1 recipe baseline, K5-A Es-only, proposed dual branch; evaluate route/branch-off interventions and matched budget/capacity limitations. Report added parameters, memory and throughput. Treat direct decoder appearance injection as a separate future design, not an undeclared part of this arm. K5-B is authorized by the latest user instruction; start after K5-A, from its own K0/G0b initialization rather than K5-A weights.

## Top10 policy proposal (separate from K5-A)

Chinese similarity is a prior, not a guarantee of Western structural proximity. Avoid using target-style Western GT as a retrieval oracle. Test all-bank and weaker-prior interventions first. If candidate coverage is the issue, construct a target-character-specific coverage subset from legal train donor glyph geometry (pixel/multiscale edge descriptors as encoder-independent audit; Ec/Es descriptors separately), retaining a small Chinese-neighbor component. Proposed fixed-budget comparison:10 nearest versus5 nearest+5 greedy coverage donors versus10 coverage donors; masks applied before selection, deterministic tie-breaking. Do not assume10 is sufficient: compare10/50/all and runtime. Separate bank coverage from prior alpha strength, and log both. Prefer soft prior or uniform diagnostic to avoid repeating unreliable cross-script similarity in routing. Any selection change for formal training must also be used in inference and separately labeled from K5-A.

## Recovery and decision reporting — user authorization 2026-09-19

Do not abandon an authorized run on a training exception. Preserve failing input/trace, rank logs and full-state identity; diagnose, make the smallest recipe-preserving repair, validate on the failing case and a real forward/backward/resume test, and continue from the last verified complete state. Use one queue lock and verify process ownership before recovery. A supervisor must exit to diagnosis for repeated deterministic faults, not endlessly restart the same command; follow-up must implement recovery rather than merely announce failure.

Keep append-only control/DECISIONS.jsonl and a readable DECISIONS.md: UTC time, stage, observed evidence, hypothesis, alternatives considered, chosen action/reason, affected files and hashes, whether protocol changed, verification results, resume checkpoint/successful step and outcome. Include runtime/AMP skips, storage choices and failed approaches. On training completion deliver this decision summary together with successful-update counts, resume history, final checkpoint identities, fixed-protocol evaluation and unresolved quality limitations. Record updates without secrets.

Routine engineering repairs and equivalent memory optimizations are authorized. Preserve batch semantics, split, donor policy, initialization and loss weights. Scientific changes to those invariants are not silent recovery actions. Do not delete unrelated data or interrupt other GPU jobs. If external storage/host access is unavailable, retain recoverable state and report exact blocker while continuing unaffected work.

ETA baseline checked live: original K1 10000 updates elapsed9546.56s (2.65h) on8 V10032GB. K5 not yet benchmarked or launched. Provisional training-only estimates: A3-5h, B4-7h, sequential total7-12h after preflight. Es feature-cache preparation, implementation, QA and inference/reporting are extra. data1 free20GiB and data2 free23GiB at check; full high-resolution Es caching may exceed this budget. Estimate cache footprint before materialization; prefer exact online frozen-encoder extraction or explicitly lossless bounded caching if needed, then remeasure throughput. Do not promise ETA independent of storage strategy. Re-estimate after200 successful updates using recent successful-update throughput and checkpoint overhead.
