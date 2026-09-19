# K6 startup preflight — engineering hold

K6-0: 192 donor audit outputs completed (operational donor proxies, not verified style contamination). Calibration: 4096 train samples; sampler equivalence100episodes; 8-rank relational gradient parity PASS.
K6-A: initialized independently from K0/G0b; 2 successful preflight updates + full-state resume to4, 8 ranks synchronized,0skips. Across32rank/update records,52GT-distance/time eligible samples, every new ranking loss is0. Thus the intended new objective is inactive in this preflight. K6-A/B formal10000-update training has NOT started. B preflight not started.

Control STOP was written by the agent as an engineering preflight hold, not by a user cancellation. Queue exception text 'K6 user STOP' is generic and not a claim about user action. Current status explicitly AWAITING_OBJECTIVE_FIX_REVIEW.

OBJECTIVE_DIAGNOSIS.json shows one eligible FZBangSKLTJW digit0 case: GT-noised estimates satisfy ranking at t50/250/500/750, while pure-noise DDIM8 gives gated mean rank loss0.158276 and nonzero gradient0.643636 on one inspected UNet parameter. This is mechanism feasibility, not final quality or statistical proof.

Proposed scientific correction, pending explicit user review: experiments/K6/OBJECTIVE_FIX_REVIEW.md. Preserve all preflight states/logs, do not remove STOP or run the archived queue unchanged. New implementation must gate acceptance on actual nonzero added-objective gradients, not only eligible sample counts. Review request has been sent. Existing hourly follow-up retains this hold while continuing independent K5 work.
