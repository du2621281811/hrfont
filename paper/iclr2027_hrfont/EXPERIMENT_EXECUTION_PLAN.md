# Current experiment plan

The authoritative plan is [G_NEXT_8V100_PLAN_20260914.md](../../reports/G_NEXT_8V100_PLAN_20260914.md).
It supersedes the former Set-Delta / Graphics-Ref D/R matrix in this file.

1. Let the current multi-shot evaluation finish.
2. Automatically run G2 and G2-RL matched continuation controls, each +5k on 8×V100.
3. Align sampler, donor mask, sample validity, and reference episodes; compare TC and controls at matching added steps on clean validation data.
4. Select the better base by generated quality, then review the next 5k continuation and TC-RefOnly / TC-Capacity studies.
5. Complete the same-bank absolute-feature control and review the budget for a Delta×TC training study.
6. Finish external comparisons, learned-evaluator calibration, and human evaluation.

The exact run IDs, launch recipe, deployed queue status, disk guard, selection rules, and underperformance branches are specified only in the authoritative plan to avoid conflicting copies.
