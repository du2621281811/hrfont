# Results to complete

Current method: task/evaluation + Mean-Delta + TC-v2. See the [execution plan](EXPERIMENT_EXECUTION_PLAN.md).

| Paper item | Required evidence |
|---|---|
| Abstract / conclusion empirical summary | Matched generation outcomes, identity and style/family results |
| Main comparison | Strong internal baseline and audited external baselines; identical valid sample keys; resource cards |
| TC increment | G2-CONT vs TC-G2, G2RL-CONT vs TC-G2RL at +2500 and +5000, matched episode and sampler |
| Target conditioning | Constant-query model trained under the same descriptor and joint recipe |
| Supervised completion | Same-capacity random-H / zero-W model with no descriptor objective; distinguish this recipe comparison from a joint-loss-only ablation |
| Mean-Delta interface | Same-bank absolute control under the same parent, interface, data, and budget |
| Complementarity | Matched Delta×TC training study if approved; inference-only removal is separate |
| Reference budgets | 1/2/4/8 on fixed checkpoints, nested episodes, separate old/new Ref8 protocols |
| Learned evaluator | Final architecture/data/objective; calibration split; scores and human agreement; lineage isolation |
| Qualitative figure | Actual matched outputs; include ordinary, decorative, and difficult cases |
| Reproducibility | Parent/model/cache/manifest hashes, draw-stream audit, measured GPU hours and inference latency |
| Ethics / submission | Font permissions, evaluation consent/compensation, current venue requirements |

Old F-series E12 or unfiltered board numbers are not substitutes for current missing results. Existing observations motivate the work; final tables use their stated protocol.
The test set has already appeared in exploratory boards. Model selection now uses validation data; do not claim that test data were never inspected.
