# K5 decisions

## 2026-09-19 — execution order and recovery contract
- User authorized K5-A then K5-B, each original K1 recipe on V2 with independent K0 initialization and10000 successful updates.
- Defer bank/donor interventions until both finish; preserve Top10/alpha during training to isolate representation/dual-branch changes.
- K4 remains stopped. Use independent K5 control and full-state recovery.
- Live historical baseline: K1 9546.56s/10000 updates. K5 throughput not measured; A3-5h and B4-7h are provisional training-only estimates.
- Storage observation: data1~20GiB and data2~23GiB available. Must size Es spatial cache before building; exact online encoding/bounded cache is an option, not yet a selected implementation.
- User explicitly requests proactive fault repair and a final decision summary. Append evidence/action/validation/resume records for each actual intervention; do not fabricate completed implementation or recovery.

## Implementation and real-input preflight
- Dedicated source /root/projects/hrfont_k5_20260919; stopped K4 source and STOP files unchanged.
- Es actual intermediate channels are64@48x48 and128@24x24. Use exact FP32 frozen first-two-block extraction, same native96 input; no3x3 upsampling. Bias-free identity-initialized1x1 trainable adapters.
- B routes Ec and Es separately and combines offsets with zero-initialized scalar tanh gate; Es branch is nonzero initialized.
- Chose bounded256-key CPU cache plus exact online extraction instead of a large persistent Es cache; both arms share preprocessing/selection.
- Checkpoints allocated on data2 with8GiB floor; source uses~10MiB root disk. Kept full state/RNG/scaler and original K1 schedule, manual rank-average gradients and existing FP32 nonfinite replay.
- A first two8-rank preflight updates finite, adapter gradient nonzero, rank spread0, noAMP skips, no donor weight-family violations. Queue performs full resume and B preflight before formal runs. These preflight steps do not count toward formal10000 updates.
