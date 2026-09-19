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

## Preflight recovery R2 — FP16 gated branch gradient
- R1 K5-B ran4 finite successful updates with synchronized ranks and nonzero gate gradients, but every recorded Es-adapter gradient was zero. The queue correctly withheld formal training (NEEDS_RECOVERY).
- Hypothesis: near-zero gate plus FP16 branch computation underflows the adapter gradient. Chosen repair: compute Es routing/offset and scalar-gate combination inFP32, cast the final combined offset to the existing branch dtype. The equations, gate initialization, loss, schedule and candidate selection are unchanged.
- R1 source/checkpoints/logs preserved. R2 source is /root/projects/hrfont_k5_20260919_r2. Rerun both arms and full-state resume rather than overwriting the failed preflight certificate. Acceptance requires nonzero Es adapter gradients after startup and nonzero B gate gradient.
- Also handled zero-active Delta-off dual payload safely for later diagnostics; active dual inputs still require both branches.
- Frozen Es prefix versus full encoder intermediate output checked on a real native96 train glyph inFP32: max absolute errors[0,0], shapes64x48x48 and128x24x24. Record control/FEATURE_EQUIVALENCE.json.
- R2 B validation at update2: Es adapter gradient2.877898808e-9 (nonzero), gate gradient0.07142036; rank spread0, noAMP skips. This supports the underflow diagnosis. First-step adapter gradient0 is expected from exactly zero gate. Full-state resume validation still precedes formal launch.
- R2 A and B each completed8-rank4-update preflight including a2-update full-state resume. B update4 adapter gradient1.867284993e-8, gate gradient0.0794740, noAMP skips, rank spread0. Queue wrote PREFLIGHT_PASSED.json for the exact source identity and launched formal K5-A-V2-K1RECIPE-S3407; B follows automatically after A10000 successful updates.

## Training completion and evaluation launch
{
  "time": 1789776019.4480135,
  "action": "training_complete_and_evaluation_launch",
  "evidence": {
    "A": {
      "updates": 10000,
      "skips": 0,
      "seconds": 10037.4
    },
    "B": {
      "updates": 10000,
      "skips": 0,
      "seconds": 13566.93
    }
  },
  "reason": "Both formal runs completed; proceed to authorized fixed protocol evaluation",
  "alternatives": "Do not load K5 weights into K4 model; use K5 model and DataContext with original sampler/scoring",
  "change": "Independent evaluation entrypoint; no live training source changes",
  "runner_sha256": "92d2a1e11b0afb3f5362caa7476f1daf86a785a666fc426f270e08ca5bafbe53",
  "recovery": "Initial system torchrun lacked accelerate before model loading; preserved evaluation.log, relaunched with wam-flow torchrun and exclusive lock",
  "protocol": "V2 bank, original train queries, full frozen v2 val/test, 1/2/4/8shot; no output reuse",
  "result": "Evaluation launched; pending actual output validation"
}

{
  "time": 1789776189.6862087,
  "action": "evaluation_environment_correction",
  "evidence": "System environment lacks accelerate; wam-flow lacks info_nce; repo scripts explicitly specify /root/miniforge3/envs/boogu/bin/python",
  "change": "boogu torchrun, evaluation_r3.log; no model or recipe changes",
  "validation": "awaiting first generated rows",
  "previous_logs": [
    "evaluation.log",
    "evaluation_r2.log"
  ]
}

{
  "time": 1789795281.1633968,
  "action": "prepare_visual_review_donor_provenance",
  "reason": "Completed A/B statistics; board requires exact selected donor glyphs",
  "change": "CPU-only control/trace.py uses frozen original retrieval with V2 train bank and weight-only exclusion; A/B protocol jobs equality asserted",
  "result": "PID4136485 running; control/trace.log and evaluation/donor_traces/PROGRESS.json",
  "protocol_changed": false
}

{
  "time": 1789798906.854432,
  "action": "replace_stalled_small_file_sync_with_archive",
  "evidence": "rsync elapsed 2h with local size unchanged at261MB; donor traces59980 complete",
  "change": "Stopped only own rsync75173/ssh75174; control/pack.py builds deduplicated prediction/GT/ref/donor tar; no remote originals removed",
  "validation": "Archive pending; will verify file count and prediction hashes after extraction",
  "protocol_changed": false
}

{
  "time": 1789806287.1243136,
  "action": "resolve_visual_asset_symlinks",
  "evidence": "All119960 local prediction hashes pass; browser broken GT/ref/donor images because original tar preserved remote absolute symlinks",
  "repair": "Repack assets with tarfile dereference=True; preserve originals; no model/output changes",
  "validation": "Pending resolved archive transfer and browser rerun"
}

{
  "time": 1789817054.035768,
  "action": "chunk_equivalence_gate_failed",
  "evidence": "AMP synthetic test:8/41472 elements above atol=.002 rtol=.002, max_abs=.00408935546875; no PASS marker written",
  "hypothesis": "FP16 chunk-dependent convolution/reduction rounding versus implementation error; not yet distinguished",
  "alternatives": "Do not relax tolerance or start six-arm study",
  "change": "Separate dense/chunk FP32 control with stricter2e-5 tolerance; unchanged trained source and weights",
  "validation": "check_chunk_fp32.log and CHUNK_FP32_EQUIVALENCE.json pending"
}

{
  "time": 1789817092.2790499,
  "action": "strict_fp32_chunk_control_passed",
  "tests": 36,
  "max_abs": 3.814697265625e-06,
  "interpretation": "FP32 dense/chunk agreement supports numerical precision sensitivity rather than gross routing implementation mismatch; real-input generation parity still required",
  "next": "Use common FP32 routing arithmetic across diagnostic arms and validate real-input dense/chunk generation before launch; preserve original AMP baseline separately"
}

{
  "time": 1789822910.0628655,
  "action": "real_generation_parity_diagnosis",
  "evidence": "Common FP32 offset with AMP denoiser yielded max3/255 and mean0.05595/255 dense/chunk difference, failing predeclared max1 and mean0.01 thresholds",
  "decision": "No bank experiments launched and no tolerance relaxed; test full FP32 denoiser plus 3-donor chunks to isolate iterative amplification",
  "files": [
    "real_parity.log",
    "real_parity_fp32.py",
    "real_parity_fp32.log"
  ],
  "recipe_changed": false
}

{
  "time": 1789830476.8466594,
  "action": "launch_six_arm_bank_diagnosis",
  "evidence": "36 FP32 module tests and six real generation cases passed; maximum1/255 pixel difference and mean<=0.000073/255",
  "decision": "All six arms fresh full-FP32 denoising; do not reuse AMP inference as baseline",
  "scope": "A/B independent frozen10k,24 original cases x2seeds x6modes each,full eligible donor list never truncated for memory",
  "entrypoint": "/root/data1/hrfont_k5_20260919/control/bank_diagnostic.py",
  "pid": 33014,
  "log": "/root/data1/hrfont_k5_20260919/control/bank_diagnostic.log",
  "output": "/root/data1/hrfont_k5_20260919/bank_diagnostic",
  "lock": "BANK.lock",
  "training_recipe_changed": false
}

{
  "time": 1789830542.1272066,
  "action": "bank_config_immutable_fix",
  "evidence": "FrozenInstanceError at Top50 configuration; Top10 conditions already saved",
  "repair": "Use dataclasses.replace to construct frozen DeltaConfig; no changes to values/formula",
  "pid": 34347,
  "log": "/root/data1/hrfont_k5_20260919/control/bank_diagnostic_r2.log",
  "resume": "Existing result hashes verified; continue missing conditions"
}
