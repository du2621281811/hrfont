# Agent decision log — F1 / F3b / E12 continuity

Date: 2026-09-08 (UTC) / 2026-09-09 (UTC+8)
Owner: agent (for PI review)

## D1 — Persist training across Cursor/SSH disconnects
**Decision:** Launch `scripts/watchdog_f1_f3b_e12.py` under `setsid` + `nohup`, ignore SIGHUP.
**Why:** User asked to avoid stops caused by network/session loss. Train processes were already nohup’d, but nothing auto-resumed on crash/OOM/reboot-of-process.
**Policy locked in watchdog:** no batch/lr/steps/seed degradation; never writes STOP; adopts live PIDs; resumes F1/F3b from higher of `stopped_step`/`last_state`.
**Artifacts:** `reports/watchdog_f1_f3b_e12/{status.json,incidents.jsonl,watchdog.pid,watchdog.stdout}`

## D2 — One planned F3b restart to apply support Ec pool cache
**Decision:** STOP F3b once, resume from `stopped_step` with code change; leave F1 and E12 membership untouched.
**Why:** F3b was ~10–18 s/it with GPU util ≈0%. Root cause: each support glyph re-read multi-scale Ec rows from a **94 GiB memmap** (random page faults). Historical F3 finished at ~1 s/it with fixed-8 support.
**Change (science-equivalent):** cache pooled Ec vectors `(font, support_cp) → float32[D]` on host (~60 MiB for 228×93), prewarm at F3b startup via `features_many`. Adapter inputs unchanged (same means of same features).
**Risk accepted:** one intentional checkpointed interrupt of F3b only; RNG restored from `trainer_state.pt`.

## D3 — Do not reboot host to clear orphan VRAM (yet)
**Decision:** Continue with orphan contexts on GPU1/2/3 (~4–9 GiB each). GPU0 was free enough after F3b STOP.
**Why:** `nvidia-smi --gpu-reset` blocked without killing all CUDA clients; reboot would interrupt F1 + membership. Remaining free VRAM still fits current jobs.
**Revisit if:** OOM, or F1/F3b cannot allocate after a crash.

## D4 — E12 φ finished; membership may restart from scratch if it crashes
**Decision:** φ early-stopped @2250, `best_auc=0.9009`. Membership has **no checkpoint resume**; watchdog relaunches from step 0 if the process dies before `test_metrics.json`.
**Why:** Existing `train_membership.py` has no resume API. Full membership is short (~18.7k steps) vs F1/F3b 80k.

## D5 — Keep F1 concurrent with F3b after fix
**Decision:** After D2, run F1 (GPU2) + F3b (GPU0) + membership (GPU1) together.
**Why:** User asked to execute all three without unnecessary serialization. If F3b remains ≫3 s/it after prewarm, agent may propose serializing F3b (record as new decision) rather than silently changing schedule.

## D6 — Heartbeat “loading_library” while step advances is not a hang
**Observation:** Heartbeat updates only every `log_interval=100`. After resume @145, next heartbeat is at step 200. Do not treat stale heartbeat alone as dead if log tqdm advances.

## D7 — Accept F3b ~4–8 s/it after pool cache (no further interrupt)
**Decision:** Do not STOP again for more optimization in this pass.
**Why:** Prewarm wrote 21204 pooled entries; step time fell from ~15 s/it to ~4–8 s/it. Still slower than F1 (~1.5 s/it) due to support K≤16 + adapter + shared host I/O with F1. Further interrupts would violate the continuity goal more than they help.
**Revisit if:** F3b regresses to ≫10 s/it, or F1 finishes and freeing GPU2/CPU would let F3b approach ~2 s/it (optional serialize then — new decision).

## D8 — Watchdog torch.load cost tolerated
**Decision:** Watchdog may `torch.load` trainer_state occasionally to pick resume dir; keep as-is.
**Why:** Correctness of resume path > micro-optimizing the watchdog. Heartbeat is preferred for step display when present.

## Status board
Live: `reports/TRAINING_STATUS_F1_F3B_E12.md`
PI locks: `reports/PI_EXEC_DECISIONS_20260909.md`
Watchdog control: `bash scripts/start_watchdog_f1_f3b_e12.sh`
