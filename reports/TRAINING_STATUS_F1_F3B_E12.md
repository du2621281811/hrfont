# Training status — 2026-09-09 23:05 CST

## Live

| Job | GPU | State | Progress | Notes |
|---|---|---|---|---|
| **F1** | 2 | running | ~36.8k/80k | solo after old F3b stop |
| **F3b (old)** | — | **STOPPED** | `stopped_step` **10883** | stroke-bank `f3b_joint_crossbank_s3407`; not resumed |
| **F3b (new)** | 0 (queued) | **waiting_f1** | 0/80k | `f3b_topology_ownfont_s3407` + topology bank SHA `6cdefe70` |
| **E12 φ / membership** | — | DONE | — | |
| **Watchdog** | — | pid 47943 | — | launches new F3b only after F1 DONE |

## F1 speed: concurrent F3b vs solo

Tqdm `s/it` from `logs/f1/F1-OFFRSI-A-S3407.log` (median of 100-step blocks):

| Window | F1 steps | median s/it |
|---|---:|---:|
| Mid, F3b still on GPU0 | 30k–34k | **3.07** |
| Late concurrent | 34k–35.3k | **2.92** |
| Just after F3b STOP (~14:31Z) | 35.5k–36.2k | **1.70** |
| Solo now | 36.2k–36.8k | **1.02** |

Solo is about **3×** the concurrent F3b period (host I/O / cache contention, not a batch/lr change). ETA from tqdm ~15–17 h to 80k at ~1.3 s/it.

## New F3b protocol (this queue)
- own-font Ec
- support chars = topology top-16; train \(k_s\sim U\{4..16\}\); infer top-8
- inject = SupportAdapter → up-path style cross-attention (not RSI)
- adapter standard init (not zero-init)
- style tokens = same 9 global as F1/F2 (no local64 in this run)

Canvas / 「太规整」结论：`reports/STYLE_REGULARITY_AND_CANVAS_20260909.md`  
生成器改动 + E12 同分不变量：`reports/GENERATOR_STYLE_PLAN_20260909.md`（明天做 A1；今天只保 F1）  
同字空间统计：`reports/samecontent_spatial_variation/`

