# sitonholy Method-aligned source overlay

This overlay preserves the current `k5_runtime.py` read from `/root/projects/hrfont_k7_v3_20260922` on `sitonholy` on 2026-09-26. It supplements, but does not overwrite, the 438-file source snapshot at `../../source/`.

To reconstruct the current 438-file identity set, use the files in `../../source/` and replace only:

`experiments/K6/implementation/r3/scripts/k5_runtime.py`

with the mirrored file in this overlay. `SOURCE_OVERLAY.json` records its current hash, the base snapshot hash, and the frozen run-config hash. The live-file delta changes process-unique temporary names for sharded cache writes and supports evaluation references by font/split; the EC/ES routing and `DualOffset` code are unchanged.

This overlay follows the current paper Method for code navigation. It does not recover the historical bytes named by the frozen 32k run config and must not be cited as an exact training-source reconstruction.
