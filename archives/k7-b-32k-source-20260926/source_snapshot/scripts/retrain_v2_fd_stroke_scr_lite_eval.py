#!/usr/bin/env python3
"""Lite Demo-8 eval for FD stroke-SCR pilot via existing probe_ckpt."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path("/root/projects/hrfont")
REP = ROOT / "reports/retrain_v2"
sys.path.insert(0, str(ROOT / "scripts"))
from retrain_v2_fd_demo8_probe import probe_ckpt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    ckpt = Path(args.ckpt_dir)
    out = REP / "fd_stroke_scr_preview" / args.label
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    m = probe_ckpt(ckpt, args.device, run_dir=None, step_override=None)
    # normalize to lite schema
    summary = {
        "label": args.label,
        "ckpt": str(ckpt),
        "updated": m.get("updated") or m.get("ts"),
        "n": m.get("n") or ((m.get("p1") or {}).get("n", 0) + (m.get("p2") or {}).get("n", 0)),
        "splits": {
            "p1": {
                "n": (m.get("p1") or {}).get("n"),
                "L1_mean": (m.get("p1") or {}).get("L1") or (m.get("p1") or {}).get("L1_mean"),
                "SSIM_mean": (m.get("p1") or {}).get("SSIM") or (m.get("p1") or {}).get("SSIM_mean"),
            },
            "p2": {
                "n": (m.get("p2") or {}).get("n"),
                "L1_mean": (m.get("p2") or {}).get("L1") or (m.get("p2") or {}).get("L1_mean"),
                "SSIM_mean": (m.get("p2") or {}).get("SSIM") or (m.get("p2") or {}).get("SSIM_mean"),
            },
        },
        "raw": m,
    }
    # optional sheet from probe imgs if present
    (out / "lite_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("label", "n", "splits")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
