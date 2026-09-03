#!/usr/bin/env python3
"""Lightweight Demo-8 probe for FontDiffuser ckpts (overfit watch).

Default: 4 Demo-8 fonts × 12 P1 + 4 P2 chars ≈ 64 gens.
Writes per-run:
  <run>/demo8_probe/step_<N>/metrics.json
  <run>/demo8_probe/curve.jsonl
Aggregates:
  reports/retrain_v2/fd_probe_curves.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from accelerate.utils import set_seed
from PIL import Image

ROOT = Path("/root/projects/hrfont")
REP = ROOT / "reports/retrain_v2"
sys.path.insert(0, str(ROOT / "scripts"))
from retrain_v2_eval_fontdiffuser import (  # noqa: E402
    build_args,
    load_pipe,
    render,
    resolve_font,
    ssim,
    to_arr,
    to_tensor,
)

PROBE_FONTS = [
    "FZHanWZKJW",
    "FZChuangHJW_DB",
    "FZDouNTJW_Te",
    "FZLingFKSJW-B",
]
PROBE_P1 = list("AaBbRrSs0123")
# filled from meta
PROBE_P2: list[str] = []


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M:%S")


def load_probe_chars() -> tuple[list[str], list[str]]:
    meta_u = json.loads((ROOT / "data/unified_v1/meta.json").read_text())
    p1 = [c for c in PROBE_P1 if c in meta_u["L_p1"]]
    p2_all = list(meta_u["L_p2"])
    # prefer common kana
    prefer = list("あいうえかきくけ")
    p2 = [c for c in prefer if c in p2_all][:4]
    if len(p2) < 4:
        p2 = p2_all[:4]
    return p1, p2


def probe_ckpt(
    ckpt_dir: Path,
    device: str,
    run_dir: Path | None = None,
    step_override: int | None = None,
) -> dict:
    p1_chars, p2_chars = load_probe_chars()
    meta_v2 = json.loads((ROOT / "data/retrain_v2/meta.json").read_text())
    style_char = (meta_v2.get("style_refs") or ["永"])[0]
    font_dir = Path("/root/data/font_50")
    content_p1 = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    content_p2 = Path(json.loads((ROOT / "data/unified_v1/meta.json").read_text())["content_cjk"])

    step = step_override
    if step is None and ckpt_dir.name.startswith("global_step_"):
        try:
            step = int(ckpt_dir.name.split("_")[-1])
        except ValueError:
            step = None
    if run_dir is None:
        run_dir = ckpt_dir.parent
    out_dir = run_dir / "demo8_probe" / (f"step_{step:06d}" if step is not None else ckpt_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "pred"
    pred_dir.mkdir(exist_ok=True)

    args = build_args(str(ckpt_dir), device)
    # slightly fewer steps for speed; still comparable across ckpts
    args.num_inference_steps = 15
    set_seed(args.seed)
    pipe = load_pipe(args)

    rows = []
    t0 = time.time()
    jobs = [("p1", PROBE_FONTS, p1_chars, content_p1, ROOT / "data/unified_v1/renders/64/gt_latin")]
    jobs.append(("p2", PROBE_FONTS, p2_chars, content_p2, ROOT / "data/unified_v1/renders/64/gt_p2"))

    with torch.no_grad():
        for split, fonts, chars, content_font, gt_root in jobs:
            for stem in fonts:
                fp = resolve_font(font_dir, stem)
                style_t = to_tensor(render(fp, style_char, 96), args.style_image_size).to(device)
                for ch in chars:
                    tag = ch if (ch.isalnum() and ord(ch) < 128) else f"u{ord(ch):04X}"
                    content_t = to_tensor(render(content_font, ch, 96), args.content_image_size).to(device)
                    images = pipe.generate(
                        content_images=content_t,
                        style_images=style_t,
                        batch_size=1,
                        order=args.order,
                        num_inference_step=args.num_inference_steps,
                        content_encoder_downsample_size=args.content_encoder_downsample_size,
                        t_start=args.t_start,
                        t_end=args.t_end,
                        dm_size=args.content_image_size,
                        algorithm_type=args.algorithm_type,
                        skip_type=args.skip_type,
                        method=args.method,
                        correcting_x0_fn=args.correcting_x0_fn,
                    )
                    pred = images[0]
                    pred.save(pred_dir / f"{split}_{stem}_{tag}.png")
                    gt = gt_root / stem / f"{tag}.png"
                    if not gt.exists():
                        gt = gt_root / stem / f"{ch}.png"
                    if not gt.exists():
                        continue
                    pa, ga = to_arr(pred, 96), to_arr(Image.open(gt), 96)
                    rows.append(
                        {
                            "split": split,
                            "font": stem,
                            "char": ch,
                            "L1": float(np.mean(np.abs(pa - ga))),
                            "SSIM": ssim(pa, ga),
                        }
                    )

    def agg(split: str | None = None):
        xs = [r for r in rows if split is None or r["split"] == split]
        if not xs:
            return {"n": 0, "L1": None, "SSIM": None}
        return {
            "n": len(xs),
            "L1": float(np.mean([r["L1"] for r in xs])),
            "SSIM": float(np.mean([r["SSIM"] for r in xs])),
        }

    metrics = {
        "ts": now_iso(),
        "run": run_dir.name,
        "ckpt": str(ckpt_dir),
        "step": step,
        "seconds": round(time.time() - t0, 1),
        "device": device,
        "inference_steps": args.num_inference_steps,
        "fonts": PROBE_FONTS,
        "chars_p1": "".join(p1_chars),
        "chars_p2": "".join(p2_chars),
        "overall": agg(),
        "p1": agg("p1"),
        "p2": agg("p2"),
        "note": "Demo-8 lightweight probe; style=永 1-shot; not full eval",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "rows.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    curve_path = run_dir / "demo8_probe" / "curve.jsonl"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    # de-dupe same step
    existing = []
    if curve_path.exists():
        for line in curve_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("step") == step:
                continue
            existing.append(rec)
    existing.append(
        {
            "ts": metrics["ts"],
            "step": step,
            "L1": metrics["overall"]["L1"],
            "SSIM": metrics["overall"]["SSIM"],
            "p1_L1": metrics["p1"]["L1"],
            "p1_SSIM": metrics["p1"]["SSIM"],
            "p2_L1": metrics["p2"]["L1"],
            "p2_SSIM": metrics["p2"]["SSIM"],
            "n": metrics["overall"]["n"],
            "seconds": metrics["seconds"],
            "ckpt": str(ckpt_dir),
        }
    )
    existing.sort(key=lambda r: (r.get("step") is None, r.get("step") or 0))
    with curve_path.open("w", encoding="utf-8") as f:
        for rec in existing:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    refresh_aggregate()
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return metrics


def refresh_aggregate() -> dict:
    FD = ROOT / "runs_retrain_v2/fontdiffuser"
    agg = {"updated": now_iso(), "runs": {}}
    for run_dir in sorted(FD.glob("ft*")):
        curve = run_dir / "demo8_probe" / "curve.jsonl"
        if not curve.exists():
            continue
        pts = [json.loads(l) for l in curve.read_text().splitlines() if l.strip()]
        # overfitting hint: last vs best SSIM
        ssims = [(p.get("step"), p.get("SSIM")) for p in pts if p.get("SSIM") is not None]
        hint = "insufficient"
        if len(ssims) >= 2:
            best = max(ssims, key=lambda x: x[1])
            last = ssims[-1]
            if last[0] != best[0] and last[1] < best[1] - 0.01:
                hint = f"possible_overfit: best SSIM@{best[0]}={best[1]:.3f} > last@{last[0]}={last[1]:.3f}"
            elif last[1] >= best[1] - 0.005:
                hint = "ok_or_still_improving"
            else:
                hint = "mild_drop"
        agg["runs"][run_dir.name] = {"points": pts, "hint": hint}
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "fd_probe_curves.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
    return agg


def list_pending(run_dir: Path) -> list[Path]:
    done_steps = set()
    curve = run_dir / "demo8_probe" / "curve.jsonl"
    if curve.exists():
        for line in curve.read_text().splitlines():
            if line.strip():
                done_steps.add(json.loads(line).get("step"))
    pending = []
    for p in sorted(run_dir.glob("global_step_*"), key=lambda x: int(x.name.split("_")[-1])):
        try:
            st = int(p.name.split("_")[-1])
        except ValueError:
            continue
        if (p / "unet.pth").exists() and st not in done_steps:
            pending.append(p)
    return pending


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="", help="single ckpt dir")
    ap.add_argument("--run_dir", type=str, default="", help="scan run for new ckpts")
    ap.add_argument("--step", type=int, default=None, help="override step label (e.g. 0 for official init)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--runs", nargs="*", default=["ft_p253_cnstyle", "ft_cnstyle"])
    args = ap.parse_args()
    FD = ROOT / "runs_retrain_v2/fontdiffuser"

    if args.ckpt:
        probe_ckpt(
            Path(args.ckpt),
            args.device,
            Path(args.run_dir) if args.run_dir else None,
            step_override=args.step,
        )
        return

    run_dirs = [FD / n for n in args.runs]

    def sweep():
        for rd in run_dirs:
            if not rd.exists():
                continue
            for ck in list_pending(rd):
                print(f"[probe] {rd.name} {ck.name}", flush=True)
                try:
                    probe_ckpt(ck, args.device, rd)
                except Exception as e:
                    print(f"[probe] FAIL {ck}: {e}", flush=True)

    if args.watch:
        print(f"[probe_watch] device={args.device} interval={args.interval}s runs={args.runs}", flush=True)
        while True:
            sweep()
            refresh_aggregate()
            time.sleep(args.interval)
    else:
        sweep()
        refresh_aggregate()


if __name__ == "__main__":
    main()
