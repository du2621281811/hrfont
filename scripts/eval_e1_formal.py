#!/usr/bin/env python3
"""E1 formal offline eval: stratified / full-charset generation + metrics.

Does NOT touch training weights. Writes only under:
  runs/E1-FTV2-A-S3407/eval_formal/<eval_id>/

Protocol (frozen for reproducibility):
  - data: A protocol disk, split_v3 228/16/16
  - style ref: fixed 永 (first available of ref8 永和书风骨韵天地)
  - DPM++ / CFG 7.5 / 20 steps / seed(s) set via CLI
  - content = ContentImage; GT = TargetImage
  - multi-GPU: shard by font (--shard i/n), one process per GPU

Metrics (diagnostic; paper ID/style axes still need E12 φ_s2):
  L1, SSIM, ink coverage, optional LPIPS; aggregated overall / by script / by font.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/root/projects/hrfont")
RUN = ROOT / "runs/E1-FTV2-A-S3407"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
VARIANT = ROOT / "code/variants/cn2west_ft_v2/FontDiffuser"
P1 = ROOT / "code/official/FontDiffuser/ckpt"
VAL_STEMS = ROOT / "manifests/pipeline_v3_val_stems.txt"
TEST_STEMS = ROOT / "manifests/pipeline_v3_test_stems.txt"
CHARSET = ROOT / "manifests/charset_cn2west_v2_planned.json"
REF8 = list("永和书风骨韵天地")

# Closed-counter / topology-sensitive Latin (plan E8 interest set + near neighbors)
CLOSED_LATIN = list("aodpqbeg")

# Stratified probe: representative across 295 buckets without full cost
STRATIFIED = (
    list("0123456789")
    + list("AGMQRWBCO")
    + list("aodpqbegcilnu")
    + list("àéüāě")
    + list("あかさん")
    + list("アカン")
    + list("ㄅㄆㄚ")
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cp_of(ch: str) -> str:
    return f"u{ord(ch):04X}"


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def script_bucket(ch: str) -> str:
    if ch.isdigit():
        return "digit"
    name = unicodedata.name(ch, "")
    if "HIRAGANA" in name:
        return "hiragana"
    if "KATAKANA" in name:
        return "katakana"
    if "BOPOMOFO" in name:
        return "bopomofo"
    if "LATIN" in name and ch.isupper():
        return "latin_upper"
    if "LATIN" in name and ch.islower():
        return "latin_lower"
    if ch in CLOSED_LATIN:
        return "latin_lower_closed"
    return "other"


def load_chars(mode: str) -> list[str]:
    raw = json.loads(CHARSET.read_text(encoding="utf-8"))["target_string"]
    all_chars = list(raw)
    if mode == "all":
        return all_chars
    if mode == "stratified":
        # preserve order, unique, only chars present in target set
        allow = set(all_chars)
        out = []
        for ch in STRATIFIED:
            if ch in allow and ch not in out:
                out.append(ch)
        return out
    raise ValueError(mode)


def load_fonts(split: str) -> list[str]:
    p = VAL_STEMS if split == "val" else TEST_STEMS
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]


def content_path(split: str, ch: str) -> Path:
    for sp in (split, "train", "val", "test"):
        p = DATA / sp / "ContentImage" / f"{cp_of(ch)}.png"
        if p.is_file():
            return p
    raise FileNotFoundError(f"content {ch}")


def gt_path(split: str, stem: str, ch: str) -> Path | None:
    p = DATA / split / "TargetImage" / stem / f"{stem}+{cp_of(ch)}.png"
    return p if p.is_file() else None


def pick_style_path(split: str, stem: str) -> Path | None:
    d = DATA / split / "StyleImage" / stem
    if not d.is_dir():
        return None
    for ch in REF8:
        p = d / f"{stem}+{cp_of(ch)}.png"
        if p.is_file():
            return p
    pngs = sorted(d.glob("*.png"))
    return pngs[0] if pngs else None


def resolve_methods(names: list[str]) -> list[tuple[str, Path, int | None]]:
    """(method_id, ckpt_dir, step)."""
    out = []
    for name in names:
        if name in ("P1", "official_P1"):
            out.append(("P1", P1, 0))
            continue
        if name in ("E1_100k", "E1-100k", "100k"):
            d = RUN / "global_step_100000"
            out.append(("E1_100k", d, 100000))
            continue
        m = re.fullmatch(r"E1_(\d+)", name)
        if m:
            step = int(m.group(1))
            d = RUN / f"global_step_{step}"
            out.append((f"E1_{step}", d, step))
            continue
        if name == "E1_valbest":
            # pick formal 5k milestone with lowest offline val loss
            hist = RUN / "viz" / "val_loss_history.json"
            if not hist.is_file():
                hist = ROOT / "reports/e1_ft_v2_dashboard/val_loss_history.json"
            rows = json.loads(hist.read_text(encoding="utf-8"))
            # only exact 5k milestones
            cands = [
                r
                for r in rows
                if int(r.get("step", -1)) % 5000 == 0 and int(r["step"]) > 0
            ]
            if not cands:
                raise RuntimeError("no 5k val loss rows for E1_valbest")
            best = min(cands, key=lambda r: (float(r["loss"]), int(r["step"])))
            step = int(best["step"])
            out.append((f"E1_valbest_{step}", RUN / f"global_step_{step}", step))
            continue
        raise ValueError(f"unknown method: {name}")
    for mid, d, _ in out:
        for f in ("unet.pth", "style_encoder.pth", "content_encoder.pth"):
            if not (d / f).is_file():
                raise FileNotFoundError(f"{mid}: missing {d/f}")
    return out


def load_pipe(ckpt_dir: Path, device: str):
    sys.path.insert(0, str(VARIANT))
    import torch
    from types import SimpleNamespace
    from src import (
        FontDiffuserDPMPipeline,
        FontDiffuserModelDPM,
        build_ddpm_scheduler,
        build_unet,
        build_content_encoder,
        build_style_encoder,
    )

    args = SimpleNamespace(
        resolution=96,
        unet_channels=(64, 128, 256, 512),
        style_image_size=(96, 96),
        content_image_size=(96, 96),
        content_encoder_downsample_size=3,
        channel_attn=True,
        content_start_channel=64,
        style_start_channel=64,
        beta_scheduler="scaled_linear",
        model_type="noise",
    )
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    unet.load_state_dict(torch.load(ckpt_dir / "unet.pth", map_location="cpu", weights_only=True))
    style_encoder.load_state_dict(
        torch.load(ckpt_dir / "style_encoder.pth", map_location="cpu", weights_only=True)
    )
    content_encoder.load_state_dict(
        torch.load(ckpt_dir / "content_encoder.pth", map_location="cpu", weights_only=True)
    )
    model = FontDiffuserModelDPM(
        unet=unet, style_encoder=style_encoder, content_encoder=content_encoder
    ).to(device)
    model.eval()
    noise_scheduler = build_ddpm_scheduler(args)
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=noise_scheduler,
        version="V3",
        model_type="noise",
        guidance_type="classifier-free",
        guidance_scale=7.5,
    )
    return pipe, args


def to_tensor96(img: Image.Image, device: str):
    import torch
    import torchvision.transforms as T

    if img.size != (96, 96):
        raise ValueError(f"expected 96x96, got {img.size}")
    t = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    return t(img.convert("RGB"))[None].to(device)


def sample_one(pipe, args, content_img, style_img, device: str, seed: int):
    import torch
    from accelerate.utils import set_seed

    set_seed(seed)
    content = to_tensor96(content_img, device)
    style = to_tensor96(style_img, device)
    with torch.no_grad():
        images = pipe.generate(
            content_images=content,
            style_images=style,
            batch_size=1,
            order=2,
            num_inference_step=20,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
            t_start=None,
            t_end=None,
            dm_size=args.content_image_size,
            algorithm_type="dpmsolver++",
            skip_type="time_uniform",
            method="multistep",
            correcting_x0_fn=None,
        )
    im = images[0]
    if isinstance(im, Image.Image):
        return im.convert("RGB")
    arr = im.detach().cpu()
    if arr.ndim == 4:
        arr = arr[0]
    arr = ((arr.clamp(-1, 1) + 1) * 0.5 * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def to_gray01(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    c1, c2 = 0.01**2, 0.03**2
    mu_a, mu_b = a.mean(), b.mean()
    sig_a, sig_b = a.var(), b.var()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    return float(
        ((2 * mu_a * mu_b + c1) * (2 * sig_ab + c2))
        / ((mu_a**2 + mu_b**2 + c1) * (sig_a + sig_b + c2))
    )


def ink_coverage(a: np.ndarray, thr: float = 0.92) -> float:
    # assume near-white background
    return float((a < thr).mean())


def pred_name(split: str, stem: str, ch: str, seed: int) -> str:
    return f"{split}__{stem}__{cp_of(ch)}__s{seed}.png"


def parse_shard(s: str) -> tuple[int, int]:
    i, n = s.split("/")
    i, n = int(i), int(n)
    if not (0 <= i < n):
        raise ValueError(s)
    return i, n


def cmd_generate(args: argparse.Namespace) -> None:
    shard_i, shard_n = parse_shard(args.shard)
    chars = load_chars(args.chars)
    methods = resolve_methods(args.methods)
    seeds = [int(x) for x in args.seeds.split(",")]
    splits = []
    if args.split in ("test", "both"):
        splits.append("test")
    if args.split in ("val", "both"):
        splits.append("val")

    eval_root = RUN / "eval_formal" / args.eval_id
    pred_root = eval_root / "preds"
    eval_root.mkdir(parents=True, exist_ok=True)

    # write protocol once (race-ok with same content)
    proto = {
        "eval_id": args.eval_id,
        "created_at": utc_now(),
        "git_head": git_head(),
        "run_id": "E1-FTV2-A-S3407",
        "data_root": str(DATA.relative_to(ROOT)),
        "protocol": "A",
        "style_policy": "ref8_prefer_first_available",
        "ref8": "".join(REF8),
        "sampling": {
            "algorithm": "dpmsolver++",
            "guidance_scale": 7.5,
            "num_inference_steps": 20,
            "order": 2,
            "method": "multistep",
        },
        "chars_mode": args.chars,
        "chars_n": len(chars),
        "chars": chars,
        "seeds": seeds,
        "splits": splits,
        "methods": [
            {
                "id": mid,
                "ckpt": str(d),
                "step": step,
                "unet_sha256": sha256_file(d / "unet.pth"),
            }
            for mid, d, step in methods
        ],
        "notes": [
            "Generation is read-only w.r.t. training artifacts.",
            "L1/SSIM/LPIPS are diagnostic; primary paper axes need E12 φ_s2 / ID-CLS.",
            "test must not be used for ckpt selection.",
        ],
    }
    (eval_root / "PROTOCOL.json").write_text(
        json.dumps(proto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (eval_root / f"chars_{args.chars}.json").write_text(
        json.dumps({"chars": chars}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    device = args.device
    jobs = []
    for split in splits:
        fonts = load_fonts(split)
        fonts = [f for j, f in enumerate(fonts) if j % shard_n == shard_i]
        for stem in fonts:
            jobs.append((split, stem))

    log_path = eval_root / f"generate_shard{shard_i}of{shard_n}.log"
    done = 0
    skipped = 0
    errors = []
    t0 = time.time()

    def log(msg: str) -> None:
        line = f"[{utc_now()}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"start shard={shard_i}/{shard_n} jobs_fonts={len(jobs)} chars={len(chars)} methods={len(methods)} seeds={seeds} device={device}")

    for mid, ckpt_dir, step in methods:
        try:
            pipe, pargs = load_pipe(ckpt_dir, device)
        except Exception as e:
            log(f"LOAD_FAIL {mid}: {e}")
            errors.append({"method": mid, "error": f"load:{e}"})
            continue
        log(f"loaded {mid} step={step}")
        for split, stem in jobs:
            style_p = pick_style_path(split, stem)
            if style_p is None:
                errors.append({"method": mid, "split": split, "stem": stem, "error": "no_style"})
                continue
            style_img = Image.open(style_p).convert("RGB")
            out_dir = pred_root / mid / split / stem
            out_dir.mkdir(parents=True, exist_ok=True)
            for ch in chars:
                try:
                    c_img = Image.open(content_path(split, ch)).convert("RGB")
                except FileNotFoundError as e:
                    errors.append({"method": mid, "stem": stem, "ch": ch, "error": str(e)})
                    continue
                for seed in seeds:
                    out_p = out_dir / pred_name(split, stem, ch, seed)
                    if out_p.is_file() and not args.overwrite:
                        skipped += 1
                        continue
                    try:
                        pred = sample_one(pipe, pargs, c_img, style_img, device, seed)
                        pred.save(out_p)
                        # sidecar meta (small)
                        meta = {
                            "method": mid,
                            "split": split,
                            "font": stem,
                            "char": ch,
                            "cp": cp_of(ch),
                            "bucket": script_bucket(ch),
                            "seed": seed,
                            "style_path": str(style_p.relative_to(ROOT)),
                            "content_path": str(content_path(split, ch).relative_to(ROOT)),
                            "gt_path": (
                                str(gt_path(split, stem, ch).relative_to(ROOT))
                                if gt_path(split, stem, ch)
                                else None
                            ),
                        }
                        out_p.with_suffix(".json").write_text(
                            json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8"
                        )
                        done += 1
                        if done % 50 == 0:
                            rate = done / max(1e-6, time.time() - t0)
                            log(f"progress done={done} skipped={skipped} rate={rate:.2f}/s last={out_p.name}")
                    except Exception as e:
                        errors.append(
                            {
                                "method": mid,
                                "split": split,
                                "stem": stem,
                                "ch": ch,
                                "seed": seed,
                                "error": str(e),
                            }
                        )
        del pipe
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    summary = {
        "shard": f"{shard_i}/{shard_n}",
        "done": done,
        "skipped": skipped,
        "errors_n": len(errors),
        "errors": errors[:200],
        "elapsed_s": round(time.time() - t0, 2),
        "finished_at": utc_now(),
    }
    (eval_root / f"generate_shard{shard_i}of{shard_n}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(f"finished {summary}")


def aggregate_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}

    def mean(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(sum(vals) / len(vals)) if vals else None

    return {
        "n": len(rows),
        "L1_mean": mean("L1"),
        "SSIM_mean": mean("SSIM"),
        "coverage_mean": mean("coverage"),
        "LPIPS_mean": mean("LPIPS"),
        "blank_rate": float(sum(1 for r in rows if (r.get("coverage") or 0) < 0.005) / len(rows)),
        "overflow_rate": float(sum(1 for r in rows if (r.get("coverage") or 0) > 0.60) / len(rows)),
    }


def cmd_metrics(args: argparse.Namespace) -> None:
    eval_root = RUN / "eval_formal" / args.eval_id
    pred_root = eval_root / "preds"
    if not pred_root.is_dir():
        raise SystemExit(f"missing preds: {pred_root}")

    lpips_fn = None
    if args.lpips:
        import torch
        import lpips

        device = args.device
        lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    rows = []
    for mid_dir in sorted(pred_root.iterdir()):
        if not mid_dir.is_dir():
            continue
        mid = mid_dir.name
        for png in mid_dir.rglob("*.png"):
            meta_p = png.with_suffix(".json")
            if meta_p.is_file():
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
            else:
                # fallback parse
                parts = png.stem.split("__")
                if len(parts) < 4:
                    continue
                split, stem, cp, sseed = parts[0], parts[1], parts[2], parts[3]
                ch = chr(int(cp[1:], 16))
                meta = {
                    "method": mid,
                    "split": split,
                    "font": stem,
                    "char": ch,
                    "cp": cp,
                    "bucket": script_bucket(ch),
                    "seed": int(sseed[1:]),
                }
            split = meta["split"]
            stem = meta["font"]
            ch = meta["char"]
            gtp = gt_path(split, stem, ch)
            if gtp is None:
                continue
            pred = Image.open(png).convert("RGB")
            gt = Image.open(gtp).convert("RGB")
            pa, ga = to_gray01(pred), to_gray01(gt)
            rec = {
                **meta,
                "L1": float(np.abs(pa - ga).mean()),
                "SSIM": ssim(pa, ga),
                "coverage": ink_coverage(pa),
                "LPIPS": None,
            }
            if lpips_fn is not None:
                import torch

                def to_n11(im: Image.Image):
                    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
                    t = torch.from_numpy(arr).permute(2, 0, 1)[None] * 2 - 1
                    return t.to(args.device)

                with torch.no_grad():
                    rec["LPIPS"] = float(lpips_fn(to_n11(pred), to_n11(gt)).item())
            rows.append(rec)

    by_method: dict[str, list] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    report = {
        "eval_id": args.eval_id,
        "computed_at": utc_now(),
        "git_head": git_head(),
        "n_total": len(rows),
        "methods": {},
        "caveat": "L1/SSIM/LPIPS vs GT are calibration diagnostics for cross-script; not the primary style claim.",
    }
    for mid, rs in sorted(by_method.items()):
        by_split: dict[str, list] = defaultdict(list)
        by_bucket: dict[str, list] = defaultdict(list)
        by_font: dict[str, list] = defaultdict(list)
        closed = [r for r in rs if r["char"] in CLOSED_LATIN]
        for r in rs:
            by_split[r["split"]].append(r)
            by_bucket[r["bucket"]].append(r)
            by_font[r["font"]].append(r)
        report["methods"][mid] = {
            "overall": aggregate_rows(rs),
            "by_split": {k: aggregate_rows(v) for k, v in sorted(by_split.items())},
            "by_bucket": {k: aggregate_rows(v) for k, v in sorted(by_bucket.items())},
            "by_font": {k: aggregate_rows(v) for k, v in sorted(by_font.items())},
            "closed_latin": aggregate_rows(closed),
        }

    out_json = eval_root / "metrics_summary.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # per-item csv-ish jsonl
    with (eval_root / "metrics_per_item.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # markdown brief
    lines = [
        f"# E1 formal metrics — `{args.eval_id}`",
        "",
        f"- computed: `{report['computed_at']}`",
        f"- git: `{report['git_head'][:12]}`",
        f"- n: **{report['n_total']}**",
        "",
        "> L1/SSIM/LPIPS 是相对 GT 的诊断轴；正式论文风格轴需 E12 φ_s2。",
        "",
    ]
    for mid, block in report["methods"].items():
        o = block["overall"]
        lines.append(f"## {mid}")
        lines.append("")
        lines.append(
            f"- overall: n={o['n']}  L1={o['L1_mean']:.5f}  SSIM={o['SSIM_mean']:.4f}  "
            f"cov={o['coverage_mean']:.4f}  blank={o['blank_rate']:.3f}  overflow={o['overflow_rate']:.3f}"
            + (f"  LPIPS={o['LPIPS_mean']:.4f}" if o.get("LPIPS_mean") is not None else "")
        )
        cl = block["closed_latin"]
        if cl.get("n"):
            lines.append(
                f"- closed Latin {{a,o,d,p,q,b,e,g}}: n={cl['n']}  L1={cl['L1_mean']:.5f}  SSIM={cl['SSIM_mean']:.4f}"
            )
        lines.append("- by script bucket:")
        for b, agg in block["by_bucket"].items():
            lines.append(
                f"  - `{b}`: n={agg['n']}  L1={agg['L1_mean']:.5f}  SSIM={agg['SSIM_mean']:.4f}"
            )
        lines.append("")
    (eval_root / "METRICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v["overall"] for k, v in report["methods"].items()}, indent=2))
    print(f"wrote {out_json}")


def _web_rel(path: Path, eval_root: Path) -> str:
    """Relative path from eval_root; do not resolve symlinks (keeps preds/ local)."""
    try:
        return os.path.relpath(path, eval_root).replace("\\", "/")
    except ValueError:
        return str(path)


def _data_url(path: Path, data_root: Path) -> str:
    """Absolute URL path when http.server is rooted at data/."""
    return "/" + os.path.relpath(path, data_root).replace("\\", "/")


def cmd_verify(args: argparse.Namespace) -> None:
    """Check preds/meta/metrics index integrity; write VERIFY.json."""
    eval_root = RUN / "eval_formal" / args.eval_id
    proto = json.loads((eval_root / "PROTOCOL.json").read_text(encoding="utf-8"))
    chars = proto["chars"]
    seeds = proto["seeds"]
    splits = proto["splits"]
    methods = [m["id"] for m in proto["methods"]]
    issues: list[dict] = []
    ok = 0
    style_yong = 0
    for mid in methods:
        for split in splits:
            fonts = load_fonts(split)
            for stem in fonts:
                for ch in chars:
                    for seed in seeds:
                        name = pred_name(split, stem, ch, seed)
                        png = eval_root / "preds" / mid / split / stem / name
                        meta_p = png.with_suffix(".json")
                        if not png.is_file():
                            issues.append({"type": "missing_png", "path": str(png)})
                            continue
                        if not meta_p.is_file():
                            issues.append({"type": "missing_meta", "path": str(meta_p)})
                            continue
                        meta = json.loads(meta_p.read_text(encoding="utf-8"))
                        expect = {
                            "method": mid,
                            "split": split,
                            "font": stem,
                            "char": ch,
                            "cp": cp_of(ch),
                            "seed": seed,
                        }
                        for k, v in expect.items():
                            if meta.get(k) != v:
                                issues.append(
                                    {
                                        "type": "meta_mismatch",
                                        "path": str(meta_p),
                                        "field": k,
                                        "got": meta.get(k),
                                        "want": v,
                                    }
                                )
                        # style should prefer 永
                        sp = meta.get("style_path") or ""
                        if sp.endswith("+u6C38.png"):
                            style_yong += 1
                        else:
                            issues.append({"type": "style_not_yong", "path": sp, "font": stem})
                        # referenced files exist
                        for key in ("style_path", "content_path", "gt_path"):
                            rel = meta.get(key)
                            if not rel:
                                if key == "gt_path":
                                    issues.append({"type": "missing_gt_ref", "font": stem, "ch": ch})
                                continue
                            if not (ROOT / rel).is_file():
                                issues.append({"type": "broken_ref", "key": key, "path": rel})
                        ok += 1

    expected = (
        len(methods)
        * sum(len(load_fonts(s)) for s in splits)
        * len(chars)
        * len(seeds)
    )
    # metrics coverage
    metrics_n = None
    metrics_path = eval_root / "metrics_summary.json"
    if metrics_path.is_file():
        ms = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics_n = {mid: ms["methods"][mid]["overall"]["n"] for mid in ms.get("methods", {})}

    # duplicate method detection (same unet sha)
    shas = {m["id"]: m.get("unet_sha256") for m in proto["methods"]}
    dup_pairs = []
    ids = list(shas)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if shas[ids[i]] and shas[ids[i]] == shas[ids[j]]:
                dup_pairs.append([ids[i], ids[j]])

    report = {
        "eval_id": args.eval_id,
        "checked_at": utc_now(),
        "expected_n": expected,
        "ok_n": ok,
        "issues_n": len(issues),
        "issues_sample": issues[:50],
        "style_yong_n": style_yong,
        "metrics_n": metrics_n,
        "duplicate_unet_methods": dup_pairs,
        "pass": len(issues) == 0 and ok == expected,
    }
    out = eval_root / "VERIFY.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("pass", "ok_n", "expected_n", "issues_n", "style_yong_n", "duplicate_unet_methods", "metrics_n")}, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


def cmd_gallery(args: argparse.Namespace) -> None:
    """Build interactive browser (index.html + browse_index.json). Full matrix via selectors."""
    eval_root = RUN / "eval_formal" / args.eval_id
    pred_root = eval_root / "preds"
    proto = json.loads((eval_root / "PROTOCOL.json").read_text(encoding="utf-8"))
    chars = proto["chars"]
    seeds = proto["seeds"]
    default_seed = seeds[0]
    methods_all = [m["id"] for m in proto["methods"]]
    # Prefer unique methods for side-by-side (drop duplicate unet sha aliases)
    sha_to_id: dict[str, str] = {}
    methods_show: list[str] = []
    aliases: dict[str, str] = {}
    for m in proto["methods"]:
        sha = m.get("unet_sha256") or m["id"]
        if sha in sha_to_id:
            aliases[m["id"]] = sha_to_id[sha]
            continue
        sha_to_id[sha] = m["id"]
        methods_show.append(m["id"])

    # per-item metrics lookup
    per_item: dict[tuple, dict] = {}
    mpi = eval_root / "metrics_per_item.jsonl"
    if mpi.is_file():
        with mpi.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                key = (r["method"], r["split"], r["font"], r["char"], int(r["seed"]))
                per_item[key] = {
                    "L1": r.get("L1"),
                    "SSIM": r.get("SSIM"),
                    "LPIPS": r.get("LPIPS"),
                    "coverage": r.get("coverage"),
                }

    items = []
    missing = 0
    for split in proto["splits"]:
        fonts = load_fonts(split)
        for stem in fonts:
            style_p = pick_style_path(split, stem)
            for ch in chars:
                bucket = script_bucket(ch)
                try:
                    c_abs = content_path(split, ch)
                    # Prefer data/-rooted URL so :8777 /e1_formal_eval/ can load siblings
                    c_rel = _data_url(c_abs, ROOT / "data")
                except FileNotFoundError:
                    c_rel = None
                    missing += 1
                gtp = gt_path(split, stem, ch)
                gt_rel = _data_url(gtp, ROOT / "data") if gtp else None
                style_rel = _data_url(style_p, ROOT / "data") if style_p else None
                preds = {}
                metrics = {}
                for mid in methods_show:
                    png = eval_root / "preds" / mid / split / stem / pred_name(split, stem, ch, default_seed)
                    if png.is_file():
                        # keep local relative path under eval dir (works with data/e1_formal_eval symlink)
                        preds[mid] = f"preds/{mid}/{split}/{stem}/{pred_name(split, stem, ch, default_seed)}"
                        metrics[mid] = per_item.get((mid, split, stem, ch, default_seed))
                    else:
                        missing += 1
                items.append(
                    {
                        "split": split,
                        "font": stem,
                        "char": ch,
                        "cp": cp_of(ch),
                        "bucket": bucket,
                        "closed": ch in CLOSED_LATIN,
                        "seed": default_seed,
                        "content": c_rel,
                        "style": style_rel,
                        "gt": gt_rel,
                        "preds": preds,
                        "metrics": metrics,
                    }
                )

    summary = None
    ms_path = eval_root / "metrics_summary.json"
    if ms_path.is_file():
        ms = json.loads(ms_path.read_text(encoding="utf-8"))
        summary = {
            mid: ms["methods"][mid]["overall"]
            for mid in methods_show
            if mid in ms.get("methods", {})
        }
        bucket_cmp = {}
        if len(methods_show) >= 2 and all(m in ms["methods"] for m in methods_show[:2]):
            a, b = methods_show[0], methods_show[1]
            # show P1 vs E1 if both present
            if "P1" in methods_show and any(x.startswith("E1") for x in methods_show):
                a = "P1"
                b = next(x for x in methods_show if x.startswith("E1"))
            ba = ms["methods"][a]["by_bucket"]
            bb = ms["methods"][b]["by_bucket"]
            for buck in sorted(set(ba) | set(bb)):
                bucket_cmp[buck] = {
                    a: ba.get(buck),
                    b: bb.get(buck),
                }
        else:
            bucket_cmp = {}
    else:
        bucket_cmp = {}

    verify = None
    if (eval_root / "VERIFY.json").is_file():
        verify = json.loads((eval_root / "VERIFY.json").read_text(encoding="utf-8"))

    payload = {
        "eval_id": args.eval_id,
        "built_at": utc_now(),
        "protocol": {
            "chars_n": len(chars),
            "chars": chars,
            "seeds": seeds,
            "splits": proto["splits"],
            "sampling": proto["sampling"],
            "ref8": proto["ref8"],
            "style_policy": proto["style_policy"],
        },
        "methods_all": methods_all,
        "methods_show": methods_show,
        "method_aliases": aliases,
        "summary": summary,
        "bucket_cmp": bucket_cmp,
        "verify": {
            "pass": (verify or {}).get("pass"),
            "ok_n": (verify or {}).get("ok_n"),
            "expected_n": (verify or {}).get("expected_n"),
            "issues_n": (verify or {}).get("issues_n"),
        }
        if verify
        else None,
        "items_n": len(items),
        "missing_pred_slots": missing,
        "items": items,
    }
    (eval_root / "browse_index.json").write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Interactive browser: script-bucket grid (all glyphs of one script at once)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>E1 正式评测 · {args.eval_id}</title>
<style>
:root {{
  --bg:#e8edf2; --card:#fff; --line:#cfd6de; --muted:#5a6570; --ink:#15202b;
  --accent:#1f4a6f; --ok:#1b7f4a; --bad:#a33; --chip:#f4f7fa;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:14px/1.4 "IBM Plex Sans","Noto Sans SC",system-ui,sans-serif; background:var(--bg); color:var(--ink); }}
header {{ position:sticky; top:0; z-index:20; background:rgba(255,255,255,.96); backdrop-filter:blur(8px); border-bottom:1px solid var(--line); padding:10px 16px; }}
h1 {{ margin:0; font-size:1.15rem; }}
.meta {{ color:var(--muted); font-size:12px; margin-top:2px; }}
.toolbar {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:10px; }}
.chips {{ display:flex; flex-wrap:wrap; gap:6px; }}
.chip {{ border:1px solid var(--line); background:var(--chip); padding:6px 10px; border-radius:999px; cursor:pointer; font:inherit; color:var(--ink); }}
.chip.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
.chip .n {{ opacity:.75; font-size:11px; margin-left:4px; }}
label.ctl {{ display:flex; align-items:center; gap:6px; font-size:12px; color:var(--muted); }}
select, button.btn {{ padding:6px 9px; border:1px solid var(--line); border-radius:6px; background:#fff; font:inherit; }}
button.btn {{ cursor:pointer; }}
main {{ padding:12px 14px 60px; max-width:100%; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px; margin-bottom:12px; }}
.card h2 {{ margin:0 0 8px; font-size:.95rem; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; border:1px solid var(--line); }}
.badge.ok {{ color:var(--ok); background:#eef8f1; }}
.badge.bad {{ color:var(--bad); background:#f8eeee; }}
table.summary, table.bucket {{ border-collapse:collapse; width:100%; font-size:12px; }}
th, td {{ border-bottom:1px solid var(--line); padding:4px 6px; text-align:left; }}
.tag {{ font-family:ui-monospace,monospace; color:var(--accent); }}
.delta {{ color:var(--ok); }} .delta.worse {{ color:var(--bad); }}
.legend {{ display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color:var(--muted); margin:6px 0 10px; }}
.legend b {{ color:var(--ink); font-weight:600; }}
.font-block {{ margin:0 0 18px; }}
.font-block h3 {{ margin:0 0 8px; font-size:13px; font-family:ui-monospace,monospace; color:var(--accent); position:sticky; top:68px; background:var(--bg); padding:6px 0; z-index:5; }}
.glyph-row {{ display:flex; gap:8px; overflow-x:auto; padding-bottom:6px; }}
.glyph {{ flex:0 0 auto; width:auto; background:#fff; border:1px solid var(--line); border-radius:8px; padding:6px; }}
.glyph .ch {{ text-align:center; font-size:13px; font-weight:600; margin-bottom:4px; }}
.glyph .ch small {{ display:block; font-weight:400; color:var(--muted); font-size:10px; font-family:ui-monospace,monospace; }}
.stack {{ display:flex; gap:4px; align-items:flex-end; }}
.stack figure {{ margin:0; text-align:center; }}
.stack img {{ width:72px; height:72px; image-rendering:pixelated; border:1px solid #e2e7ee; background:#fff; display:block; }}
.stack figcaption {{ font-size:9px; color:var(--muted); margin-top:2px; }}
.stack .score {{ font-size:9px; font-family:ui-monospace,monospace; color:var(--ink); max-width:72px; }}
.hint {{ font-size:12px; color:var(--muted); }}
details.stats {{ margin-bottom:12px; }}
details.stats > summary {{ cursor:pointer; font-weight:600; }}
.empty {{ padding:24px; color:var(--muted); }}
</style>
</head>
<body>
<header>
  <h1>E1 正式评测 · 按语种一次看全</h1>
  <div class="meta"><span class="tag">{args.eval_id}</span> · 点语种芯片 → 该语种下全部测试字×全部字体直接铺开 · ←→ 切换语种</div>
  <div class="toolbar">
    <div class="chips" id="chips"></div>
  </div>
  <div class="toolbar">
    <label class="ctl">字体
      <select id="sel-font"><option value="ALL">全部字体</option></select>
    </label>
    <label class="ctl">显示列
      <select id="sel-cols">
        <option value="gt_p1_e1">GT | P1 | E1</option>
        <option value="p1_e1">P1 | E1</option>
        <option value="e1">仅 E1</option>
        <option value="gt_e1">GT | E1</option>
      </select>
    </label>
    <label class="ctl"><input type="checkbox" id="chk-closed"/> 仅闭合拉丁 aodpqbeg</label>
    <span class="hint" id="count"></span>
  </div>
</header>
<main>
  <details class="card stats" id="stats">
    <summary>总览指标 / 索引状态（点击展开）</summary>
    <div id="status" style="margin:8px 0"></div>
    <div id="summary"></div>
    <div style="overflow-x:auto;margin-top:8px"><table class="bucket" id="bucket-table"><thead></thead><tbody></tbody></table></div>
  </details>
  <section class="card">
    <div class="legend" id="legend"></div>
    <div id="grid"></div>
  </section>
</main>
<script>
const BUCKET_LABEL = {{
  digit:'数字', latin_upper:'拉丁大写', latin_lower:'拉丁小写',
  hiragana:'平假名', katakana:'片假名', bopomofo:'注音', other:'其他'
}};
async function main() {{
  const DATA = await (await fetch('browse_index.json?t='+Date.now())).json();
  const methods = DATA.methods_show;
  const base = methods.includes('P1') ? 'P1' : methods[0];
  const other = methods.find(m => m !== base) || methods[0];
  const items = DATA.items;
  const order = DATA.protocol.chars;

  // status
  const v = DATA.verify || {{}};
  document.getElementById('status').innerHTML =
    `<span class="badge ${{v.pass?'ok':'bad'}}">${{v.pass?'VERIFY PASS':'VERIFY ?'}}</span>
     · items ${{DATA.items_n}} · 展示 ${{methods.join(' / ')}}
     ${{Object.keys(DATA.method_aliases||{{}}).length?(' · 折叠别名 '+Object.keys(DATA.method_aliases).join(',')):''}}`;

  // summary table
  if (DATA.summary) {{
    let t = '<table class="summary"><thead><tr><th>method</th><th>n</th><th>L1↓</th><th>SSIM↑</th><th>LPIPS↓</th></tr></thead><tbody>';
    for (const mid of methods) {{
      const o = DATA.summary[mid]; if (!o) continue;
      t += `<tr><td class="tag">${{mid}}</td><td>${{o.n}}</td><td>${{Number(o.L1_mean).toFixed(5)}}</td><td>${{Number(o.SSIM_mean).toFixed(4)}}</td><td>${{o.LPIPS_mean==null?'—':Number(o.LPIPS_mean).toFixed(4)}}</td></tr>`;
    }}
    document.getElementById('summary').innerHTML = t + '</tbody></table>';
  }}
  const bt = document.querySelector('#bucket-table thead');
  const bb = document.querySelector('#bucket-table tbody');
  if (DATA.bucket_cmp) {{
    bt.innerHTML = `<tr><th>语种</th><th>${{base}} L1</th><th>${{other}} L1</th><th>ΔL1</th></tr>`;
    Object.keys(DATA.bucket_cmp).sort().forEach(buck => {{
      const row = DATA.bucket_cmp[buck];
      const a = row[base], b = row[other];
      if (!a||!b) return;
      const d = b.L1_mean - a.L1_mean;
      bb.insertAdjacentHTML('beforeend', `<tr><td>${{BUCKET_LABEL[buck]||buck}}</td><td>${{a.L1_mean.toFixed(5)}}</td><td>${{b.L1_mean.toFixed(5)}}</td><td class="${{d<0?'delta':'delta worse'}}">${{d.toFixed(5)}}</td></tr>`);
    }});
  }}

  // bucket chips
  const buckets = [...new Set(items.map(i => i.bucket))];
  buckets.sort((a,b) => (BUCKET_LABEL[a]||a).localeCompare(BUCKET_LABEL[b]||b,'zh'));
  const chips = document.getElementById('chips');
  let curBucket = buckets.includes('latin_lower') ? 'latin_lower' : buckets[0];
  function bucketCount(b) {{
    return items.filter(i => i.bucket===b).length;
  }}
  function renderChips() {{
    chips.innerHTML = '';
    buckets.forEach(b => {{
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'chip' + (b===curBucket?' active':'');
      btn.innerHTML = `${{BUCKET_LABEL[b]||b}}<span class="n">${{bucketCount(b)}}</span>`;
      btn.onclick = () => {{ curBucket = b; renderChips(); fillFonts(); renderGrid(); }};
      chips.appendChild(btn);
    }});
  }}

  const selFont = document.getElementById('sel-font');
  const selCols = document.getElementById('sel-cols');
  const chkClosed = document.getElementById('chk-closed');
  const grid = document.getElementById('grid');
  const count = document.getElementById('count');
  const legend = document.getElementById('legend');

  function fillFonts() {{
    const fonts = [...new Set(items.filter(i => i.bucket===curBucket).map(i => i.font))].sort();
    const keep = selFont.value;
    selFont.innerHTML = '<option value="ALL">全部字体</option>' + fonts.map(f => `<option value="${{f}}">${{f}}</option>`).join('');
    if ([...selFont.options].some(o => o.value===keep)) selFont.value = keep;
    else selFont.value = 'ALL';
  }}

  function fig(label, src, score) {{
    return `<figure><img loading="lazy" decoding="async" src="${{src||''}}" alt="${{label}}" onerror="this.style.opacity=.2"/><figcaption>${{label}}</figcaption>${{score?`<div class="score">${{score}}</div>`:''}}</figure>`;
  }}

  function scoreHtml(it, mid) {{
    const m = (it.metrics||{{}})[mid];
    if (!m || m.L1==null) return '';
    let s = `L1 ${{Number(m.L1).toFixed(3)}}`;
    if (mid !== base) {{
      const mb = (it.metrics||{{}})[base];
      if (mb && mb.L1!=null) {{
        const d = m.L1 - mb.L1;
        s += ` <span class="${{d<0?'delta':'delta worse'}}">${{d<0?'':'+'}}${{d.toFixed(3)}}</span>`;
      }}
    }}
    return s;
  }}

  function renderGrid() {{
    let pool = items.filter(i => i.bucket === curBucket);
    if (chkClosed.checked) pool = pool.filter(i => i.closed);
    if (selFont.value !== 'ALL') pool = pool.filter(i => i.font === selFont.value);
    const fonts = [...new Set(pool.map(i => i.font))].sort();
    const chars = [...new Set(pool.map(i => i.char))];
    chars.sort((a,b) => order.indexOf(a) - order.indexOf(b));
    const mode = selCols.value;
    const cols = [];
    if (mode.includes('gt')) cols.push('gt');
    if (mode.includes('p1') || mode === 'p1_e1') cols.push(base);
    if (mode.includes('e1') || mode === 'p1_e1' || mode === 'e1' || mode === 'gt_e1') cols.push(other);

    legend.innerHTML = `<b>${{BUCKET_LABEL[curBucket]||curBucket}}</b>
      · ${{chars.length}} 字 × ${{fonts.length}} 字体 = ${{pool.length}} 组
      · 每格列顺序：${{cols.map(c => c==='gt'?'GT':c).join(' | ')}}
      · 图下数字为相对 GT 的 L1（诊断）`;
    count.textContent = `当前 ${{pool.length}} 组`;

    if (!pool.length) {{
      grid.innerHTML = '<div class="empty">该语种下无样本</div>';
      return;
    }}

    // index for quick lookup
    const map = new Map();
    pool.forEach(it => map.set(it.font + '||' + it.char, it));

    let html = '';
    for (const font of fonts) {{
      html += `<div class="font-block"><h3>${{font}}</h3><div class="glyph-row">`;
      for (const ch of chars) {{
        const it = map.get(font + '||' + ch);
        if (!it) continue;
        let stack = '';
        for (const c of cols) {{
          if (c === 'gt') stack += fig('GT', it.gt, '');
          else stack += fig(c, (it.preds||{{}})[c], scoreHtml(it, c));
        }}
        html += `<div class="glyph"><div class="ch">'${{ch}}'<small>${{it.cp}}</small></div><div class="stack">${{stack}}</div></div>`;
      }}
      html += `</div></div>`;
    }}
    grid.innerHTML = html;
  }}

  selFont.onchange = renderGrid;
  selCols.onchange = renderGrid;
  chkClosed.onchange = () => {{
    // closed filter only meaningful for latin_lower; still apply
    renderGrid();
  }};
  document.addEventListener('keydown', (e) => {{
    if (e.target && (e.target.tagName==='SELECT' || e.target.tagName==='INPUT')) return;
    const i = buckets.indexOf(curBucket);
    if (e.key === 'ArrowLeft' && i > 0) {{ curBucket = buckets[i-1]; renderChips(); fillFonts(); renderGrid(); }}
    if (e.key === 'ArrowRight' && i < buckets.length-1) {{ curBucket = buckets[i+1]; renderChips(); fillFonts(); renderGrid(); }}
  }});

  renderChips();
  fillFonts();
  renderGrid();
}}
main().catch(err => {{
  document.getElementById('grid').innerHTML = '<div class="empty">加载失败：'+err+'</div>';
}});
</script>
</body>
</html>
"""
    out = eval_root / "index.html"
    out.write_text(html, encoding="utf-8")
    (eval_root / "gallery.html").write_text(
        '<!doctype html><meta http-equiv="refresh" content="0; url=index.html"/>'
        '<p><a href="index.html">Open interactive browser</a></p>\n',
        encoding="utf-8",
    )
    print(f"wrote {out} items={len(items)} methods_show={methods_show} aliases={aliases}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate preds (optionally sharded)")
    g.add_argument("--eval-id", required=True)
    g.add_argument("--methods", nargs="+", default=["P1", "E1_100k", "E1_valbest"])
    g.add_argument("--split", choices=["test", "val", "both"], default="test")
    g.add_argument("--chars", choices=["stratified", "all"], default="stratified")
    g.add_argument("--seeds", default="3407")
    g.add_argument("--shard", default="0/1", help="i/n font shard")
    g.add_argument("--device", default="cuda:0")
    g.add_argument("--overwrite", action="store_true")
    g.set_defaults(func=cmd_generate)

    m = sub.add_parser("metrics", help="Compute L1/SSIM/coverage/(LPIPS)")
    m.add_argument("--eval-id", required=True)
    m.add_argument("--lpips", action="store_true")
    m.add_argument("--device", default="cuda:0")
    m.set_defaults(func=cmd_metrics)

    gal = sub.add_parser("gallery", help="Build interactive HTML browser + browse_index.json")
    gal.add_argument("--eval-id", required=True)
    gal.set_defaults(func=cmd_gallery)

    v = sub.add_parser("verify", help="Verify preds/meta/refs index integrity")
    v.add_argument("--eval-id", required=True)
    v.set_defaults(func=cmd_verify)

    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
