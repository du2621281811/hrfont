#!/usr/bin/env python3
"""Eval FontDiffuser retrain_v2 ckpt on Demo-8 P1/P2 (same protocol as run_fontdiffuser_p1p2)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from accelerate.utils import set_seed
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/root/projects/hrfont")
FD_ROOT = ROOT / "repos" / "FontDiffuser"
sys.path.insert(0, str(FD_ROOT))
from src import (  # noqa: E402
    FontDiffuserDPMPipeline,
    FontDiffuserModelDPM,
    build_ddpm_scheduler,
    build_unet,
    build_content_encoder,
    build_style_encoder,
)
from configs.fontdiffuser import get_parser  # noqa: E402


def build_args(ckpt_dir, device):
    parser = get_parser()
    args, _ = parser.parse_known_args([])
    args.ckpt_dir = ckpt_dir
    args.device = device
    args.style_image_size = (96, 96)
    args.content_image_size = (96, 96)
    args.resolution = 96
    args.algorithm_type = "dpmsolver++"
    args.guidance_type = "classifier-free"
    args.guidance_scale = 7.5
    args.num_inference_steps = 20
    args.method = "multistep"
    args.order = 2
    args.seed = 123
    args.model_type = "noise"
    args.t_start = None
    args.t_end = None
    args.skip_type = "time_uniform"
    args.correcting_x0_fn = None
    return args


def load_pipe(args):
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth", map_location="cpu"))
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth", map_location="cpu"))
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(
        torch.load(f"{args.ckpt_dir}/content_encoder.pth", map_location="cpu")
    )
    model = FontDiffuserModelDPM(
        unet=unet, style_encoder=style_encoder, content_encoder=content_encoder
    )
    model.to(args.device).eval()
    train_scheduler = build_ddpm_scheduler(args=args)
    return FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )


def to_tensor(img, size):
    tfm = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    return tfm(img.convert("RGB")).unsqueeze(0)


def render(font_path, ch, size=96):
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    lo, hi, best = 8, size, None
    for _ in range(14):
        fs = (lo + hi) // 2
        f = ImageFont.truetype(str(font_path), fs)
        bbox = draw.textbbox((0, 0), ch, font=f)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= size - 8 and h <= size - 8:
            best = (f, bbox)
            lo = fs + 1
        else:
            hi = fs - 1
    if best is None:
        f = ImageFont.truetype(str(font_path), max(10, size // 2))
        bbox = draw.textbbox((0, 0), ch, font=f)
    else:
        f, bbox = best
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((size - w) // 2 - bbox[0], (size - h) // 2 - bbox[1]),
        ch,
        font=f,
        fill=(0, 0, 0),
    )
    return img


def to_arr(im, size=96):
    return np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0


def ssim(a, b):
    C1, C2 = 0.01**2, 0.03**2
    mu_a, mu_b = a.mean(), b.mean()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    return float(
        ((2 * mu_a * mu_b + C1) * (2 * sig_ab + C2))
        / ((mu_a**2 + mu_b**2 + C1) * (a.var() + b.var() + C2))
    )


def resolve_font(font_dir: Path, stem: str) -> Path:
    for ext in (".TTF", ".ttf", ".OTF", ".otf"):
        p = font_dir / f"{stem}{ext}"
        if p.exists():
            return p
    hits = list(font_dir.glob(stem + ".*"))
    if not hits:
        raise FileNotFoundError(stem)
    return hits[0]


def eval_split(split: str, ckpt_dir: Path, device: str, tag: str | None = None) -> dict:
    meta_u = json.loads((ROOT / "data/unified_v1/meta.json").read_text())
    meta_v2 = json.loads((ROOT / "data/retrain_v2/meta.json").read_text())
    fonts = meta_v2["test_fonts"]
    chars = list(meta_u["L_p1"] if split == "p1" else meta_u["L_p2"])
    content_font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if split == "p2":
        content_font = Path(meta_u["content_cjk"])
    font_dir = Path("/root/data/font_50")
    gt_root = ROOT / "data/unified_v1/renders/128" / ("gt_latin" if split == "p1" else "gt_p2")
    suffix = f"{split}_eval" if not tag else f"{split}_eval_{tag}"
    out_root = ROOT / "runs_retrain_v2/fontdiffuser" / suffix
    pred_dir = out_root / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)

    args = build_args(str(ckpt_dir), device)
    set_seed(args.seed)
    pipe = load_pipe(args)
    style_char = (meta_v2.get("style_refs") or ["永"])[0]
    rows = []
    t0 = time.time()
    with torch.no_grad():
        for fname in fonts:
            stem = Path(fname).stem if "." in fname else fname
            fp = resolve_font(font_dir, stem)
            style_img = render(fp, style_char, 96)
            style_t = to_tensor(style_img, args.style_image_size).to(device)
            for ch in chars:
                tag = ch if (ch.isalnum() and ord(ch) < 128) else f"u{ord(ch):04X}"
                content_img = render(content_font, ch, 96)
                content_t = to_tensor(content_img, args.content_image_size).to(device)
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
                pred_path = pred_dir / f"{stem}_{tag}.png"
                pred.save(pred_path)
                gt_path = gt_root / stem / f"{tag}.png"
                if not gt_path.exists():
                    gt_path = gt_root / stem / f"{ch}.png"
                if not gt_path.exists():
                    continue
                pa, ga = to_arr(pred), to_arr(Image.open(gt_path))
                rows.append(
                    {
                        "font": stem,
                        "char": ch,
                        "l1": float(np.mean(np.abs(pa - ga))),
                        "ssim": ssim(pa, ga),
                    }
                )
    metrics = {
        "method": "fontdiffuser",
        "split": split,
        "n": len(rows),
        "L1_mean": float(np.mean([r["l1"] for r in rows])) if rows else None,
        "SSIM_mean": float(np.mean([r["ssim"] for r in rows])) if rows else None,
        "l1": float(np.mean([r["l1"] for r in rows])) if rows else None,
        "ssim": float(np.mean([r["ssim"] for r in rows])) if rows else None,
        "seconds": round(time.time() - t0, 1),
        "ckpt": str(ckpt_dir),
        "tag": tag,
        "note": "eval style=CN 永 (1-shot); train should use StyleImage CN refs",
    }
    (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_root / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    print(json.dumps(metrics), flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["p1", "p2", "both"], default="both")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--ckpt_dir",
        default=str(ROOT / "runs/ft/global_step_30000"),
    )
    ap.add_argument("--tag", default=None, help="Write to p*_eval_<tag>/ instead of overwriting p*_eval/")
    args = ap.parse_args()
    splits = ["p1", "p2"] if args.split == "both" else [args.split]
    for s in splits:
        eval_split(s, Path(args.ckpt_dir), args.device, tag=args.tag)


if __name__ == "__main__":
    main()
