#!/usr/bin/env python3
"""Mid-training visual preview: sample one ckpt on test fonts and build a contact sheet.

Read-only w.r.t. runs/: writes only under --out. Safe to run while training.

Example:
  python scripts/preview_ckpt_grid.py \
    --variant code/variants/cn2west_f0_rsifree/FontDiffuser \
    --ckpt runs/F0-RSIFREE-FT-A-S3407/global_step_75000 \
    --label F0@75k --out reports/preview_f0_75k --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path("/root/projects/hrfont")
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
TEST_STEMS = ROOT / "manifests/pipeline_v3_test_stems.txt"
REF8 = list("永和书风骨韵天地")
DEFAULT_CHARS = "AGae g5àあアㄅ".replace(" ", "")


def cp_of(ch: str) -> str:
    return f"u{ord(ch):04X}"


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


def load_pipe(variant: Path, ckpt_dir: Path, device: str):
    sys.path.insert(0, str(variant))
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
        torch.load(ckpt_dir / "style_encoder.pth", map_location="cpu", weights_only=True))
    content_encoder.load_state_dict(
        torch.load(ckpt_dir / "content_encoder.pth", map_location="cpu", weights_only=True))
    model = FontDiffuserModelDPM(
        unet=unet, style_encoder=style_encoder, content_encoder=content_encoder).to(device)
    model.eval()
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=build_ddpm_scheduler(args),
        version="V3",
        model_type="noise",
        guidance_type="classifier-free",
        guidance_scale=7.5)
    return pipe, args


def to_tensor96(img: Image.Image, device: str):
    import torchvision.transforms as T

    if img.size != (96, 96):
        raise ValueError(f"expected 96x96, got {img.size}")
    t = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    return t(img.convert("RGB"))[None].to(device)


def sample_one(pipe, args, content_img, style_img, device: str, seed: int) -> Image.Image:
    import torch
    from accelerate.utils import set_seed

    set_seed(seed)
    with torch.no_grad():
        images = pipe.generate(
            content_images=to_tensor96(content_img, device),
            style_images=to_tensor96(style_img, device),
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
            correcting_x0_fn=None)
    im = images[0]
    if isinstance(im, Image.Image):
        return im.convert("RGB")
    arr = im.detach().cpu()
    if arr.ndim == 4:
        arr = arr[0]
    arr = ((arr.clamp(-1, 1) + 1) * 0.5 * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def build_sheet(rows: list[tuple[str, list[Image.Image]]], chars: list[str], label: str) -> Image.Image:
    cell, pad, head, left = 96, 6, 26, 150
    n_col = len(chars)
    n_row = len(rows)
    w = left + n_col * (cell + pad) + pad
    h = head + n_row * (cell + pad) + pad
    sheet = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(sheet)
    d.text((8, 8), label, fill="black")
    for j, ch in enumerate(chars):
        d.text((left + j * (cell + pad) + cell // 2 - 4, 10), ch, fill="black")
    for i, (name, imgs) in enumerate(rows):
        y = head + i * (cell + pad)
        d.text((8, y + cell // 2 - 6), name[:22], fill="black")
        for j, im in enumerate(imgs):
            sheet.paste(im.resize((cell, cell)), (left + j * (cell + pad), y))
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fonts", type=int, default=4)
    ap.add_argument("--chars", default=DEFAULT_CHARS)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    variant = Path(a.variant) if Path(a.variant).is_absolute() else ROOT / a.variant
    ckpt = Path(a.ckpt) if Path(a.ckpt).is_absolute() else ROOT / a.ckpt
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    chars = list(dict.fromkeys(a.chars))
    stems = [l.strip() for l in TEST_STEMS.read_text().splitlines() if l.strip()][: a.fonts]
    pipe, args = load_pipe(variant, ckpt, a.device)

    rows: list[tuple[str, list[Image.Image]]] = []
    for stem in stems:
        style_p = pick_style_path(a.split, stem)
        if style_p is None:
            print(f"skip {stem}: no style", file=sys.stderr)
            continue
        style_img = Image.open(style_p)
        gts, preds = [], []
        for ch in chars:
            content_img = Image.open(content_path(a.split, ch))
            pred = sample_one(pipe, args, content_img, style_img, a.device, a.seed)
            pred.save(out / f"{stem}+{cp_of(ch)}.png")
            preds.append(pred)
            g = gt_path(a.split, stem, ch)
            gts.append(Image.open(g).convert("RGB") if g else Image.new("RGB", (96, 96), "white"))
        rows.append((f"{stem} GT", gts))
        rows.append((f"{stem} {a.label}", preds))
        print(f"done {stem}", flush=True)

    sheet = build_sheet(rows, chars, f"{a.label}  ckpt={ckpt.name}  split={a.split}  seed={a.seed}")
    sheet_p = out / "contact_sheet.png"
    sheet.save(sheet_p)
    (out / "preview_meta.json").write_text(json.dumps({
        "label": a.label,
        "variant": str(variant),
        "ckpt": str(ckpt),
        "split": a.split,
        "fonts": stems,
        "chars": chars,
        "seed": a.seed,
        "sampler": "dpmsolver++ order2 20 steps, CFG 7.5",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"sheet={sheet_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
