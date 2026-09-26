#!/usr/bin/env python3
"""Batch FontDiffuser inference for CN-style -> Latin content."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from accelerate.utils import set_seed

ROOT = Path(__file__).resolve().parents[1]
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


def build_args(ckpt_dir: str, device: str):
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
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth", map_location="cpu"))
    model = FontDiffuserModelDPM(
        unet=unet, style_encoder=style_encoder, content_encoder=content_encoder
    )
    model.to(args.device)
    model.eval()
    train_scheduler = build_ddpm_scheduler(args=args)
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    return pipe


def to_tensor(img: Image.Image, size):
    tfm = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    return tfm(img.convert("RGB"))[None, :]


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    x = t.detach().cpu().clamp(-1, 1)
    x = (x + 1) / 2
    x = (x.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(x)


def make_compare(style, content, pred, gt, label, size=128):
    imgs = []
    for im in [style, content, pred, gt]:
        imgs.append(im.convert("RGB").resize((size, size), Image.BILINEAR))
    canvas = Image.new("RGB", (size * 4, size + 28), (255, 255, 255))
    for i, im in enumerate(imgs):
        canvas.paste(im, (i * size, 28))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    headers = ["style(CN)", "content", "pred", "GT"]
    for i, h in enumerate(headers):
        draw.text((i * size + 8, 6), h, fill=(0, 0, 0), font=font)
    draw.text((8, size + 8), label, fill=(20, 20, 20), font=font)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["phase0", "phase1a"], default="phase0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt_dir", default=str(FD_ROOT / "ckpt"))
    ap.add_argument("--max_fonts", type=int, default=None)
    args_cli = ap.parse_args()

    data = ROOT / "data" / "canonical"
    meta = json.loads((data / "meta.json").read_text())
    out_root = ROOT / "runs" / "fontdiffuser" / args_cli.phase
    pred_dir = out_root / "pred"
    cmp_dir = out_root / "compare"
    pred_dir.mkdir(parents=True, exist_ok=True)
    cmp_dir.mkdir(parents=True, exist_ok=True)

    if args_cli.phase == "phase0":
        fonts = meta["demo_fonts"][:1]
        chars = meta["phase0_chars"]
    else:
        fonts = meta["demo_fonts"] if args_cli.max_fonts is None else meta["demo_fonts"][: args_cli.max_fonts]
        # quick subset for overnight report
        chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("agqrs")

    style_char = meta["ref16"][0]  # 永 for 1-shot
    style_uname = f"U+{ord(style_char):04X}"

    args = build_args(args_cli.ckpt_dir, args_cli.device)
    set_seed(args.seed)
    print(f"Loading FontDiffuser on {args.device} ...")
    pipe = load_pipe(args)
    print("Model loaded.")

    records = []
    t0 = time.time()
    for fi, font_file in enumerate(fonts):
        stem = Path(font_file).stem
        style_path = data / "style_cn" / stem / f"{style_uname}.png"
        style_img = Image.open(style_path).convert("RGB")
        style_t = to_tensor(style_img, args.style_image_size).to(args.device)

        for ch in chars:
            def resolve(ch, folder):
                candidates = [
                    folder / f"{ch}.png",
                    folder / f"u{ord(ch):04X}.png",
                    folder / f"U+{ord(ch):04X}.png",
                    folder / f"U+{ord(ch):04X}_{ch}.png",
                ]
                for c in candidates:
                    if c.exists():
                        return c
                return None
            content_path = resolve(ch, data / "content_latin")
            gt_path = resolve(ch, data / "gt_latin" / stem)
            if content_path is None or gt_path is None:
                print("skip missing", stem, ch, content_path, gt_path)
                continue
            tag = ch if ch.isalnum() else f"u{ord(ch):04X}"
            content_img = Image.open(content_path).convert("RGB")
            content_t = to_tensor(content_img, args.content_image_size).to(args.device)
            with torch.no_grad():
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
            if not isinstance(pred, Image.Image):
                pred = tensor_to_pil(pred)
            else:
                pred = pred.convert("RGB")
            pred_path = pred_dir / f"{stem}__{tag}.png"
            pred.save(pred_path)
            gt = Image.open(gt_path).convert("RGB")
            cmp = make_compare(style_img, content_img, pred, gt, f"{stem} | char={ch}")
            cmp.save(cmp_dir / f"{stem}__{tag}.png")
            records.append({"font": stem, "char": ch, "pred": str(pred_path)})
            print(f"[{fi+1}/{len(fonts)}] {stem} {ch} ok")

    elapsed = time.time() - t0
    summary = {
        "phase": args_cli.phase,
        "n": len(records),
        "fonts": fonts,
        "chars": chars,
        "style_ref": style_char,
        "elapsed_sec": elapsed,
        "device": args_cli.device,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE", summary)


if __name__ == "__main__":
    main()
