#!/usr/bin/env python3
"""Mid-run mentor preview: Stage A (Delta-RSI) vs ft_cnstyle (official RSI).

Default GPU: CUDA_VISIBLE_DEVICES=3. Does not stop training.
Writes reports/hrfont_overnight/mentor_preview/.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("HRFONT_PREVIEW_GPU", "3"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms

ROOT = Path("/root/projects/hrfont")
BANK = ROOT / "data/hrfont/e0_bank"
FT = ROOT / "runs/ft_cnstyle/global_step_25000"
STAGEA = ROOT / "runs/e2_stageA96"
OUT = ROOT / "reports/hrfont_overnight/mentor_preview"
REF8 = list("永和书风骨韵天地")
SIZE = 96
TAU = 0.07
TOP_M = 3
CHARS = list("AaOoRg8S")
STEPS = 25


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def png(font: str, ch: str) -> Path:
    return BANK / "r96" / font / f"u{ord(ch):04X}.png"


def load_im(path: Path) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.Resize((SIZE, SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    return tfm(Image.open(path).convert("RGB"))


def to_pil(t: torch.Tensor) -> Image.Image:
    x = (t.detach().float().cpu().clamp(-1, 1) + 1) * 0.5
    arr = (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def _load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def make_args():
    from configs.fontdiffuser import get_parser

    args = get_parser().parse_args([])
    args.resolution = SIZE
    args.content_image_size = (SIZE, SIZE)
    args.style_image_size = (SIZE, SIZE)
    args.unet_channels = (64, 128, 256, 512)
    args.channel_attn = True
    args.content_encoder_downsample_size = 3
    args.content_start_channel = 64
    args.style_start_channel = 64
    args.beta_scheduler = "scaled_linear"
    return args


def load_stack(args, device, unet_state):
    from src import build_content_encoder, build_style_encoder, build_unet

    unet = build_unet(args=args)
    se = build_style_encoder(args=args)
    ce = build_content_encoder(args=args)
    unet.load_state_dict(unet_state)
    se.load_state_dict(_load(FT / "style_encoder.pth"))
    ce.load_state_dict(_load(FT / "content_encoder.pth"))
    unet.to(device).eval()
    se.to(device).eval()
    ce.to(device).eval()
    return unet, se, ce


@torch.no_grad()
def compute_style_proto(se, font: str, device) -> torch.Tensor:
    feats = []
    for r in REF8:
        p = png(font, r)
        if not p.exists():
            continue
        f, _, _ = se(load_im(p).unsqueeze(0).to(device))
        feats.append(f.flatten(1).mean(0).cpu())
    if not feats:
        raise RuntimeError(f"no style refs for {font}")
    return torch.stack(feats).mean(0)


def alpha_mix(proto_q: torch.Tensor, bank_proto: dict, train_fonts: list[str], ch: str):
    scores = []
    for g in train_fonts:
        if g not in bank_proto or not png(g, ch).exists():
            continue
        v = bank_proto[g].float().flatten()
        scores.append((g, float(F.cosine_similarity(proto_q.flatten()[None], v[None]).item())))
    scores.sort(key=lambda x: -x[1])
    top = scores[:TOP_M]
    if not top:
        raise RuntimeError(f"empty alpha bank for {ch!r}")
    logits = torch.tensor([s / TAU for _, s in top])
    w = torch.softmax(logits, dim=0)
    mix = None
    for i, (g, _) in enumerate(top):
        im = load_im(png(g, ch))
        mix = im * float(w[i]) if mix is None else mix + im * float(w[i])
    return mix, [(g, float(w[i])) for i, (g, _) in enumerate(top)]


@torch.no_grad()
def sample(unet, se, ce, content, style, struct_img, args, device, use_delta: bool):
    from src import build_ddpm_scheduler

    style_feat, _, _ = se(style)
    b, c, h, w = style_feat.shape
    style_hidden = style_feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
    content_feat, content_res = ce(content)
    struct_feat, struct_res = ce(struct_img)
    content_res = list(content_res) + [content_feat]
    struct_res = list(struct_res) + [struct_feat]
    if use_delta:
        struct_res = [s - c_ for s, c_ in zip(struct_res, content_res)]
    hidden = [style_feat, content_res, style_hidden, struct_res]

    sched = build_ddpm_scheduler(args)
    sched.set_timesteps(STEPS, device=device)
    x = torch.randn_like(content)
    for t in sched.timesteps:
        out = unet(
            x,
            t.expand(x.shape[0]),
            encoder_hidden_states=hidden,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
        )
        x = sched.step(out[0], t, x).prev_sample
    return x


def tile(img: Image.Image, caption: str) -> Image.Image:
    canvas = Image.new("RGB", (SIZE, SIZE + 16), (248, 248, 248))
    canvas.paste(img.resize((SIZE, SIZE)), (0, 16))
    ImageDraw.Draw(canvas).text((3, 1), caption[:22], fill=(20, 20, 20))
    return canvas


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    meta = json.loads((BANK / "meta.json").read_text())
    demo = list(meta["demo8"])
    train = [f for f in meta["train_fonts"] if f not in set(demo)]
    bank_proto = _load(BANK / "cache/ec_es_r96.pt")["style_proto"]
    gap_by_ch = {r["ch"]: r["gap"] for r in json.loads((BANK / "gap_b0_iou.json").read_text())["rows"]}

    ck_path = STAGEA / "last.pt"
    if not ck_path.exists():
        cks = sorted(STAGEA.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        ck_path = cks[-1]
    blob = _load(ck_path)
    step = int(blob.get("step", 0))
    log(f"ckpt={ck_path.name} step={step} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    sys.path.insert(0, str(ROOT / "code/FontDiffuser"))
    args = make_args()
    unet_a, se, ce = load_stack(args, device, blob["unet"])
    unet_b, se_b, ce_b = load_stack(args, device, _load(FT / "unet.pth"))

    fonts = []
    for f in demo:
        ok = sum(png(f, ch).exists() for ch in CHARS) >= 6 and any(png(f, r).exists() for r in REF8)
        if ok:
            fonts.append(f)
        if len(fonts) >= 4:
            break
    if len(fonts) < 2:
        fonts = [f for f in demo if any(png(f, r).exists() for r in REF8)][:4]
    log(f"fonts={fonts}")

    cells = []
    strips = []
    m_a, m_b = [], []
    by_char = {}

    for font in fonts:
        proto = compute_style_proto(se, font, device)
        style_ch = next(r for r in REF8 if png(font, r).exists())
        style = load_im(png(font, style_ch)).unsqueeze(0).to(device)

        row_cells = []
        for ch in CHARS:
            if not png(font, ch).exists() or not png("_B0_", ch).exists():
                continue
            content = load_im(png("_B0_", ch)).unsqueeze(0).to(device)
            gt = load_im(png(font, ch)).unsqueeze(0).to(device)
            mix, top = alpha_mix(proto, bank_proto, train, ch)
            mix_b = mix.unsqueeze(0).to(device)

            pred_b = sample(unet_b, se_b, ce_b, content, style, style, args, device, use_delta=False)
            pred_a = sample(unet_a, se, ce, content, style, mix_b, args, device, use_delta=True)

            la = float((pred_a - gt).abs().mean())
            lb = float((pred_b - gt).abs().mean())
            m_a.append(la)
            m_b.append(lb)
            by_char.setdefault(ch, {"a": [], "b": [], "gap": gap_by_ch.get(ch)})
            by_char[ch]["a"].append(la)
            by_char[ch]["b"].append(lb)

            cell = Image.new("RGB", (SIZE * 4 + 12, SIZE + 16), (255, 255, 255))
            for i, (im, cap) in enumerate(
                [
                    (to_pil(content[0]), f"C:{ch}"),
                    (to_pil(gt[0]), "GT"),
                    (to_pil(pred_b[0]), f"ft {lb:.3f}"),
                    (to_pil(pred_a[0]), f"A {la:.3f}"),
                ]
            ):
                cell.paste(tile(im, cap), (i * (SIZE + 4), 0))
            row_cells.append(cell)
            cells.append(
                {
                    "font": font,
                    "ch": ch,
                    "gap": gap_by_ch.get(ch),
                    "l1_stageA": la,
                    "l1_ft": lb,
                    "alpha_top": top,
                    "style_ref": style_ch,
                }
            )
            log(f"{font} {ch!r} gap={gap_by_ch.get(ch)} L1 ft={lb:.4f} A={la:.4f}")

        if not row_cells:
            continue
        w = max(c.width for c in row_cells)
        h = sum(c.height for c in row_cells) + 28
        strip = Image.new("RGB", (w, h), (255, 255, 255))
        ImageDraw.Draw(strip).text(
            (6, 6),
            f"{font} | style={style_ch} | C | GT | ft_cnstyle(official RSI) | StageA(Delta-RSI) @ {step}",
            fill=(10, 10, 10),
        )
        y = 26
        for c in row_cells:
            strip.paste(c, (0, y))
            y += c.height
        strip.save(OUT / f"grid_{font}.png")
        strips.append(strip)

    if strips:
        W = max(s.width for s in strips)
        H = sum(s.height for s in strips) + 8 * len(strips)
        master = Image.new("RGB", (W, H), (235, 235, 235))
        y = 0
        for s in strips:
            master.paste(s, (0, y))
            y += s.height + 8
        master.save(OUT / "gallery.png")

    summary = {
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stageA_ckpt": str(ck_path),
        "stageA_step": step,
        "baseline": str(FT),
        "fonts": fonts,
        "chars": CHARS,
        "sample_steps": STEPS,
        "mean_l1_stageA": float(np.mean(m_a)) if m_a else None,
        "mean_l1_ft_cnstyle": float(np.mean(m_b)) if m_b else None,
        "by_char": {
            ch: {
                "gap": v["gap"],
                "mean_l1_stageA": float(np.mean(v["a"])),
                "mean_l1_ft": float(np.mean(v["b"])),
            }
            for ch, v in by_char.items()
        },
        "cells": cells,
        "disclaimer": "Mid-train preview on Demo-8. L1 to same-font Latin GT. Not a paper table; Support (Stage B) not on yet.",
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"mean L1 ft={summary['mean_l1_ft_cnstyle']} A={summary['mean_l1_stageA']}")
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
