#!/usr/bin/env python3
"""Rebuild mentor section-2 visuals: method grid @70k + timeline.

Default GPU: CUDA_VISIBLE_DEVICES=3. Does not stop training.
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
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

ROOT = Path("/root/projects/hrfont")
BANK = ROOT / "data/hrfont/e0_bank"
FT_DIR = ROOT / "runs/ft_cnstyle/global_step_25000"
STAGEA = ROOT / "runs/e2_stageA96"
OUT = ROOT / "reports/hrfont_overnight/mentor_preview"

REF8 = list("永和书风骨韵天地")
CHARS = list("AaOoRg8S")
TIMELINE_CHARS = ["a", "g", "R", "8"]
SIZE = 96
CELL = 56
TAU = 0.07
TOP_M = 3
SAMPLE_STEPS = 25
SEED = 42
BEST_STEP = 70000
EARLY_STEP = 10000


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
    x = t.detach().float().cpu()
    if x.dim() == 4:
        x = x[0]
    x = (x.clamp(-1, 1) + 1) * 0.5
    arr = (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def load_pt(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def short_font(name: str) -> str:
    return name.replace("FZ", "")[:16]


def ui_font(size: int = 12):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


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


def build_encoders(args, device):
    from src import build_content_encoder, build_style_encoder

    se = build_style_encoder(args=args)
    ce = build_content_encoder(args=args)
    se.load_state_dict(load_pt(FT_DIR / "style_encoder.pth"))
    ce.load_state_dict(load_pt(FT_DIR / "content_encoder.pth"))
    return se.to(device).eval(), ce.to(device).eval()


def build_unet(args, device, state):
    from src import build_unet as _build

    unet = _build(args=args)
    unet.load_state_dict(state)
    return unet.to(device).eval()


@torch.no_grad()
def style_proto(se, font: str, device) -> torch.Tensor:
    feats = []
    for r in REF8:
        p = png(font, r)
        if not p.exists():
            continue
        feat, _, _ = se(load_im(p).unsqueeze(0).to(device))
        feats.append(feat.flatten(1).mean(0).cpu())
    if not feats:
        raise RuntimeError(f"no style refs for {font}")
    return torch.stack(feats).mean(0)


def alpha_mix(proto_q: torch.Tensor, bank_proto: dict, candidates: list[str], ch: str) -> torch.Tensor:
    scores = []
    for g in candidates:
        if g not in bank_proto or not png(g, ch).exists():
            continue
        v = bank_proto[g].float().flatten()
        scores.append((g, float(F.cosine_similarity(proto_q.flatten()[None], v[None]).item())))
    scores.sort(key=lambda x: -x[1])
    top = scores[:TOP_M]
    if not top:
        raise RuntimeError(f"empty alpha for {ch!r}")
    w = torch.softmax(torch.tensor([s / TAU for _, s in top]), dim=0)
    mix = None
    for i, (g, _) in enumerate(top):
        im = load_im(png(g, ch))
        mix = im * float(w[i]) if mix is None else mix + im * float(w[i])
    return mix


@torch.no_grad()
def sample(unet, se, ce, content, style, struct_img, args, device, use_delta: bool, generator: torch.Generator):
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
    sched.set_timesteps(SAMPLE_STEPS, device=device)
    x = torch.randn(content.shape, device=device, generator=generator)
    for t in sched.timesteps:
        out = unet(
            x,
            t.expand(x.shape[0]),
            encoder_hidden_states=hidden,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
        )
        x = sched.step(out[0], t, x).prev_sample
    return x


def paste_glyph(canvas: Image.Image, img: Image.Image, x: int, y: int, border=None) -> None:
    g = img.resize((CELL, CELL), Image.Resampling.BILINEAR)
    if border:
        frame = Image.new("RGB", (CELL + 4, CELL + 4), border)
        frame.paste(g, (2, 2))
        canvas.paste(frame, (x - 2, y - 2))
    else:
        canvas.paste(g, (x, y))


def build_method_panel(font: str, rows: list[dict], font_ui) -> Image.Image:
    pad, label_w, col_w = 8, 28, CELL + 10
    header_h, row_h = 36, CELL + 18
    w = pad * 2 + label_w + 3 * col_w
    h = header_h + len(rows) * row_h + pad
    im = Image.new("RGB", (w, h), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    dr.text((pad, 6), short_font(font), fill=(18, 21, 26), font=font_ui)
    for i, lab in enumerate(["GT", "ft", "A@70k"]):
        dr.text((pad + label_w + i * col_w + 6, 20), lab, fill=(92, 101, 112), font=font_ui)
    for ri, r in enumerate(rows):
        y = header_h + ri * row_h
        better = r["l1_a"] < r["l1_ft"]
        border = (15, 122, 74) if better else (180, 83, 9)
        dr.text((pad + 4, y + CELL // 2 - 6), r["ch"], fill=(18, 21, 26), font=font_ui)
        paste_glyph(im, r["gt"], pad + label_w, y)
        paste_glyph(im, r["ft"], pad + label_w + col_w, y)
        paste_glyph(im, r["a"], pad + label_w + 2 * col_w, y, border=border)
        dr.text((pad + label_w + col_w, y + CELL + 2), f"{r['l1_ft']:.2f}", fill=(180, 83, 9), font=font_ui)
        dr.text((pad + label_w + 2 * col_w, y + CELL + 2), f"{r['l1_a']:.2f}", fill=border, font=font_ui)
    return im


def build_timeline(font: str, rows: list[dict], steps: list[int], font_ui) -> Image.Image:
    pad, label_w, col_w = 8, 28, CELL + 12
    cols = ["GT"] + [f"A@{s // 1000}k" for s in steps]
    header_h, row_h = 40, CELL + 18
    w = pad * 2 + label_w + len(cols) * col_w
    h = header_h + len(rows) * row_h + pad
    im = Image.new("RGB", (w, h), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    dr.text((pad, 4), f"{short_font(font)} · train progress (Stage A only)", fill=(18, 21, 26), font=font_ui)
    for i, lab in enumerate(cols):
        col = (15, 122, 74) if "70" in lab else (92, 101, 112)
        dr.text((pad + label_w + i * col_w + 2, 22), lab, fill=col, font=font_ui)
    for ri, r in enumerate(rows):
        y = header_h + ri * row_h
        dr.text((pad + 4, y + CELL // 2 - 6), r["ch"], fill=(18, 21, 26), font=font_ui)
        paste_glyph(im, r["gt"], pad + label_w, y)
        for si, step in enumerate(steps):
            border = (15, 122, 74) if step == BEST_STEP else None
            paste_glyph(im, r["preds"][step], pad + label_w + (si + 1) * col_w, y, border=border)
            dr.text(
                (pad + label_w + (si + 1) * col_w, y + CELL + 2),
                f"{r['l1'][step]:.2f}",
                fill=(15, 122, 74) if step == BEST_STEP else (92, 101, 112),
                font=font_ui,
            )
    return im


def build_callouts(cells: list[dict], font_ui) -> Image.Image:
    ranked = sorted(cells, key=lambda c: c["l1_ft"] - c["l1_a"], reverse=True)
    picks = ranked[:3] + ranked[-2:]
    tags = ["win", "win", "win", "lose", "lose"]
    pad = 8
    block = 28 + 3 * (CELL + 8)
    row_h = CELL + 30
    im = Image.new("RGB", (pad * 2 + block + 8, 28 + len(picks) * row_h + pad), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    dr.text((pad, 6), "callouts: largest gaps (green=A better)", fill=(18, 21, 26), font=font_ui)
    for i, (c, tag) in enumerate(zip(picks, tags)):
        y = 28 + i * row_h
        delta = c["l1_ft"] - c["l1_a"]
        col = (15, 122, 74) if delta > 0 else (180, 83, 9)
        dr.text((pad, y), f"{tag} · {short_font(c['font'])} · {c['ch']} · dL1={delta:+.2f}", fill=col, font=font_ui)
        yy = y + 14
        paste_glyph(im, c["gt"], pad + 28, yy)
        paste_glyph(im, c["ft"], pad + 28 + CELL + 8, yy)
        paste_glyph(im, c["a"], pad + 28 + 2 * (CELL + 8), yy, border=col)
        for j, lab in enumerate(["GT", "ft", "A"]):
            dr.text((pad + 28 + j * (CELL + 8), yy + CELL + 1), lab, fill=(92, 101, 112), font=font_ui)
    return im


def ckpt_path(step: int, latest_step: int) -> Path:
    named = STAGEA / f"step_{step}.pt"
    if named.exists():
        return named
    if step == latest_step and (STAGEA / "last.pt").exists():
        return STAGEA / "last.pt"
    raise FileNotFoundError(f"missing ckpt for step {step}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    font_ui = ui_font(11)

    meta = json.loads((BANK / "meta.json").read_text())
    demo = list(meta["demo8"])
    train = [f for f in meta["train_fonts"] if f not in set(demo)]
    bank_proto = load_pt(BANK / "cache/ec_es_r96.pt")["style_proto"]
    fonts = [f for f in demo if sum(png(f, ch).exists() for ch in CHARS) >= 6][:4]

    last_blob = load_pt(STAGEA / "last.pt")
    latest_step = int(last_blob["step"])
    timeline_steps = [EARLY_STEP, BEST_STEP, latest_step]
    log(f"fonts={fonts} timeline={timeline_steps} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    sys.path.insert(0, str(ROOT / "code/ours/FontDiffuser"))
    args = make_args()
    se, ce = build_encoders(args, device)

    jobs = []
    for font in fonts:
        proto = style_proto(se, font, device)
        style_ch = next(r for r in REF8 if png(font, r).exists())
        style = load_im(png(font, style_ch))
        others = [g for g in train if g != font]
        for ch in CHARS:
            if not png(font, ch).exists() or not png("_B0_", ch).exists():
                continue
            jobs.append(
                {
                    "font": font,
                    "ch": ch,
                    "content": load_im(png("_B0_", ch)),
                    "style": style,
                    "gt": load_im(png(font, ch)),
                    "mix": alpha_mix(proto, bank_proto, others, ch),
                }
            )
    log(f"jobs={len(jobs)}")
    job_ix = {(j["font"], j["ch"]): j for j in jobs}

    unet_ft = build_unet(args, device, load_pt(FT_DIR / "unet.pth"))
    ft_preds = {}
    for j, job in enumerate(jobs):
        g = torch.Generator(device=device)
        g.manual_seed(SEED + j * 17)
        pred = sample(
            unet_ft,
            se,
            ce,
            job["content"].unsqueeze(0).to(device),
            job["style"].unsqueeze(0).to(device),
            job["style"].unsqueeze(0).to(device),
            args,
            device,
            False,
            g,
        )
        ft_preds[(job["font"], job["ch"])] = pred.cpu()
        job["l1_ft"] = float((pred.cpu() - job["gt"]).abs().mean())
    del unet_ft
    torch.cuda.empty_cache()
    log(f"ft mean L1={float(np.mean([j['l1_ft'] for j in jobs])):.4f}")

    preds_by_step = {}
    for step in sorted(set(timeline_steps)):
        path = ckpt_path(step, latest_step)
        blob = load_pt(path)
        unet = build_unet(args, device, blob["unet"])
        store = {}
        for j, job in enumerate(jobs):
            g = torch.Generator(device=device)
            g.manual_seed(SEED + j * 17)
            pred = sample(
                unet,
                se,
                ce,
                job["content"].unsqueeze(0).to(device),
                job["style"].unsqueeze(0).to(device),
                job["mix"].unsqueeze(0).to(device),
                args,
                device,
                True,
                g,
            )
            store[(job["font"], job["ch"])] = pred.cpu()
        preds_by_step[step] = store
        del unet
        torch.cuda.empty_cache()
        log(f"sampled StageA @{step} from {path.name}")

    panels = []
    all_cells = []
    for font in fonts:
        rows = []
        for ch in CHARS:
            key = (font, ch)
            if key not in job_ix or key not in ft_preds:
                continue
            job = job_ix[key]
            gt_t = job["gt"]
            ft_t = ft_preds[key]
            a_t = preds_by_step[BEST_STEP][key]
            l1_ft = float((ft_t - gt_t).abs().mean())
            l1_a = float((a_t - gt_t).abs().mean())
            cell = {
                "font": font,
                "ch": ch,
                "gt": to_pil(gt_t),
                "ft": to_pil(ft_t),
                "a": to_pil(a_t),
                "l1_ft": l1_ft,
                "l1_a": l1_a,
            }
            rows.append(cell)
            all_cells.append(cell)
        panels.append(build_method_panel(font, rows, font_ui))

    gap = 12
    method = Image.new(
        "RGB",
        (sum(p.width for p in panels) + gap * (len(panels) - 1), max(p.height for p in panels)),
        (246, 247, 249),
    )
    x = 0
    for p in panels:
        method.paste(p, (x, 0))
        x += p.width + gap
    method_path = OUT / "viz_method_70k.png"
    method.save(method_path)
    log(f"wrote {method_path} {method.size}")

    callouts = build_callouts(all_cells, font_ui)
    callout_path = OUT / "viz_callouts_70k.png"
    callouts.save(callout_path)
    log(f"wrote {callout_path} {callouts.size}")

    vis_font = fonts[0]
    trows = []
    for ch in TIMELINE_CHARS:
        key = (vis_font, ch)
        job = job_ix[key]
        trows.append(
            {
                "ch": ch,
                "gt": to_pil(job["gt"]),
                "preds": {s: to_pil(preds_by_step[s][key]) for s in timeline_steps},
                "l1": {s: float((preds_by_step[s][key] - job["gt"]).abs().mean()) for s in timeline_steps},
            }
        )
    timeline = build_timeline(vis_font, trows, timeline_steps, font_ui)
    timeline_path = OUT / "viz_timeline.png"
    timeline.save(timeline_path)
    log(f"wrote {timeline_path} {timeline.size}")

    meta_out = {
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fonts": fonts,
        "chars": CHARS,
        "best_step": BEST_STEP,
        "timeline_steps": timeline_steps,
        "n_method_cells": len(all_cells),
        "wins_at_70k": sum(1 for c in all_cells if c["l1_a"] < c["l1_ft"]),
        "mean_l1_a_70k": float(np.mean([c["l1_a"] for c in all_cells])),
        "mean_l1_ft": float(np.mean([c["l1_ft"] for c in all_cells])),
        "files": {
            "method": method_path.name,
            "callouts": callout_path.name,
            "timeline": timeline_path.name,
        },
    }
    (OUT / "viz_meta.json").write_text(json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"done {meta_out}")


if __name__ == "__main__":
    main()
