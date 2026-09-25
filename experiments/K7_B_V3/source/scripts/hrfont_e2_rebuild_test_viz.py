#!/usr/bin/env python3
"""Rebuild mentor test visuals: full Demo-8 @ best ckpt + hard cases + step compare.

Outputs under reports/hrfont_overnight/mentor_preview/viz_test/
Default GPU: CUDA_VISIBLE_DEVICES=3
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
DEJAVU = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FT = ROOT / "runs/ft_cnstyle/global_step_25000"
STAGEA = ROOT / "runs/e2_stageA96"
OUT = ROOT / "reports/hrfont_overnight/mentor_preview"
VIZ = OUT / "viz_test"
REF8 = list("永和书风骨韵天地")
CHARS = list("AaOoRg8S")
SIZE = 96
CELL = 52
TAU = 0.07
TOP_M = 3
SAMPLE_STEPS = 25
SEED = 42
COMPARE_STEPS = [70000, 100000, 145500]
HARDCASE_FONTS = ["FZHuoYYJW-T", "FZFengYKSJ", "FZJingYLLTJW", "FZHanWZKJW"]
HARDCASE_CHARS = ["8", "R", "O", "g"]
STEP_DEMO_FONTS = ["FZChuangHJW_DB", "FZHuoYYJW-T", "FZJingYLLTJW", "FZLingFKSJW-B"]
STEP_DEMO_CHARS = ["a", "8", "R", "O"]


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


def render_dejavu_content(ch: str) -> torch.Tensor:
    """ft_cnstyle native content: DejaVu @96 (matches retrain_v2_eval_fontdiffuser)."""
    tfm = transforms.Compose(
        [
            transforms.Resize((SIZE, SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    img = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    lo, hi, best = 8, SIZE, None
    for _ in range(14):
        fs = (lo + hi) // 2
        font = ImageFont.truetype(str(DEJAVU), fs)
        bbox = draw.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= SIZE - 8 and h <= SIZE - 8:
            best = (font, bbox)
            lo = fs + 1
        else:
            hi = fs - 1
    if best is None:
        font = ImageFont.truetype(str(DEJAVU), max(10, SIZE // 2))
        bbox = draw.textbbox((0, 0), ch, font=font)
    else:
        font, bbox = best
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((SIZE - w) // 2 - bbox[0], (SIZE - h) // 2 - bbox[1]),
        ch,
        font=font,
        fill=(0, 0, 0),
    )
    return tfm(img)


def to_pil(t: torch.Tensor) -> Image.Image:
    x = t.detach().float().cpu()
    if x.dim() == 4:
        x = x[0]
    x = (x.clamp(-1, 1) + 1) * 0.5
    return Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def _load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def ui_font(size: int = 11):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def short_font(name: str) -> str:
    return name.replace("FZ", "").replace("JW", "").replace("_", "")[:14]


def best_step() -> int:
    curves = OUT / "train_val_test_curves.json"
    if curves.exists():
        d = json.loads(curves.read_text())
        if d.get("best_test"):
            return int(d["best_test"]["step"])
    ext = OUT / "extended_demo8_eval.json"
    if ext.exists():
        results = json.loads(ext.read_text()).get("results", {})
        if results:
            best_k = min(results, key=lambda k: results[k]["mean_l1_a"])
            return int(best_k)
    last = STAGEA / "last.pt"
    if last.exists():
        return int(_load(last).get("step", 145500))
    return 145500


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
    se.load_state_dict(_load(FT / "style_encoder.pth"))
    ce.load_state_dict(_load(FT / "content_encoder.pth"))
    se.to(device).eval()
    ce.to(device).eval()
    return se, ce


def build_unet(args, device, state):
    from src import build_unet

    unet = build_unet(args=args)
    unet.load_state_dict(state)
    unet.to(device).eval()
    return unet


@torch.no_grad()
def style_proto(se, font: str, device) -> torch.Tensor:
    feats = []
    for r in REF8:
        p = png(font, r)
        if p.exists():
            f, _, _ = se(load_im(p).unsqueeze(0).to(device))
            feats.append(f.flatten(1).mean(0).cpu())
    return torch.stack(feats).mean(0)


def alpha_mix(proto_q, bank_proto, train_fonts, ch):
    scores = []
    for g in train_fonts:
        if g not in bank_proto or not png(g, ch).exists():
            continue
        v = bank_proto[g].float().flatten()
        scores.append((g, float(F.cosine_similarity(proto_q.flatten()[None], v[None]).item())))
    scores.sort(key=lambda x: -x[1])
    top = scores[:TOP_M]
    w = torch.softmax(torch.tensor([s / TAU for _, s in top]), dim=0)
    mix = None
    for i, (g, _) in enumerate(top):
        im = load_im(png(g, ch))
        mix = im * float(w[i]) if mix is None else mix + im * float(w[i])
    return mix


@torch.no_grad()
def sample(unet, se, ce, content, style, struct_img, args, device, use_delta, generator):
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


def resolve_ckpt(step: int) -> tuple[int, Path]:
    exact = STAGEA / f"step_{step}.pt"
    if exact.exists():
        return step, exact
    cands = sorted(STAGEA.glob("step_*.pt"), key=lambda q: abs(int(q.stem.split("_")[1]) - step))
    if cands:
        nearest = cands[0]
        ns = int(nearest.stem.split("_")[1])
        if abs(ns - step) <= 5000:
            return ns, nearest
    last = STAGEA / "last.pt"
    if last.exists():
        blob = _load(last)
        ls = int(blob.get("step", 0))
        if abs(ls - step) <= 5000:
            return ls, last
    raise FileNotFoundError(f"no ckpt within 5k of step {step}")


def paste_cell(canvas, pil: Image.Image, x: int, y: int, border: tuple | None = None):
    g = pil.resize((CELL, CELL))
    if border:
        fr = Image.new("RGB", (CELL + 4, CELL + 4), border)
        fr.paste(g, (2, 2))
        canvas.paste(fr, (x - 2, y - 2))
    else:
        canvas.paste(g, (x, y))


def font_panel(
    font: str,
    jobs: list[dict],
    ft_preds: dict,
    ft_dj_preds: dict,
    a_preds: dict,
    title_suffix: str = "",
) -> Image.Image:
    fnt = ui_font(11)
    fnt_b = ui_font(12)
    rows = [j for j in jobs if j["font"] == font]
    cols = CHARS
    pad, lw, gap = 10, 44, 4
    cw = CELL + gap
    hdr = 44
    rh = CELL + 16
    w = pad * 2 + lw + len(cols) * cw
    h = hdr + 4 * rh + 8
    im = Image.new("RGB", (w, h), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    wins = sum(1 for j in rows if j["l1_a"] < j["l1_ft"])
    mean_a = float(np.mean([j["l1_a"] for j in rows]))
    mean_dj = float(np.mean([j["l1_ft_dj"] for j in rows]))
    dr.rectangle([0, 0, w, hdr - 4], fill=(246, 247, 249))
    dr.text((pad, 8), f"{short_font(font)}{title_suffix}", fill=(18, 21, 26), font=fnt_b)
    dr.text(
        (pad, 24),
        f"胜 {wins}/{len(rows)} · A {mean_a:.3f} · ft·DJ {mean_dj:.3f}",
        fill=(92, 101, 112),
        font=fnt,
    )
    labels = ["GT", "ft·B0", "ft·DJ", "A"]
    colors = [(15, 122, 74), (180, 83, 9), (120, 72, 160), (31, 111, 143)]
    for ci, ch in enumerate(cols):
        dr.text((pad + lw + ci * cw + 8, hdr - 14), ch, fill=(18, 21, 26), font=fnt_b)
    for ri, (lab, col) in enumerate(zip(labels, colors)):
        y = hdr + ri * rh
        dr.text((pad + 2, y + CELL // 2 - 4), lab, fill=col, font=fnt)
        for ci, ch in enumerate(cols):
            job = next((j for j in rows if j["ch"] == ch), None)
            if not job:
                continue
            key = (font, ch)
            x = pad + lw + ci * cw
            if lab == "GT":
                paste_cell(im, to_pil(job["gt"]), x, y)
            elif lab == "ft·B0":
                paste_cell(im, to_pil(ft_preds[key]), x, y)
                dr.text((x, y + CELL + 2), f"{job['l1_ft']:.2f}", fill=(180, 83, 9), font=fnt)
            elif lab == "ft·DJ":
                paste_cell(im, to_pil(ft_dj_preds[key]), x, y)
                dr.text((x, y + CELL + 2), f"{job['l1_ft_dj']:.2f}", fill=(120, 72, 160), font=fnt)
            else:
                better = job["l1_a"] < job["l1_ft"]
                br = (15, 122, 74) if better else (180, 83, 9)
                paste_cell(im, to_pil(a_preds[key]), x, y, border=br)
                dr.text((x, y + CELL + 2), f"{job['l1_a']:.2f}", fill=br, font=fnt)
    return im


def hardcase_panel(jobs, ft_preds, ft_dj_preds, a_preds, fonts, chars) -> Image.Image:
    fnt = ui_font(10)
    fnt_b = ui_font(11)
    pad, gap = 12, 6
    col_w = 5 * (CELL + gap) + 40
    row_h = CELL + 28
    hdr = 36
    w = pad * 2 + col_w * len(chars)
    h = pad * 2 + hdr + row_h * len(fonts)
    im = Image.new("RGB", (w, h), (250, 251, 252))
    dr = ImageDraw.Draw(im)
    dr.text((pad, pad), "难例放大 · GT | ft·B0 | ft·DJ | A@best | ΔL1", fill=(18, 21, 26), font=fnt_b)
    for ci, ch in enumerate(chars):
        x0 = pad + ci * col_w + 20
        dr.text((x0 + 2 * CELL, pad + 18), ch, fill=(92, 101, 112), font=fnt_b)
        for li, lab in enumerate(["GT", "B0", "DJ", "A"]):
            dr.text((x0 + li * (CELL + gap), pad + hdr - 12), lab, fill=(120, 120, 120), font=fnt)
    for ri, font in enumerate(fonts):
        y = pad + hdr + ri * row_h
        dr.text((4, y + CELL // 2 - 4), short_font(font)[:10], fill=(18, 21, 26), font=fnt)
        for ci, ch in enumerate(chars):
            job = next((j for j in jobs if j["font"] == font and j["ch"] == ch), None)
            if not job:
                continue
            key = (font, ch)
            x0 = pad + ci * col_w + 20
            paste_cell(im, to_pil(job["gt"]), x0, y)
            paste_cell(im, to_pil(ft_preds[key]), x0 + CELL + gap, y)
            paste_cell(im, to_pil(ft_dj_preds[key]), x0 + 2 * (CELL + gap), y)
            br = (15, 122, 74) if job["l1_a"] < job["l1_ft"] else (180, 83, 9)
            paste_cell(im, to_pil(a_preds[key]), x0 + 3 * (CELL + gap), y, border=br)
            delta = job["l1_ft"] - job["l1_a"]
            dr.text(
                (x0 + 4 * (CELL + gap), y + CELL // 2 - 4),
                f"{delta:+.2f}",
                fill=br,
                font=fnt_b,
            )
    return im


def step_compare_panel(jobs, ft_preds, step_preds: dict[int, dict], fonts, chars) -> Image.Image:
    fnt = ui_font(10)
    fnt_b = ui_font(11)
    steps = sorted(step_preds)
    pad, gap = 12, 5
    ncols = 1 + 1 + len(steps)  # GT + ft + A steps
    col_w = ncols * (CELL + gap) + 36
    row_h = CELL + 22
    hdr = 40
    w = pad * 2 + col_w * len(chars)
    h = pad * 2 + hdr + row_h * len(fonts)
    im = Image.new("RGB", (w, h), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    dr.text((pad, pad), "训练步数对比 · 同 seed", fill=(18, 21, 26), font=fnt_b)
    for ci, ch in enumerate(chars):
        x0 = pad + ci * col_w + 28
        dr.text((x0 + 2 * CELL, pad + 18), ch, fill=(92, 101, 112), font=fnt_b)
        hdrs = ["GT", "ft"] + [f"A@{s // 1000}k" for s in steps]
        for li, lab in enumerate(hdrs):
            dr.text((x0 + li * (CELL + gap), pad + hdr - 12), lab, fill=(120, 120, 120), font=fnt)
    for ri, font in enumerate(fonts):
        y = pad + hdr + ri * row_h
        dr.text((4, y + CELL // 2 - 4), short_font(font)[:10], fill=(18, 21, 26), font=fnt)
        for ci, ch in enumerate(chars):
            job = next((j for j in jobs if j["font"] == font and j["ch"] == ch), None)
            if not job:
                continue
            key = (font, ch)
            x0 = pad + ci * col_w + 28
            paste_cell(im, to_pil(job["gt"]), x0, y)
            paste_cell(im, to_pil(ft_preds[key]), x0 + CELL + gap, y)
            for si, step in enumerate(steps):
                x = x0 + (2 + si) * (CELL + gap)
                paste_cell(im, to_pil(step_preds[step][key]), x, y)
    return im


def main() -> None:
    VIZ.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    best = best_step()
    log(f"best_step={best} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    meta = json.loads((BANK / "meta.json").read_text())
    demo8 = list(meta["demo8"])
    train = [f for f in meta["train_fonts"] if f not in set(demo8)]
    bank_proto = _load(BANK / "cache/ec_es_r96.pt")["style_proto"]
    fonts = [f for f in demo8 if any(png(f, r).exists() for r in REF8)]
    log(f"demo8={len(fonts)} fonts")

    sys.path.insert(0, str(ROOT / "code/ours/FontDiffuser"))
    args = make_args()
    se, ce = build_encoders(args, device)

    jobs = []
    for font in fonts:
        proto = style_proto(se, font, device)
        style_ch = next(r for r in REF8 if png(font, r).exists())
        style = load_im(png(font, style_ch))
        for ch in CHARS:
            if not png(font, ch).exists() or not png("_B0_", ch).exists():
                continue
            jobs.append({
                "font": font,
                "ch": ch,
                "content": load_im(png("_B0_", ch)),
                "gt": load_im(png(font, ch)),
                "style": style,
                "mix": alpha_mix(proto, bank_proto, train, ch),
            })
    log(f"jobs={len(jobs)}")

    unet_ft = build_unet(args, device, _load(FT / "unet.pth"))
    ft_preds = {}
    ft_dj_preds = {}
    for j, job in enumerate(jobs):
        g = torch.Generator(device=device)
        g.manual_seed(SEED + j * 17)
        pred = sample(
            unet_ft, se, ce,
            job["content"].unsqueeze(0).to(device),
            job["style"].unsqueeze(0).to(device),
            job["style"].unsqueeze(0).to(device),
            args, device, False, g,
        )
        ft_preds[(job["font"], job["ch"])] = pred.cpu()
        job["l1_ft"] = float((pred.cpu() - job["gt"]).abs().mean())

        content_dj = render_dejavu_content(job["ch"]).unsqueeze(0).to(device)
        g_dj = torch.Generator(device=device)
        g_dj.manual_seed(SEED + j * 17)
        pred_dj = sample(
            unet_ft, se, ce,
            content_dj,
            job["style"].unsqueeze(0).to(device),
            job["style"].unsqueeze(0).to(device),
            args, device, False, g_dj,
        )
        ft_dj_preds[(job["font"], job["ch"])] = pred_dj.cpu()
        job["l1_ft_dj"] = float((pred_dj.cpu() - job["gt"]).abs().mean())
    del unet_ft
    torch.cuda.empty_cache()
    log(
        f"ft done mean L1 B0={float(np.mean([j['l1_ft'] for j in jobs])):.4f} "
        f"DJ={float(np.mean([j['l1_ft_dj'] for j in jobs])):.4f}"
    )

    a_preds: dict = {}
    step_preds: dict[int, dict] = {}
    best_actual = best
    for step in sorted(set([best] + COMPARE_STEPS)):
        actual_step, ck = resolve_ckpt(step)
        unet_a = build_unet(args, device, _load(ck)["unet"])
        preds = {}
        for j, job in enumerate(jobs):
            g = torch.Generator(device=device)
            g.manual_seed(SEED + j * 17)
            pred = sample(
                unet_a, se, ce,
                job["content"].unsqueeze(0).to(device),
                job["style"].unsqueeze(0).to(device),
                job["mix"].unsqueeze(0).to(device),
                args, device, True, g,
            )
            preds[(job["font"], job["ch"])] = pred.cpu()
            if step == best:
                job["l1_a"] = float((pred.cpu() - job["gt"]).abs().mean())
        if step == best:
            a_preds = preds
            best_actual = actual_step
        if step in COMPARE_STEPS:
            step_preds[step] = preds
        del unet_a
        torch.cuda.empty_cache()
        log(f"A@{step} (ckpt {actual_step}) done")

    # Per-font panels
    font_imgs = []
    for font in fonts:
        panel = font_panel(font, jobs, ft_preds, ft_dj_preds, a_preds, f" @ {best_actual // 1000}k")
        slug = font.replace("/", "_")
        path = VIZ / f"font_{slug}.png"
        panel.save(path)
        font_imgs.append((font, panel, path.name))
        log(f"saved {path.name} {panel.size}")

    # 2×4 master grid
    gw = max(p.width for _, p, _ in font_imgs)
    gh = max(p.height for _, p, _ in font_imgs)
    gap = 14
    grid = Image.new("RGB", (2 * gw + 3 * gap, 4 * gh + 5 * gap), (246, 247, 249))
    for i, (_, panel, _) in enumerate(font_imgs):
        r, c = divmod(i, 2)
        grid.paste(panel, (gap + c * (gw + gap), gap + r * (gh + gap)))
    grid_path = VIZ / "demo8_grid_2x4.png"
    grid.save(grid_path)
    log(f"saved {grid_path} {grid.size}")

    hard = hardcase_panel(jobs, ft_preds, ft_dj_preds, a_preds, HARDCASE_FONTS, HARDCASE_CHARS)
    hard_path = VIZ / "hardcases.png"
    hard.save(hard_path)

    steps_img = step_compare_panel(jobs, ft_preds, step_preds, STEP_DEMO_FONTS, STEP_DEMO_CHARS)
    steps_path = VIZ / "step_compare.png"
    steps_img.save(steps_path)

    wins = sum(1 for j in jobs if j["l1_a"] < j["l1_ft"])
    cells_sorted = sorted(jobs, key=lambda j: j["l1_ft"] - j["l1_a"], reverse=True)
    callouts = {
        "best_step": best,
        "wins": [
            {
                "font": c["font"],
                "ch": c["ch"],
                "l1_ft": c["l1_ft"],
                "l1_a": c["l1_a"],
                "delta": c["l1_ft"] - c["l1_a"],
            }
            for c in cells_sorted[:6]
        ],
        "losses": [
            {
                "font": c["font"],
                "ch": c["ch"],
                "l1_ft": c["l1_ft"],
                "l1_a": c["l1_a"],
                "delta": c["l1_ft"] - c["l1_a"],
            }
            for c in sorted(jobs, key=lambda j: j["l1_ft"] - j["l1_a"])[:4]
        ],
        "n_cells": len(jobs),
        "wins_n": wins,
    }
    viz_meta = {
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_step": best,
        "best_ckpt_step": best_actual,
        "fonts": fonts,
        "chars": CHARS,
        "mean_l1_ft": float(np.mean([j["l1_ft"] for j in jobs])),
        "mean_l1_ft_dj": float(np.mean([j["l1_ft_dj"] for j in jobs])),
        "mean_l1_a": float(np.mean([j["l1_a"] for j in jobs])),
        "content_ft_b0": "FZKTJW _B0_ (hrfont bank)",
        "content_ft_dj": "DejaVu Sans @96 (ft_cnstyle native)",
        "wins": wins,
        "n": len(jobs),
        "font_images": [name for _, _, name in font_imgs],
        "grid": "viz_test/demo8_grid_2x4.png",
        "hardcases": "viz_test/hardcases.png",
        "step_compare": "viz_test/step_compare.png",
    }
    (OUT / "viz_test_meta.json").write_text(json.dumps(viz_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "viz_callouts.json").write_text(json.dumps(callouts, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"done wins={wins}/{len(jobs)} mean A={viz_meta['mean_l1_a']:.4f}")


if __name__ == "__main__":
    main()
