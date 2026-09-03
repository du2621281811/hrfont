#!/usr/bin/env python3
"""Multi-checkpoint mid-eval: Stage A (Delta-RSI) vs ft_cnstyle across training.

Default GPU: CUDA_VISIBLE_DEVICES=3. Does not stop training.
Writes reports/hrfont_overnight/mentor_preview/trend/.
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
TREND = OUT / "trend"
REF8 = list("永和书风骨韵天地")
SIZE = 96
TAU = 0.07
TOP_M = 3
CHARS = list("AaOoRg8S")
SAMPLE_STEPS = 25
SEED = 42
SWEEP_STEPS = [10000, 25000, 40000, 55000, 70000, 85000, 100000, 115000]


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


def build_encoders(args, device):
    from src import build_content_encoder, build_style_encoder

    se = build_style_encoder(args=args)
    ce = build_content_encoder(args=args)
    se.load_state_dict(_load(FT / "style_encoder.pth"))
    ce.load_state_dict(_load(FT / "content_encoder.pth"))
    se.to(device).eval()
    ce.to(device).eval()
    return se, ce


def build_unet_only(args, device, state):
    from src import build_unet

    unet = build_unet(args=args)
    unet.load_state_dict(state)
    unet.to(device).eval()
    return unet


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
        raise RuntimeError(f"empty alpha for {ch!r}")
    logits = torch.tensor([s / TAU for _, s in top])
    w = torch.softmax(logits, dim=0)
    mix = None
    for i, (g, _) in enumerate(top):
        im = load_im(png(g, ch))
        mix = im * float(w[i]) if mix is None else mix + im * float(w[i])
    return mix, [(g, float(w[i])) for i, (g, _) in enumerate(top)]


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


def tile(img: Image.Image, caption: str, w: int = 72) -> Image.Image:
    canvas = Image.new("RGB", (w, w + 14), (250, 250, 250))
    canvas.paste(img.resize((w, w)), (0, 14))
    ImageDraw.Draw(canvas).text((2, 1), caption[:16], fill=(30, 30, 30))
    return canvas


def resolve_ckpts(wanted: list[int]) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for s in wanted:
        p = STAGEA / f"step_{s}.pt"
        if p.exists():
            out.append((s, p))
            continue
        cands = sorted(STAGEA.glob("step_*.pt"), key=lambda q: abs(int(q.stem.split("_")[1]) - s))
        if not cands:
            continue
        best = cands[0]
        bs = int(best.stem.split("_")[1])
        if abs(bs - s) <= 5000:
            out.append((bs, best))
    last = STAGEA / "last.pt"
    if last.exists():
        blob_step = int(_load(last).get("step", 0))
        if blob_step:
            named = STAGEA / f"step_{blob_step}.pt"
            out.append((blob_step, named if named.exists() else last))
    seen = set()
    uniq = []
    for s, p in sorted(out, key=lambda x: x[0]):
        if s in seen:
            continue
        seen.add(s)
        uniq.append((s, p))
    return uniq


def main() -> None:
    TREND.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    meta = json.loads((BANK / "meta.json").read_text())
    demo = list(meta["demo8"])
    train = [f for f in meta["train_fonts"] if f not in set(demo)]
    bank_proto = _load(BANK / "cache/ec_es_r96.pt")["style_proto"]
    gap_by_ch = {r["ch"]: r["gap"] for r in json.loads((BANK / "gap_b0_iou.json").read_text())["rows"]}

    fonts = []
    for f in demo:
        ok = sum(png(f, ch).exists() for ch in CHARS) >= 6 and any(png(f, r).exists() for r in REF8)
        if ok:
            fonts.append(f)
        if len(fonts) >= 4:
            break
    if len(fonts) < 2:
        fonts = [f for f in demo if any(png(f, r).exists() for r in REF8)][:4]
    log(f"fonts={fonts} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    ckpts = resolve_ckpts(SWEEP_STEPS)
    log(f"ckpts={[s for s, _ in ckpts]}")

    sys.path.insert(0, str(ROOT / "code/ours/FontDiffuser"))
    args = make_args()
    se, ce = build_encoders(args, device)

    jobs = []
    for font in fonts:
        proto = compute_style_proto(se, font, device)
        style_ch = next(r for r in REF8 if png(font, r).exists())
        style = load_im(png(font, style_ch))
        for ch in CHARS:
            if not png(font, ch).exists() or not png("_B0_", ch).exists():
                continue
            mix, top = alpha_mix(proto, bank_proto, train, ch)
            jobs.append(
                {
                    "font": font,
                    "ch": ch,
                    "gap": gap_by_ch.get(ch),
                    "style_ch": style_ch,
                    "content": load_im(png("_B0_", ch)),
                    "gt": load_im(png(font, ch)),
                    "style": style,
                    "mix": mix,
                    "alpha_top": top,
                }
            )
    log(f"jobs={len(jobs)}")

    unet_ft = build_unet_only(args, device, _load(FT / "unet.pth"))
    ft_preds = {}
    for j, job in enumerate(jobs):
        g = torch.Generator(device=device)
        g.manual_seed(SEED + j * 17)
        content = job["content"].unsqueeze(0).to(device)
        style = job["style"].unsqueeze(0).to(device)
        pred = sample(unet_ft, se, ce, content, style, style, args, device, False, g)
        ft_preds[(job["font"], job["ch"])] = pred.cpu()
        job["l1_ft"] = float((pred.cpu() - job["gt"]).abs().mean())
    mean_ft = float(np.mean([j["l1_ft"] for j in jobs]))
    log(f"baseline ft mean L1={mean_ft:.4f}")
    del unet_ft
    torch.cuda.empty_cache()

    series = []
    vis_font = fonts[0]
    vis_chars = ["a", "g", "R", "8"]

    for step, ck_path in ckpts:
        t0 = time.time()
        blob = _load(ck_path)
        unet_a = build_unet_only(args, device, blob["unet"])
        m_a = []
        by_char = {}
        cells = []
        vis_tiles = []

        for j, job in enumerate(jobs):
            g = torch.Generator(device=device)
            g.manual_seed(SEED + j * 17)
            content = job["content"].unsqueeze(0).to(device)
            style = job["style"].unsqueeze(0).to(device)
            mix = job["mix"].unsqueeze(0).to(device)
            pred_a = sample(unet_a, se, ce, content, style, mix, args, device, True, g)
            pred_ft = ft_preds[(job["font"], job["ch"])]
            la = float((pred_a.cpu() - job["gt"]).abs().mean())
            lb = job["l1_ft"]
            m_a.append(la)
            by_char.setdefault(job["ch"], {"a": [], "b": [], "gap": job["gap"]})
            by_char[job["ch"]]["a"].append(la)
            by_char[job["ch"]]["b"].append(lb)
            cells.append({"font": job["font"], "ch": job["ch"], "gap": job["gap"], "l1_stageA": la, "l1_ft": lb})
            if job["font"] == vis_font and job["ch"] in vis_chars:
                cell = Image.new("RGB", (72 * 4 + 9, 72 + 14), (255, 255, 255))
                for i, (im, cap) in enumerate(
                    [
                        (to_pil(job["content"]), f"C:{job['ch']}"),
                        (to_pil(job["gt"]), "GT"),
                        (to_pil(pred_ft), f"ft{lb:.2f}"),
                        (to_pil(pred_a[0]), f"A{la:.2f}"),
                    ]
                ):
                    cell.paste(tile(im, cap, 72), (i * (72 + 3), 0))
                vis_tiles.append((job["ch"], cell))

        mean_a = float(np.mean(m_a))
        wins = sum(1 for c in cells if c["l1_stageA"] < c["l1_ft"])
        rec = {
            "step": step,
            "ckpt": str(ck_path),
            "mean_l1_stageA": mean_a,
            "mean_l1_ft_cnstyle": mean_ft,
            "wins": wins,
            "n": len(cells),
            "win_rate": wins / max(len(cells), 1),
            "by_char": {
                ch: {
                    "gap": v["gap"],
                    "mean_l1_stageA": float(np.mean(v["a"])),
                    "mean_l1_ft": float(np.mean(v["b"])),
                }
                for ch, v in by_char.items()
            },
            "cells": cells,
            "sec": round(time.time() - t0, 1),
        }
        series.append(rec)
        (TREND / f"metrics_step_{step}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")

        if vis_tiles:
            vis_tiles.sort(key=lambda x: vis_chars.index(x[0]) if x[0] in vis_chars else 99)
            ww = max(t.width for _, t in vis_tiles)
            hh = sum(t.height for _, t in vis_tiles) + 22
            strip = Image.new("RGB", (ww, hh), (255, 255, 255))
            ImageDraw.Draw(strip).text((4, 4), f"step {step} | {vis_font} | C|GT|ft|A", fill=(10, 10, 10))
            y = 20
            for _, t in vis_tiles:
                strip.paste(t, (0, y))
                y += t.height
            strip.save(TREND / f"strip_step_{step}.png")

        log(f"step={step} L1 A={mean_a:.4f} ft={mean_ft:.4f} wins={wins}/{len(cells)} dt={time.time()-t0:.0f}s")
        del unet_a
        torch.cuda.empty_cache()

    trend = {
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "fonts": fonts,
            "chars": CHARS,
            "sample_steps": SAMPLE_STEPS,
            "seed": SEED,
            "baseline": str(FT),
            "metric": "L1 vs same-font Latin GT",
            "note": "ft evaluated once; StageA unet swapped per ckpt; same noise seeds",
        },
        "series": [
            {
                "step": r["step"],
                "mean_l1_stageA": r["mean_l1_stageA"],
                "mean_l1_ft_cnstyle": r["mean_l1_ft_cnstyle"],
                "win_rate": r["win_rate"],
                "wins": r["wins"],
                "n": r["n"],
                "by_char": r["by_char"],
            }
            for r in series
        ],
    }
    (TREND / "trend.json").write_text(json.dumps(trend, indent=2, ensure_ascii=False), encoding="utf-8")
    if series:
        latest = series[-1]
        payload = json.loads((TREND / f"metrics_step_{latest['step']}.json").read_text())
        payload.update(
            {
                "t": time.strftime("%Y-%m-%d %H:%M:%S"),
                "stageA_ckpt": payload["ckpt"],
                "stageA_step": payload["step"],
                "baseline": str(FT),
                "fonts": fonts,
                "chars": CHARS,
                "sample_steps": SAMPLE_STEPS,
                "mean_l1_stageA": payload["mean_l1_stageA"],
                "mean_l1_ft_cnstyle": payload["mean_l1_ft_cnstyle"],
                "disclaimer": "Multi-ckpt mid-eval. See mentor_preview/trend/. Not a paper table.",
            }
        )
        (OUT / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {TREND}/trend.json points={len(series)}")


if __name__ == "__main__":
    main()
