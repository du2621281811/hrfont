#!/usr/bin/env python3
"""Full Demo-8 test eval: all 8 holdout fonts × 8 chars at multiple Stage A ckpts.

Demo-8 fonts are excluded from Stage A training (see StageASet).
Default GPU: CUDA_VISIBLE_DEVICES=3. Writes extended_demo8_eval.json.
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
from PIL import Image
from torchvision import transforms

ROOT = Path("/root/projects/hrfont")
BANK = ROOT / "data/hrfont/e0_bank"
FT = ROOT / "runs/ft_cnstyle/global_step_25000"
STAGEA = ROOT / "runs/e2_stageA96"
OUT = ROOT / "reports/hrfont_overnight/mentor_preview"
REF8 = list("永和书风骨韵天地")
CHARS = list("AaOoRg8S")
SIZE = 96
TAU = 0.07
TOP_M = 3
SAMPLE_STEPS = 25
SEED = 42
SWEEP_STEPS = [10000, 25000, 40000, 55000, 70000, 85000, 100000, 115000, 130000, 140000]


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
    if not feats:
        raise RuntimeError(f"no refs for {font}")
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
    logits = torch.tensor([s / TAU for _, s in top])
    w = torch.softmax(logits, dim=0)
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
        out = unet(x, t.expand(x.shape[0]), encoder_hidden_states=hidden,
                   content_encoder_downsample_size=args.content_encoder_downsample_size)
        x = sched.step(out[0], t, x).prev_sample
    return x


def resolve_ckpts(wanted: list[int]) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for s in wanted:
        p = STAGEA / f"step_{s}.pt"
        if p.exists():
            out.append((s, p))
    last = STAGEA / "last.pt"
    if last.exists():
        bs = int(_load(last).get("step", 0))
        if bs:
            p = STAGEA / f"step_{bs}.pt"
            out.append((bs, p if p.exists() else last))
    seen: set[int] = set()
    uniq = []
    for s, p in sorted(out, key=lambda x: x[0]):
        if s in seen:
            continue
        seen.add(s)
        uniq.append((s, p))
    return uniq


def eval_step(unet_a, unet_ft, se, ce, jobs, args, device) -> dict:
    m_a, m_ft, cells = [], [], []
    for j, job in enumerate(jobs):
        g = torch.Generator(device=device)
        g.manual_seed(SEED + j * 17)
        content = job["content"].unsqueeze(0).to(device)
        style = job["style"].unsqueeze(0).to(device)
        mix = job["mix"].unsqueeze(0).to(device)
        pred_a = sample(unet_a, se, ce, content, style, mix, args, device, True, g)
        g2 = torch.Generator(device=device)
        g2.manual_seed(SEED + j * 17)
        pred_ft = sample(unet_ft, se, ce, content, style, style, args, device, False, g2)
        la = float((pred_a.cpu() - job["gt"]).abs().mean())
        lf = float((pred_ft.cpu() - job["gt"]).abs().mean())
        m_a.append(la)
        m_ft.append(lf)
        cells.append({"font": job["font"], "ch": job["ch"], "l1_a": la, "l1_ft": lf, "win": la < lf})
    by_font: dict[str, dict] = {}
    for c in cells:
        fr = by_font.setdefault(c["font"], {"wins": 0, "n": 0, "a": [], "ft": []})
        fr["n"] += 1
        fr["a"].append(c["l1_a"])
        fr["ft"].append(c["l1_ft"])
        fr["wins"] += int(c["win"])
    for fr in by_font.values():
        fr["mean_a"] = float(np.mean(fr["a"]))
        fr["mean_ft"] = float(np.mean(fr["ft"]))
        del fr["a"], fr["ft"]
    wins = sum(c["win"] for c in cells)
    n = len(cells)
    return {
        "mean_l1_a": float(np.mean(m_a)),
        "mean_l1_ft": float(np.mean(m_ft)),
        "wins": wins,
        "n": n,
        "win_rate": wins / max(n, 1),
        "by_font": by_font,
        "cells": cells,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    meta = json.loads((BANK / "meta.json").read_text())
    demo8 = list(meta["demo8"])
    train = [f for f in meta["train_fonts"] if f not in set(demo8)]
    bank_proto = _load(BANK / "cache/ec_es_r96.pt")["style_proto"]

    fonts = [f for f in demo8 if any(png(f, r).exists() for r in REF8)]
    log(f"demo8 fonts={len(fonts)} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    sys.path.insert(0, str(ROOT / "code/FontDiffuser"))
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

    ckpts = resolve_ckpts(SWEEP_STEPS)
    log(f"ckpts={[s for s, _ in ckpts]}")

    unet_ft = build_unet(args, device, _load(FT / "unet.pth"))
    results: dict[str, dict] = {}
    for step, ck_path in ckpts:
        t0 = time.time()
        unet_a = build_unet(args, device, _load(ck_path)["unet"])
        rec = eval_step(unet_a, unet_ft, se, ce, jobs, args, device)
        rec["step"] = step
        rec["ckpt"] = str(ck_path)
        rec["sec"] = round(time.time() - t0, 1)
        results[str(step)] = rec
        log(f"step={step} test L1 A={rec['mean_l1_a']:.4f} ft={rec['mean_l1_ft']:.4f} "
            f"wins={rec['wins']}/{rec['n']} dt={rec['sec']}s")
        del unet_a
        torch.cuda.empty_cache()

    payload = {
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "split": "test",
            "description": "Demo-8 holdout fonts (excluded from Stage A training)",
            "fonts": fonts,
            "chars": CHARS,
            "sample_steps": SAMPLE_STEPS,
            "seed": SEED,
            "metric": "generation L1 vs same-font Latin GT",
        },
        "fonts": fonts,
        "chars": CHARS,
        "results": results,
    }
    (OUT / "extended_demo8_eval.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {OUT / 'extended_demo8_eval.json'} steps={len(results)}")


if __name__ == "__main__":
    main()
