#!/usr/bin/env python3
"""Rebuild Demo-8 TEST visuals for mentor page.

Outputs (under mentor_preview/):
  viz_test_overview.png   — 8 fonts × 4 chars @ best-test ckpt (GT|ft|A)
  viz_test_hard.png       — 4 hard fonts × 8 chars @ best-test
  viz_test_timeline.png   — 6 callout cells × GT|A@10k|70k|best
  viz_test_callouts.png   — top-3 win + top-3 lose @ best-test
  viz_test_meta.json

Uses best_test step from train_val_test_curves.json (fallback 145500).
GPU: CUDA_VISIBLE_DEVICES=3
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
FT = ROOT / "runs/ft_cnstyle/global_step_25000"
STAGEA = ROOT / "runs/e2_stageA96"
OUT = ROOT / "reports/hrfont_overnight/mentor_preview"
CURVES = OUT / "train_val_test_curves.json"
EXT = OUT / "extended_demo8_eval.json"

REF8 = list("永和书风骨韵天地")
ALL_CHARS = list("AaOoRg8S")
OVERVIEW_CHARS = ["A", "a", "R", "8"]
TIMELINE_STEPS = [10000, 70000]  # + best filled dynamically
HARD_FONTS = ["FZHuoYYJW-T", "FZJingYLLTJW", "FZFengYKSJ", "FZLingFKSJW-B"]
SIZE = 96
CELL = 52
TAU = 0.07
TOP_M = 3
SAMPLE_STEPS = 25
SEED = 42


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
    return Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def load_pt(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def short_font(name: str) -> str:
    return name.replace("FZ", "").replace("JW", "")[:14]


def ui_font(size: int = 11):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def best_test_step() -> int:
    if CURVES.exists():
        c = json.loads(CURVES.read_text())
        if c.get("best_test"):
            return int(c["best_test"]["step"])
    if EXT.exists():
        ext = json.loads(EXT.read_text())
        if ext.get("results"):
            best_k = min(ext["results"], key=lambda k: ext["results"][k]["mean_l1_a"])
            return int(best_k)
    last = load_pt(STAGEA / "last.pt")
    return int(last.get("step", 145500))


def ckpt_path(step: int) -> Path:
    p = STAGEA / f"step_{step}.pt"
    if p.exists():
        return p
    last = STAGEA / "last.pt"
    if last.exists() and int(load_pt(last).get("step", 0)) == step:
        return last
    cands = sorted(STAGEA.glob("step_*.pt"), key=lambda q: abs(int(q.stem.split("_")[1]) - step))
    if cands and abs(int(cands[0].stem.split("_")[1]) - step) <= 2500:
        return cands[0]
    raise FileNotFoundError(f"no ckpt near step {step}")


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
    se.load_state_dict(load_pt(FT / "style_encoder.pth"))
    ce.load_state_dict(load_pt(FT / "content_encoder.pth"))
    return se.to(device).eval(), ce.to(device).eval()


def build_unet(args, device, state):
    from src import build_unet

    unet = build_unet(args=args)
    unet.load_state_dict(state)
    return unet.to(device).eval()


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


def paste_glyph(canvas, img, x, y, border=None):
    g = img.resize((CELL, CELL), Image.Resampling.BILINEAR)
    if border:
        fr = Image.new("RGB", (CELL + 4, CELL + 4), border)
        fr.paste(g, (2, 2))
        canvas.paste(fr, (x - 2, y - 2))
    else:
        canvas.paste(g, (x, y))


def font_panel(font: str, rows: list[dict], title: str, font_ui, cols=("GT", "ft", "A")) -> Image.Image:
    pad, lw, cw = 6, 22, CELL + 8
    hh, rh = 32, CELL + 16
    w = pad * 2 + lw + len(cols) * cw
    h = hh + len(rows) * rh + pad
    im = Image.new("RGB", (w, h), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    dr.text((pad, 4), title, fill=(18, 21, 26), font=font_ui)
    for i, lab in enumerate(cols):
        dr.text((pad + lw + i * cw + 4, 18), lab, fill=(92, 101, 112), font=font_ui)
    for ri, r in enumerate(rows):
        y = hh + ri * rh
        win = r.get("win", r["l1_a"] < r["l1_ft"])
        border = (15, 122, 74) if win else (180, 83, 9)
        dr.text((pad + 2, y + CELL // 2 - 5), r["ch"], fill=(18, 21, 26), font=font_ui)
        paste_glyph(im, r["gt"], pad + lw, y)
        paste_glyph(im, r["ft"], pad + lw + cw, y)
        paste_glyph(im, r["a"], pad + lw + 2 * cw, y, border=border)
        dr.text((pad + lw + cw, y + CELL + 1), f"{r['l1_ft']:.2f}", fill=(180, 83, 9), font=font_ui)
        dr.text((pad + lw + 2 * cw, y + CELL + 1), f"{r['l1_a']:.2f}", fill=border, font=font_ui)
    return im


def grid_panels(panels: list[Image.Image], ncol: int, bg=(246, 247, 249), gap=10) -> Image.Image:
    if not panels:
        return Image.new("RGB", (100, 100), bg)
    rows_n = (len(panels) + ncol - 1) // ncol
    pw, ph = panels[0].width, panels[0].height
    W = ncol * pw + (ncol + 1) * gap
    H = rows_n * ph + (rows_n + 1) * gap
    canvas = Image.new("RGB", (W, H), bg)
    for i, p in enumerate(panels):
        r, c = divmod(i, ncol)
        canvas.paste(p, (gap + c * (pw + gap), gap + r * (ph + gap)))
    return canvas


class Sampler:
    def __init__(self, args, device, se, ce, train, bank_proto):
        self.args, self.device, self.se, self.ce = args, device, se, ce
        self.train, self.bank_proto = train, bank_proto
        self.unet_ft = build_unet(args, device, load_pt(FT / "unet.pth"))
        self.cache: dict[tuple[int, str, str], dict] = {}

    def _job(self, font, ch):
        proto = style_proto(self.se, font, self.device)
        style_ch = next(r for r in REF8 if png(font, r).exists())
        style = load_im(png(font, style_ch))
        return {
            "font": font,
            "ch": ch,
            "content": load_im(png("_B0_", ch)),
            "gt": load_im(png(font, ch)),
            "style": style,
            "mix": alpha_mix(proto, self.bank_proto, self.train, ch),
        }

    @torch.no_grad()
    def infer(self, step: int, font: str, ch: str, j_idx: int) -> dict:
        key = (step, font, ch)
        if key in self.cache:
            return self.cache[key]
        job = self._job(font, ch)
        g = torch.Generator(device=self.device)
        g.manual_seed(SEED + j_idx * 17)
        content = job["content"].unsqueeze(0).to(self.device)
        style = job["style"].unsqueeze(0).to(self.device)
        pred_ft = sample(
            self.unet_ft, self.se, self.ce, content, style, style,
            self.args, self.device, False, g,
        )
        blob = load_pt(ckpt_path(step))
        unet_a = build_unet(self.args, self.device, blob["unet"])
        g2 = torch.Generator(device=self.device)
        g2.manual_seed(SEED + j_idx * 17)
        pred_a = sample(
            unet_a, self.se, self.ce, content, style, job["mix"].unsqueeze(0).to(self.device),
            self.args, self.device, True, g2,
        )
        del unet_a
        torch.cuda.empty_cache()
        rec = {
            "gt": to_pil(job["gt"]),
            "ft": to_pil(pred_ft[0]),
            "a": to_pil(pred_a[0]),
            "l1_ft": float((pred_ft.cpu() - job["gt"]).abs().mean()),
            "l1_a": float((pred_a.cpu() - job["gt"]).abs().mean()),
            "win": float((pred_a.cpu() - job["gt"]).abs().mean()) < float((pred_ft.cpu() - job["gt"]).abs().mean()),
            "ch": ch,
        }
        self.cache[key] = rec
        return rec


def pick_callouts_from_json(best_step: int) -> tuple[list, list]:
    if not EXT.exists():
        return [], []
    ext = json.loads(EXT.read_text())
    r = ext.get("results", {}).get(str(best_step))
    if not r:
        return [], []
    cells = r.get("cells", [])
    ranked = sorted(cells, key=lambda c: c["l1_ft"] - c["l1_a"], reverse=True)
    wins = ranked[:3]
    losses = ranked[-3:]
    return wins, losses


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    font_ui = ui_font(11)
    best = best_test_step()
    timeline_steps = TIMELINE_STEPS + [best]
    log(f"best_test_step={best} timeline={timeline_steps} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    meta_bank = json.loads((BANK / "meta.json").read_text())
    demo8 = list(meta_bank["demo8"])
    train = [f for f in meta_bank["train_fonts"] if f not in set(demo8)]
    bank_proto = load_pt(BANK / "cache/ec_es_r96.pt")["style_proto"]
    fonts = [f for f in demo8 if any(png(f, r).exists() for r in REF8)]

    sys.path.insert(0, str(ROOT / "code/FontDiffuser"))
    args = make_args()
    se, ce = build_encoders(args, device)
    sampler = Sampler(args, device, se, ce, train, bank_proto)

    job_map = {}
    j_idx = 0
    for font in fonts:
        for ch in ALL_CHARS:
            if png(font, ch).exists() and png("_B0_", ch).exists():
                job_map[(font, ch)] = j_idx
                j_idx += 1

    # --- overview: 8 fonts × 4 chars @ best ---
    overview_panels = []
    for font in fonts:
        rows = []
        for ch in OVERVIEW_CHARS:
            if (font, ch) not in job_map:
                continue
            rec = sampler.infer(best, font, ch, job_map[(font, ch)])
            rows.append(rec)
        if rows:
            overview_panels.append(
                font_panel(font, rows, f"{short_font(font)} @ {best // 1000}k", font_ui)
            )
    overview = grid_panels(overview_panels, ncol=4)
    p_over = OUT / "viz_test_overview.png"
    overview.save(p_over)
    log(f"wrote {p_over} {overview.size}")

    # --- hard fonts: 4 × 8 chars @ best ---
    hard_panels = []
    for font in HARD_FONTS:
        if font not in fonts:
            continue
        rows = []
        for ch in ALL_CHARS:
            if (font, ch) not in job_map:
                continue
            rec = sampler.infer(best, font, ch, job_map[(font, ch)])
            rows.append(rec)
        if rows:
            hard_panels.append(
                font_panel(font, rows, f"HARD · {short_font(font)} @ {best // 1000}k", font_ui)
            )
    hard = grid_panels(hard_panels, ncol=2)
    p_hard = OUT / "viz_test_hard.png"
    hard.save(p_hard)
    log(f"wrote {p_hard} {hard.size}")

    # --- timeline: 6 cells from callouts or defaults ---
    wins_json, loss_json = pick_callouts_from_json(best)
    picks = []
    for c in wins_json[:2] + loss_json[:2]:
        picks.append((c["font"], c["ch"]))
    defaults = [
        ("FZHuoYYJW-T", "R"),
        ("FZJingYLLTJW", "O"),
        ("FZFengYKSJ", "g"),
        ("FZChuangHJW_DB", "S"),
    ]
    for d in defaults:
        if d not in picks and d in job_map:
            picks.append(d)
    picks = picks[:6]

    pad, lw, cw = 6, 22, CELL + 8
    cols = ["GT"] + [f"A@{s // 1000}k" for s in timeline_steps]
    rh = CELL + 16
    hh = 36
    W = pad * 2 + lw + len(cols) * cw
    H = hh + len(picks) * rh + pad
    timeline = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(timeline)
    dr.text((pad, 4), f"Test timeline · GT vs Stage A checkpoints · best={best // 1000}k", fill=(18, 21, 26), font=font_ui)
    for i, lab in enumerate(cols):
        col = (15, 122, 74) if str(best // 1000) in lab else (92, 101, 112)
        dr.text((pad + lw + i * cw + 2, 20), lab, fill=col, font=font_ui)
    for ri, (font, ch) in enumerate(picks):
        y = hh + ri * rh
        job = sampler._job(font, ch)
        dr.text((pad + 2, y + 4), f"{short_font(font)}·{ch}", fill=(18, 21, 26), font=font_ui)
        paste_glyph(timeline, to_pil(job["gt"]), pad + lw, y)
        for si, step in enumerate(timeline_steps):
            rec = sampler.infer(step, font, ch, job_map[(font, ch)])
            border = (15, 122, 74) if step == best else None
            paste_glyph(timeline, rec["a"], pad + lw + (si + 1) * cw, y, border=border)
            dr.text(
                (pad + lw + (si + 1) * cw, y + CELL + 1),
                f"{rec['l1_a']:.2f}",
                fill=(15, 122, 74) if step == best else (92, 101, 112),
                font=font_ui,
            )
    p_time = OUT / "viz_test_timeline.png"
    timeline.save(p_time)
    log(f"wrote {p_time} {timeline.size}")

    # --- callouts strip @ best ---
    all_cells = []
    for font in fonts:
        for ch in ALL_CHARS:
            if (font, ch) not in job_map:
                continue
            rec = sampler.infer(best, font, ch, job_map[(font, ch)])
            all_cells.append({**rec, "font": font})
    ranked = sorted(all_cells, key=lambda c: c["l1_ft"] - c["l1_a"], reverse=True)
    callout_picks = ranked[:3] + ranked[-3:]
    tags = ["WIN", "WIN", "WIN", "LOSE", "LOSE", "LOSE"]
    pad_c = 8
    block_w = 24 + 3 * (CELL + 6)
    row_h = CELL + 34
    callout_im = Image.new("RGB", (pad_c * 2 + block_w, 28 + len(callout_picks) * row_h), (255, 255, 255))
    dr = ImageDraw.Draw(callout_im)
    dr.text((pad_c, 6), f"Demo-8 test callouts @ {best // 1000}k (64 cells)", fill=(18, 21, 26), font=font_ui)
    for i, (c, tag) in enumerate(zip(callout_picks, tags)):
        y = 28 + i * row_h
        delta = c["l1_ft"] - c["l1_a"]
        col = (15, 122, 74) if delta > 0 else (180, 83, 9)
        dr.text((pad_c, y), f"{tag} {short_font(c['font'])}·{c['ch']} dL1={delta:+.2f}", fill=col, font=font_ui)
        yy = y + 14
        paste_glyph(callout_im, c["gt"], pad_c + 20, yy)
        paste_glyph(callout_im, c["ft"], pad_c + 20 + CELL + 6, yy)
        paste_glyph(callout_im, c["a"], pad_c + 20 + 2 * (CELL + 6), yy, border=col)
        for j, lab in enumerate(["GT", "ft", "A"]):
            dr.text((pad_c + 20 + j * (CELL + 6), yy + CELL + 1), lab, fill=(92, 101, 112), font=font_ui)
    p_call = OUT / "viz_test_callouts.png"
    callout_im.save(p_call)
    log(f"wrote {p_call} {callout_im.size}")

    meta_out = {
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_test_step": best,
        "fonts": fonts,
        "hard_fonts": HARD_FONTS,
        "overview_chars": OVERVIEW_CHARS,
        "timeline_steps": timeline_steps,
        "timeline_picks": [{"font": f, "ch": c} for f, c in picks],
        "files": {
            "overview": p_over.name,
            "hard": p_hard.name,
            "timeline": p_time.name,
            "callouts": p_call.name,
        },
        "wins_at_best": sum(1 for c in all_cells if c["win"]),
        "n_cells": len(all_cells),
    }
    (OUT / "viz_test_meta.json").write_text(json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "viz_callouts.json").write_text(
        json.dumps(
            {
                "best_step": best,
                "wins": [
                    {
                        "font": c["font"],
                        "ch": c["ch"],
                        "l1_ft": c["l1_ft"],
                        "l1_a": c["l1_a"],
                        "delta": c["l1_ft"] - c["l1_a"],
                    }
                    for c in ranked[:3]
                ],
                "losses": [
                    {
                        "font": c["font"],
                        "ch": c["ch"],
                        "l1_ft": c["l1_ft"],
                        "l1_a": c["l1_a"],
                        "delta": c["l1_ft"] - c["l1_a"],
                    }
                    for c in ranked[-3:]
                ],
                "n_cells": len(all_cells),
                "wins_n": meta_out["wins_at_best"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log(f"done wins={meta_out['wins_at_best']}/{meta_out['n_cells']}")


if __name__ == "__main__":
    main()
