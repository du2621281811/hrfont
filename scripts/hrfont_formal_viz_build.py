#!/usr/bin/env python3
"""Formal-protocol viz: GT | ft | Stage A | Stage B (DejaVu + 1-shot永 + DPM20).

Outputs under formal_preview/:
  viz_overview.png   — 8 Demo-8 fonts × 4 chars
  viz_callouts.png   — top-3 A wins + top-3 A loses vs ft
  viz_meta.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("HRFONT_VIZ_GPU", "3"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/root/projects/hrfont")
sys.path.insert(0, str(ROOT / "scripts"))

from hrfont_e2_official_eval import (  # noqa: E402
    DEJAVU,
    FT,
    GT_ROOT,
    META_V2,
    REF8,
    STYLE_CHAR,
    FontDiffuserDeltaDPM,
    FontDiffuserOfficialDPM,
    alpha_mix,
    build_pipe,
    load_bank_im,
    make_args,
    png,
    render,
    resolve_font,
    to_arr,
    to_tensor,
    _load,
)
from hrfont_e2_stageB_official_eval import (  # noqa: E402
    FontDiffuserDeltaSupportDPM,
    load_support_imgs,
    pool_ec,
    resolve_stageb_ckpt,
)
from hrfont_support_adapter import SupportAdapter
from hrfont_support_utils import pick_support

BANK = ROOT / "data/hrfont/e0_bank"
STAGEA = ROOT / "runs/e2_stageA96_formal/step_80000.pt"
OUT = ROOT / "reports/hrfont_overnight/formal_preview"
METRICS = ROOT / "reports/hrfont_overnight/formal_eval/metrics_step_80000.json"

OVERVIEW_CHARS = ["A", "a", "R", "8"]
CELL = 56
SEED = 123


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def ui_font(size: int = 11):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def short_font(name: str) -> str:
    m = {
        "FZChuangHJW_DB": "创黑",
        "FZCuanBZBKSJW": "篆变",
        "FZDouNTJW_Te": "斗体",
        "FZFengYKSJ": "风雅",
        "FZHanWZKJW": "汉真",
        "FZHuoYYJW-T": "活页",
        "FZJingYLLTJW": "静雅",
        "FZLingFKSJW-B": "凌风",
    }
    return m.get(name, name.replace("FZ", "")[:8])


def pred_to_pil(t) -> Image.Image:
    if isinstance(t, Image.Image):
        return t
    x = t.detach().float().cpu()
    if x.dim() == 4:
        x = x[0]
    x = (x.clamp(-1, 1) + 1) * 0.5
    return Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def load_gt(font: str, ch: str) -> Image.Image | None:
    tag = ch if ch.isalnum() and ord(ch) < 128 else f"u{ord(ch):04X}"
    for name in (f"{tag}.png", f"{ch}.png"):
        p = GT_ROOT / font / name
        if p.exists():
            return Image.open(p).convert("RGB")
    p = png(font, ch)
    if p.exists():
        return Image.open(p).convert("RGB")
    return None


def paste_glyph(canvas, img: Image.Image, x, y, border=None):
    g = img.resize((CELL, CELL), Image.Resampling.BILINEAR)
    if border:
        fr = Image.new("RGB", (CELL + 4, CELL + 4), border)
        fr.paste(g, (2, 2))
        canvas.paste(fr, (x - 2, y - 2))
    else:
        canvas.paste(g, (x, y))


def font_panel(font: str, rows: list[dict], title: str, font_ui) -> Image.Image:
    cols = ("GT", "ft", "A", "B")
    pad, lw, cw = 6, 28, CELL + 10
    hh, rh = 34, CELL + 18
    w = pad * 2 + lw + len(cols) * cw
    h = hh + len(rows) * rh + pad
    im = Image.new("RGB", (w, h), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    dr.text((pad, 4), title, fill=(18, 21, 26), font=font_ui)
    colors = {"GT": (92, 101, 112), "ft": (180, 83, 9), "A": (31, 111, 143), "B": (124, 58, 237)}
    for i, lab in enumerate(cols):
        dr.text((pad + lw + i * cw + 4, 18), lab, fill=colors[lab], font=font_ui)
    for ri, r in enumerate(rows):
        y = hh + ri * rh
        a_win = r["l1_a"] < r["l1_ft"]
        dr.text((pad + 2, y + CELL // 2 - 5), r["ch"], fill=(18, 21, 26), font=font_ui)
        paste_glyph(im, r["gt"], pad + lw, y)
        paste_glyph(im, r["ft"], pad + lw + cw, y)
        paste_glyph(im, r["a"], pad + lw + 2 * cw, y, border=(15, 122, 74) if a_win else (180, 83, 9))
        paste_glyph(im, r["b"], pad + lw + 3 * cw, y)
        dr.text((pad + lw + cw, y + CELL + 2), f"{r['l1_ft']:.2f}", fill=colors["ft"], font=font_ui)
        dr.text((pad + lw + 2 * cw, y + CELL + 2), f"{r['l1_a']:.2f}", fill=colors["A"], font=font_ui)
        dr.text((pad + lw + 3 * cw, y + CELL + 2), f"{r['l1_b']:.2f}", fill=colors["B"], font=font_ui)
    return im


def grid_panels(panels: list[Image.Image], ncol: int = 4) -> Image.Image:
    gap = 10
    pw, ph = panels[0].width, panels[0].height
    rows_n = (len(panels) + ncol - 1) // ncol
    W = ncol * pw + (ncol + 1) * gap
    H = rows_n * ph + (rows_n + 1) * gap
    canvas = Image.new("RGB", (W, H), (246, 247, 249))
    for i, p in enumerate(panels):
        r, c = divmod(i, ncol)
        canvas.paste(p, (gap + c * (pw + gap), gap + r * (ph + gap)))
    return canvas


class FormalViz:
    def __init__(self, device: str):
        from accelerate.utils import set_seed
        from src import build_content_encoder, build_style_encoder, build_unet

        set_seed(SEED)
        self.device = device
        self.args = make_args()
        meta = json.loads((BANK / "meta.json").read_text())
        self.demo8 = list(META_V2["test_fonts"])
        self.train = [f for f in meta["train_fonts"] if f not in set(self.demo8)]
        self.bank_proto = _load(BANK / "cache/ec_es_r96.pt")["style_proto"]

        se = build_style_encoder(args=self.args)
        ce = build_content_encoder(args=self.args)
        se.load_state_dict(_load(FT / "style_encoder.pth"))
        ce.load_state_dict(_load(FT / "content_encoder.pth"))
        se.eval().to(device)
        ce.eval().to(device)
        self.se, self.ce = se, ce

        self.unet_ft = build_unet(args=self.args)
        self.unet_ft.load_state_dict(_load(FT / "unet.pth"))
        self.unet_ft.eval().to(device)
        self.pipe_ft = build_pipe(FontDiffuserOfficialDPM(self.unet_ft, se, ce), device, self.args)

        blob_a = _load(STAGEA)
        self.unet_a = build_unet(args=self.args)
        self.unet_a.load_state_dict(blob_a["unet"])
        self.unet_a.eval().to(device)
        ph = torch.zeros(1, 3, 96, 96, device=device)
        self.model_a = FontDiffuserDeltaDPM(self.unet_a, se, ce, ph)
        self.pipe_a = build_pipe(self.model_a, device, self.args)

        blob_b = _load(resolve_stageb_ckpt())
        self.unet_b = build_unet(args=self.args)
        self.unet_b.load_state_dict(blob_b["unet"])
        self.unet_b.eval().to(device)
        with torch.no_grad():
            f, rs = ce(torch.randn(1, 3, 96, 96, device=device))
            in_dim = pool_ec(f, list(rs)).numel()
            style_dim = se(torch.randn(1, 3, 96, 96, device=device))[0].shape[1]
        self.support_adapter = SupportAdapter(in_dim, style_dim).to(device)
        self.support_adapter.load_state_dict(blob_b["support_adapter"])
        self.support_adapter.eval()
        self.model_b = FontDiffuserDeltaSupportDPM(self.unet_b, se, ce, self.support_adapter, ph, [])
        self.pipe_b = build_pipe(self.model_b, device, self.args)

    @torch.no_grad()
    def _sample(self, pipe, model_a_or_b, font, ch, content_t, style_t, proto):
        delta_img = alpha_mix(proto.to(self.device), self.bank_proto, self.train, ch, self.device) if proto is not None else None
        if delta_img is None:
            delta_img = content_t
        if hasattr(model_a_or_b, "set_delta"):
            model_a_or_b.set_delta(delta_img)
        elif hasattr(model_a_or_b, "set_cond"):
            spec = pick_support(font, ch, self.train, png, font_proto=proto)
            sup = load_support_imgs(spec, self.device)
            model_a_or_b.set_cond(delta_img, sup)
        return pipe.generate(
            content_images=content_t,
            style_images=style_t,
            batch_size=1,
            order=2,
            num_inference_step=20,
            content_encoder_downsample_size=3,
            dm_size=(96, 96),
            algorithm_type="dpmsolver++",
            method="multistep",
        )[0]

    @torch.no_grad()
    def infer_cell(self, font: str, ch: str) -> dict | None:
        gt_im = load_gt(font, ch)
        if gt_im is None:
            return None
        fp = resolve_font(font)
        style_t = to_tensor(render(fp, STYLE_CHAR, 96), 96).to(self.device)
        content_t = to_tensor(render(DEJAVU, ch, 96), 96).to(self.device)
        proto = self.bank_proto.get(font)
        if proto is None:
            feats = []
            for r in REF8:
                p = png(font, r)
                if p.exists():
                    feats.append(self.se(load_bank_im(p).to(self.device))[0].flatten(1).mean(0).cpu())
            proto = torch.stack(feats).mean(0) if feats else None

        pred_ft = self._sample(self.pipe_ft, None, font, ch, content_t, style_t, proto)
        pred_a = self._sample(self.pipe_a, self.model_a, font, ch, content_t, style_t, proto)
        pred_b = self._sample(self.pipe_b, self.model_b, font, ch, content_t, style_t, proto)

        ga = to_arr(gt_im)
        l1_ft = float(np.mean(np.abs(to_arr(pred_ft) - ga)))
        l1_a = float(np.mean(np.abs(to_arr(pred_a) - ga)))
        l1_b = float(np.mean(np.abs(to_arr(pred_b) - ga)))
        return {
            "font": font,
            "ch": ch,
            "gt": gt_im,
            "ft": pred_ft if isinstance(pred_ft, Image.Image) else pred_to_pil(pred_ft),
            "a": pred_to_pil(pred_a),
            "b": pred_to_pil(pred_b),
            "l1_ft": l1_ft,
            "l1_a": l1_a,
            "l1_b": l1_b,
        }


def main() -> None:
    sys.path.insert(0, str(ROOT / "code/FontDiffuser"))
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    font_ui = ui_font(11)
    log(f"formal viz gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    viz = FormalViz(device)
    fonts = list(viz.demo8)
    t0 = time.time()

    overview_panels = []
    all_cells = []
    for font in fonts:
        rows = []
        for ch in OVERVIEW_CHARS:
            rec = viz.infer_cell(font, ch)
            if rec:
                rows.append(rec)
                all_cells.append(rec)
            log(f"  {short_font(font)}·{ch} ft={rec['l1_ft']:.3f} A={rec['l1_a']:.3f} B={rec['l1_b']:.3f}" if rec else f"  skip {font}·{ch}")
        if rows:
            overview_panels.append(font_panel(font, rows, f"{short_font(font)} · formal", font_ui))

    overview = grid_panels(overview_panels, ncol=4)
    banner = Image.new("RGB", (overview.width, 36), (255, 255, 255))
    dr = ImageDraw.Draw(banner)
    dr.text(
        (8, 8),
        "Formal 协议 · DejaVu content · 1-shot「永」· DPM++20 CFG7.5 · 列=GT|ft|A@80k|B@25k",
        fill=(18, 21, 26),
        font=ui_font(12),
    )
    final = Image.new("RGB", (overview.width, overview.height + 36), (246, 247, 249))
    final.paste(banner, (0, 0))
    final.paste(overview, (0, 36))
    p_over = OUT / "viz_overview.png"
    final.save(p_over)
    log(f"wrote {p_over} {final.size}")

    ranked = sorted(all_cells, key=lambda c: c["l1_ft"] - c["l1_a"], reverse=True)
    picks = ranked[:3] + ranked[-3:]
    tags = ["A胜ft", "A胜ft", "A胜ft", "A负ft", "A负ft", "A负ft"]
    pad, block_w, row_h = 8, 28 + 4 * (CELL + 8), CELL + 36
    callout = Image.new("RGB", (pad * 2 + block_w, 32 + len(picks) * row_h), (255, 255, 255))
    dr = ImageDraw.Draw(callout)
    dr.text((pad, 6), "典型样例 · A vs ft（绿框=A 优于 ft）", fill=(18, 21, 26), font=font_ui)
    for i, (rec, tag) in enumerate(zip(picks, tags)):
        y = 32 + i * row_h
        delta = rec["l1_ft"] - rec["l1_a"]
        col = (15, 122, 74) if delta > 0 else (180, 83, 9)
        dr.text((pad, y), f"{tag} {short_font(rec['font'])}·{rec['ch']} ΔL1={delta:+.3f}", fill=col, font=font_ui)
        yy = y + 14
        for j, (key, lab) in enumerate([("gt", "GT"), ("ft", "ft"), ("a", "A"), ("b", "B")]):
            x = pad + 20 + j * (CELL + 8)
            border = col if key == "a" else None
            paste_glyph(callout, rec[key], x, yy, border=border)
            dr.text((x, yy + CELL + 2), lab, fill=(92, 101, 112), font=font_ui)
    p_call = OUT / "viz_callouts.png"
    callout.save(p_call)
    log(f"wrote {p_call}")

    meta = {
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "Demo-8 × AaR8 · DejaVu · 1-shot永 · DPM20 CFG7.5",
        "n_cells": len(all_cells),
        "seconds": round(time.time() - t0, 1),
        "files": ["viz_overview.png", "viz_callouts.png"],
    }
    (OUT / "viz_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"done {meta['seconds']}s")


if __name__ == "__main__":
    main()
