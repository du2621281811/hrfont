#!/usr/bin/env python3
"""Official-protocol eval: DPM++ + CFG 7.5 + DejaVu content + 1-shot「永」.

Compares ft_cnstyle@25k (official RSI) vs Stage A (Delta-RSI) on Demo-8 P1.
Writes reports/hrfont_overnight/formal_eval/metrics.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("HRFONT_EVAL_GPU", "3"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import set_seed
from PIL import Image

ROOT = Path("/root/projects/hrfont")
BANK = ROOT / "data/hrfont/e0_bank"
FT = ROOT / "runs/ft_cnstyle/global_step_25000"
STAGEA = ROOT / "runs/e2_stageA96_formal"
OUT = ROOT / "reports/hrfont_overnight/formal_eval"
REF8 = list("永和书风骨韵天地")
SIZE = 96
TAU = 0.07
TOP_M = 3
DEJAVU = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_DIR = Path("/root/data/font_50")
META_V2 = json.loads((ROOT / "data/retrain_v2/meta.json").read_text())
META_U = json.loads((ROOT / "data/unified_v1/meta.json").read_text())
STYLE_CHAR = (META_V2.get("style_refs") or ["永"])[0]
P1_CHARS = list(META_U["L_p1"])
GT_ROOT = ROOT / "data/unified_v1/renders/128/gt_latin"


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def _load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def render(font_path: Path, ch: str, size: int = 96) -> Image.Image:
    from PIL import ImageDraw, ImageFont

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
    draw.text(((size - w) // 2 - bbox[0], (size - h) // 2 - bbox[1]), ch, font=f, fill=(0, 0, 0))
    return img


def resolve_font(stem: str) -> Path:
    for ext in (".TTF", ".ttf", ".OTF", ".otf"):
        p = FONT_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    hits = list(FONT_DIR.glob(stem + ".*"))
    if not hits:
        raise FileNotFoundError(stem)
    return hits[0]


def to_tensor(img: Image.Image, size=96) -> torch.Tensor:
    import torchvision.transforms as T

    tfm = T.Compose([T.Resize(size), T.ToTensor(), T.Normalize([0.5], [0.5])])
    return tfm(img.convert("RGB")).unsqueeze(0)


def to_arr(im: Image.Image, size=96) -> np.ndarray:
    return np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0


def png(font: str, ch: str) -> Path:
    return BANK / "r96" / font / f"u{ord(ch):04X}.png"


def load_bank_im(path: Path) -> torch.Tensor:
    return to_tensor(Image.open(path).convert("RGB"), SIZE)


class FontDiffuserOfficialDPM(nn.Module):
    """Standard ft_cnstyle forward for DPM pipeline."""

    def __init__(self, unet, style_encoder, content_encoder):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x_t, timesteps, cond, content_encoder_downsample_size, version):
        content_images, style_images = cond[0], cond[1]
        style_feat, _, _ = self.style_encoder(style_images)
        b, c, h, w = style_feat.shape
        style_hidden = style_feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
        content_feat, content_res = self.content_encoder(content_images)
        content_res = list(content_res) + [content_feat]
        struct_feat, struct_res = self.content_encoder(style_images)
        struct_res = list(struct_res) + [struct_feat]
        hidden = [style_feat, content_res, style_hidden, struct_res]
        return self.unet(x_t, timesteps, encoder_hidden_states=hidden, content_encoder_downsample_size=content_encoder_downsample_size)[0]


class FontDiffuserDeltaDPM(nn.Module):
    """Stage A: RSI structure from Δ (bank mix − DejaVu content features)."""

    def __init__(self, unet, style_encoder, content_encoder, delta_img: torch.Tensor):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        if delta_img is None:
            raise ValueError("delta_img required")
        self.register_buffer("delta_img", delta_img)

    @property
    def device(self):
        return next(self.parameters()).device

    def set_delta(self, delta_img: torch.Tensor) -> None:
        self.register_buffer("delta_img", delta_img, persistent=False)

    def _hidden(self, content_images, style_images, zero_delta: bool = False):
        # CFG uncond branch uses blank content/style (all ~1.0 after normalize)
        if content_images.mean().item() > 0.95 and style_images.mean().item() > 0.95:
            zero_delta = True
        style_feat, _, _ = self.style_encoder(style_images)
        b, c, h, w = style_feat.shape
        style_hidden = style_feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
        content_feat, content_res = self.content_encoder(content_images)
        content_res = list(content_res) + [content_feat]
        mix_feat, mix_res = self.content_encoder(self.delta_img.expand_as(content_images))
        mix_res = list(mix_res) + [mix_feat]
        if zero_delta:
            delta_res = [torch.zeros_like(x) for x in mix_res]
        else:
            delta_res = [m - c_ for m, c_ in zip(mix_res, content_res)]
        return [style_feat, content_res, style_hidden, delta_res]

    def forward(self, x_t, timesteps, cond, content_encoder_downsample_size, version):
        content_images, style_images = cond[0], cond[1]
        hidden = self._hidden(content_images, style_images, zero_delta=False)
        return self.unet(x_t, timesteps, encoder_hidden_states=hidden, content_encoder_downsample_size=content_encoder_downsample_size)[0]


@torch.no_grad()
def alpha_mix(proto_q, bank_proto, train_fonts, ch, device):
    scores = []
    for g in train_fonts:
        if g not in bank_proto or not png(g, ch).exists():
            continue
        v = bank_proto[g].float().flatten().to(proto_q.device)
        scores.append((g, float(F.cosine_similarity(proto_q.flatten()[None], v[None]).item())))
    scores.sort(key=lambda x: -x[1])
    top = scores[:TOP_M]
    if not top:
        return None
    logits = torch.tensor([s / TAU for _, s in top], device=device)
    w = torch.softmax(logits, dim=0)
    mix = None
    for i, (g, _) in enumerate(top):
        im = load_bank_im(png(g, ch)).to(device)
        mix = im * float(w[i]) if mix is None else mix + im * float(w[i])
    return mix


def build_pipe(model, device, args):
    sys.path.insert(0, str(ROOT / "code/ours/FontDiffuser"))
    from src import build_ddpm_scheduler
    from src.dpm_solver.pipeline_dpm_solver import FontDiffuserDPMPipeline

    train_sched = build_ddpm_scheduler(args)
    model.to(device).eval()
    return FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_sched,
        guidance_type="classifier-free",
        guidance_scale=7.5,
    )


def make_args():
    sys.path.insert(0, str(ROOT / "code/ours/FontDiffuser"))
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
    args.algorithm_type = "dpmsolver++"
    args.num_inference_steps = 20
    args.order = 2
    args.method = "multistep"
    return args


def resolve_stagea_ckpt(explicit: Path | None = None) -> Path:
    if explicit is not None and explicit.exists():
        return explicit
    env_ck = os.environ.get("HRFONT_STAGEA_CKPT")
    if env_ck:
        p = Path(env_ck)
        if p.exists():
            return p
    for name in ("step_80000.pt", "step_70000.pt", "last.pt"):
        p = STAGEA / name
        if p.exists():
            return p
    cks = sorted(STAGEA.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if cks:
        return cks[-1]
    raise FileNotFoundError(f"no Stage A formal ckpt under {STAGEA}")


def eval_method(name: str, pipe, fonts: list[str], chars: list[str], device: str) -> dict:
    rows = []
    t0 = time.time()
    for stem in fonts:
        fp = resolve_font(stem)
        style_t = to_tensor(render(fp, STYLE_CHAR, SIZE), SIZE).to(device)
        for ch in chars:
            content_t = to_tensor(render(DEJAVU, ch, SIZE), SIZE).to(device)
            pred = pipe.generate(
                content_images=content_t,
                style_images=style_t,
                batch_size=1,
                order=2,
                num_inference_step=20,
                content_encoder_downsample_size=3,
                dm_size=(SIZE, SIZE),
                algorithm_type="dpmsolver++",
                method="multistep",
            )[0]
            tag = ch if ch.isalnum() and ord(ch) < 128 else f"u{ord(ch):04X}"
            gt_path = GT_ROOT / stem / f"{tag}.png"
            if not gt_path.exists():
                gt_path = GT_ROOT / stem / f"{ch}.png"
            if not gt_path.exists():
                continue
            pa, ga = to_arr(pred), to_arr(Image.open(gt_path))
            rows.append({"font": stem, "char": ch, "l1": float(np.mean(np.abs(pa - ga)))})
    return {
        "method": name,
        "n": len(rows),
        "L1_mean": float(np.mean([r["l1"] for r in rows])) if rows else None,
        "seconds": round(time.time() - t0, 1),
        "protocol": "Demo-8 P1 DejaVu content 1-shot永 DPM20 cfg7.5",
        "rows": rows,
    }


def resolve_e3_ckpt(explicit: Path | None = None) -> Path:
    e3 = ROOT / "runs/e3_rsi_style_control"
    if explicit is not None and explicit.exists():
        return explicit
    for name in ("step_10000.pt", "last.pt"):
        p = e3 / name
        if p.exists():
            return p
    cks = sorted(e3.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if cks:
        return cks[-1]
    raise FileNotFoundError(f"no E3 ckpt under {e3}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None, help="Stage A / E3 ckpt path")
    ap.add_argument("--out", type=Path, default=None, help="Output metrics JSON")
    ap.add_argument("--skip-ft", action="store_true", help="Skip ft_cnstyle eval (ablation speedup)")
    ap.add_argument(
        "--rsi-style",
        action="store_true",
        help="E3 control: eval with FontDiffuserOfficialDPM (RSI←Ec(style)), not Δ",
    )
    args_cli = ap.parse_args()
    out_path = args_cli.out or (OUT / "metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda:0"
    args = make_args()
    set_seed(123)

    sys.path.insert(0, str(ROOT / "code/ours/FontDiffuser"))
    from src import build_content_encoder, build_style_encoder, build_unet

    demo8 = list(META_V2["test_fonts"])

    se = build_style_encoder(args=args)
    ce = build_content_encoder(args=args)
    se.load_state_dict(_load(FT / "style_encoder.pth"))
    ce.load_state_dict(_load(FT / "content_encoder.pth"))
    se.eval()
    ce.eval()

    # --- E3 only: RSI still Ec(style) ---
    if args_cli.rsi_style:
        ck_path = resolve_e3_ckpt(args_cli.ckpt)
        blob = _load(ck_path)
        step = int(blob.get("step", 0))
        ref80 = OUT / "metrics_step_80000.json"
        if ref80.exists():
            res_ft = json.loads(ref80.read_text(encoding="utf-8"))["results"]["ft_cnstyle"]
        else:
            res_ft = {"L1_mean": None, "protocol": "Demo-8 P1 DejaVu content 1-shot永 DPM20 cfg7.5"}
        unet_e3 = build_unet(args=args)
        unet_e3.load_state_dict(blob["unet"])
        pipe_e3 = build_pipe(FontDiffuserOfficialDPM(unet_e3, se, ce), device, args)
        res_e3 = eval_method("e3_rsi_style@10k", pipe_e3, demo8, P1_CHARS, device)
        del unet_e3, pipe_e3
        torch.cuda.empty_cache()
        ref_a = json.loads(ref80.read_text(encoding="utf-8")) if ref80.exists() else {}
        payload = {
            "t": time.strftime("%Y-%m-%d %H:%M:%S"),
            "protocol": res_e3["protocol"],
            "experiment": "E3_rsi_style_control",
            "note": "Same formal data/init as Stage A; RSI structure = Ec(style) not Δ",
            "ft_cnstyle": ref80 and ref_a.get("ft_cnstyle"),
            "stageA_formal": ref_a.get("stageA_formal"),
            "e3_rsi_style": {
                k: res_e3[k]
                for k in ("L1_mean", "n", "seconds", "protocol")
            }
            | {"step": step, "ckpt": str(ck_path)},
            "results": {"ft_cnstyle": res_ft, "e3_rsi_style": res_e3},
        }
        if ref_a.get("stageA_formal", {}).get("L1_mean") and res_e3["L1_mean"]:
            payload["delta_L1_vs_A"] = ref_a["stageA_formal"]["L1_mean"] - res_e3["L1_mean"]
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"E3 L1={res_e3['L1_mean']:.4f} wrote {out_path}")
        return

    meta = json.loads((BANK / "meta.json").read_text())
    train = [f for f in meta["train_fonts"] if f not in set(demo8)]
    bank_proto = _load(BANK / "cache/ec_es_r96.pt")["style_proto"]

    # --- ft_cnstyle official ---
    if args_cli.skip_ft:
        ref = OUT / "metrics_step_80000.json"
        if ref.exists():
            res_ft = json.loads(ref.read_text(encoding="utf-8"))["results"]["ft_cnstyle"]
            log(f"ft L1={res_ft['L1_mean']:.4f} (cached)")
        else:
            res_ft = {"L1_mean": None, "n": 0, "seconds": 0, "protocol": "skipped"}
            log("skip-ft: no cached ft metrics")
    else:
        unet_ft = build_unet(args=args)
        unet_ft.load_state_dict(_load(FT / "unet.pth"))
        pipe_ft = build_pipe(FontDiffuserOfficialDPM(unet_ft, se, ce), device, args)
        res_ft = eval_method("ft_cnstyle@25k", pipe_ft, demo8, P1_CHARS, device)
        log(f"ft L1={res_ft['L1_mean']:.4f} n={res_ft['n']}")
        del unet_ft, pipe_ft
        torch.cuda.empty_cache()

    # --- Stage A formal (per-char Δ for DPM) ---
    ck_path = resolve_stagea_ckpt(args_cli.ckpt)
    blob = _load(ck_path)
    step = int(blob.get("step", 0))
    unet_a = build_unet(args=args)
    unet_a.load_state_dict(blob["unet"])
    placeholder = torch.zeros(1, 3, SIZE, SIZE, device=device)
    model_a = FontDiffuserDeltaDPM(unet_a, se, ce, placeholder)
    pipe_a = build_pipe(model_a, device, args)

    rows_a = []
    t0 = time.time()
    for stem in demo8:
        fp = resolve_font(stem)
        style_t = to_tensor(render(fp, STYLE_CHAR, SIZE), SIZE).to(device)
        proto = bank_proto.get(stem)
        if proto is None:
            feats = []
            for r in REF8:
                p = png(stem, r)
                if p.exists():
                    feats.append(se(load_bank_im(p).to(device))[0].flatten(1).mean(0).cpu())
            proto = torch.stack(feats).mean(0) if feats else None
        for ch in P1_CHARS:
            content_t = to_tensor(render(DEJAVU, ch, SIZE), SIZE).to(device)
            delta_img = alpha_mix(proto.to(device), bank_proto, train, ch, device) if proto is not None else None
            if delta_img is None:
                delta_img = content_t
            model_a.set_delta(delta_img)
            pred = pipe_a.generate(
                content_images=content_t,
                style_images=style_t,
                batch_size=1,
                order=2,
                num_inference_step=20,
                content_encoder_downsample_size=3,
                dm_size=(SIZE, SIZE),
                algorithm_type="dpmsolver++",
                method="multistep",
            )[0]
            tag = ch if ch.isalnum() and ord(ch) < 128 else f"u{ord(ch):04X}"
            gt_path = GT_ROOT / stem / f"{tag}.png"
            if not gt_path.exists():
                gt_path = GT_ROOT / stem / f"{ch}.png"
            if not gt_path.exists():
                continue
            pa, ga = to_arr(pred), to_arr(Image.open(gt_path))
            rows_a.append({"font": stem, "char": ch, "l1": float(np.mean(np.abs(pa - ga)))})
    del pipe_a, model_a
    res_a = {
        "method": "stageA_formal",
        "ckpt": str(ck_path),
        "step": step,
        "n": len(rows_a),
        "L1_mean": float(np.mean([r["l1"] for r in rows_a])) if rows_a else None,
        "seconds": round(time.time() - t0, 1),
        "protocol": res_ft["protocol"],
        "rows": rows_a,
    }
    log(f"StageA L1={res_a['L1_mean']:.4f} n={res_a['n']} ckpt={ck_path.name}")

    payload = {
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": res_ft["protocol"],
        "ft_cnstyle": {k: res_ft[k] for k in ("L1_mean", "n", "seconds", "protocol")},
        "stageA_formal": {k: res_a[k] for k in ("L1_mean", "n", "seconds", "step", "ckpt", "protocol")},
        "delta_L1": (res_ft["L1_mean"] - res_a["L1_mean"]) if res_a["L1_mean"] and res_ft["L1_mean"] else None,
        "results": {"ft_cnstyle": res_ft, "stageA_formal": res_a},
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
