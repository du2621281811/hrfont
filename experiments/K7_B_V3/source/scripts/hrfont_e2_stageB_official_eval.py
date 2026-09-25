#!/usr/bin/env python3
"""Official-protocol eval for Stage B (Stage A UNet + SupportAdapter).

Same protocol as Stage A: DejaVu content, 1-shot「永」, DPM++ CFG 7.5, Demo-8×P1.
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
from accelerate.utils import set_seed
from PIL import Image

ROOT = Path("/root/projects/hrfont")
sys.path.insert(0, str(ROOT / "scripts"))

from hrfont_e2_official_eval import (  # noqa: E402
    DEJAVU,
    FT,
    GT_ROOT,
    META_U,
    META_V2,
    P1_CHARS,
    REF8,
    SIZE,
    STYLE_CHAR,
    TAU,
    TOP_M,
    FontDiffuserDeltaDPM,
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
from hrfont_support_adapter import SupportAdapter
from hrfont_support_utils import pick_support

BANK = ROOT / "data/hrfont/e0_bank"
STAGEB = ROOT / "runs/e2_stageB96"
OUT = ROOT / "reports/hrfont_overnight/formal_eval"


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def pool_ec(feat, res_list) -> torch.Tensor:
    parts = []
    for x in list(res_list) + [feat]:
        if x.dim() == 4:
            parts.append(x.mean(dim=(2, 3)).squeeze(0))
        else:
            parts.append(x.flatten())
    return torch.cat(parts, dim=0)


class FontDiffuserDeltaSupportDPM(nn.Module):
    """Stage B: Δ-RSI + optional support tokens via SupportAdapter."""

    def __init__(self, unet, style_encoder, content_encoder, support_adapter, delta_img, support_tensors):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.support_adapter = support_adapter
        self.delta_img = delta_img
        self.support_tensors = support_tensors  # list of (1,3,H,W) or empty

    @property
    def device(self):
        return next(self.parameters()).device

    def set_cond(self, delta_img: torch.Tensor, support_tensors: list) -> None:
        self.delta_img = delta_img
        self.support_tensors = support_tensors

    def _hidden(self, content_images, style_images, zero_delta: bool = False, drop_support: bool = False):
        if content_images.mean().item() > 0.95 and style_images.mean().item() > 0.95:
            zero_delta = True
            drop_support = True
        style_feat, _, _ = self.style_encoder(style_images)
        b, c, h, w = style_feat.shape
        style_hidden = style_feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
        content_feat, content_res = self.content_encoder(content_images)
        content_res = list(content_res) + [content_feat]
        delta = self.delta_img if self.delta_img is not None else content_images
        mix_feat, mix_res = self.content_encoder(delta.expand_as(content_images))
        mix_res = list(mix_res) + [mix_feat]
        if zero_delta:
            delta_res = [torch.zeros_like(x) for x in mix_res]
        else:
            delta_res = [m - c_ for m, c_ in zip(mix_res, content_res)]
        if (not drop_support) and self.support_tensors:
            vecs = []
            for im in self.support_tensors:
                f, rs = self.content_encoder(im.to(content_images.device))
                vecs.append(pool_ec(f, list(rs)).unsqueeze(0))
            if vecs:
                sup = torch.cat(vecs, dim=0).unsqueeze(0)
                sup_tok = self.support_adapter(sup)
                if sup_tok.shape[0] != style_hidden.shape[0]:
                    sup_tok = sup_tok.expand(style_hidden.shape[0], -1, -1)
                style_hidden = torch.cat([style_hidden, sup_tok], dim=1)
        return [style_feat, content_res, style_hidden, delta_res]

    def forward(self, x_t, timesteps, cond, content_encoder_downsample_size, version):
        content_images, style_images = cond[0], cond[1]
        hidden = self._hidden(content_images, style_images)
        return self.unet(x_t, timesteps, encoder_hidden_states=hidden, content_encoder_downsample_size=content_encoder_downsample_size)[0]


def resolve_stageb_ckpt() -> Path:
    for name in ("step_25000.pt", "last.pt"):
        p = STAGEB / name
        if p.exists():
            return p
    cks = sorted(STAGEB.glob("step_*.pt"), key=lambda x: int(x.stem.split("_")[1]))
    if cks:
        return cks[-1]
    raise FileNotFoundError(f"no Stage B ckpt under {STAGEB}")


def load_support_imgs(spec: dict, device) -> list[torch.Tensor]:
    from torchvision import transforms

    tfm = transforms.Compose(
        [
            transforms.Resize((SIZE, SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    out = []
    for p in spec.get("support_imgs", []):
        path = Path(p)
        if path.exists():
            out.append(tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(device))
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=OUT / "metrics_stageB_25000.json")
    args = ap.parse_args()

    device = "cuda:0"
    args_fd = make_args()
    set_seed(123)

    sys.path.insert(0, str(ROOT / "code/ours/FontDiffuser"))
    from src import build_content_encoder, build_style_encoder, build_unet

    ck_path = args.ckpt or resolve_stageb_ckpt()
    blob = _load(ck_path)
    step = int(blob.get("step", 0))
    log(f"Stage B ckpt {ck_path.name} step={step}")

    demo8 = list(META_V2["test_fonts"])
    meta = json.loads((BANK / "meta.json").read_text())
    train = [f for f in meta["train_fonts"] if f not in set(demo8)]
    bank_proto = _load(BANK / "cache/ec_es_r96.pt")["style_proto"]

    se = build_style_encoder(args=args_fd)
    ce = build_content_encoder(args=args_fd)
    se.load_state_dict(_load(FT / "style_encoder.pth"))
    ce.load_state_dict(_load(FT / "content_encoder.pth"))
    se.eval().to(device)
    ce.eval().to(device)

    unet = build_unet(args=args_fd)
    unet.load_state_dict(blob["unet"])
    unet.eval().to(device)

    with torch.no_grad():
        f, rs = ce(torch.randn(1, 3, SIZE, SIZE, device=device))
        in_dim = pool_ec(f, list(rs)).numel()
        style_dim = se(torch.randn(1, 3, SIZE, SIZE, device=device))[0].shape[1]
    support_adapter = SupportAdapter(in_dim, style_dim).to(device)
    support_adapter.load_state_dict(blob["support_adapter"])
    support_adapter.eval()

    ref80 = OUT / "metrics_step_80000.json"
    if ref80.exists():
        res_ft = json.loads(ref80.read_text())["results"]["ft_cnstyle"]
        log(f"ft L1={res_ft['L1_mean']:.4f} (cached from 80k)")
    else:
        res_ft = {"L1_mean": None, "protocol": "Demo-8 P1 DejaVu content 1-shot永 DPM20 cfg7.5"}

    placeholder = torch.zeros(1, 3, SIZE, SIZE, device=device)
    model = FontDiffuserDeltaSupportDPM(unet, se, ce, support_adapter, placeholder, [])
    pipe = build_pipe(model, device, args_fd)

    rows = []
    t0 = time.time()
    n_sup = 0
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
            spec = pick_support(stem, ch, train, png, font_proto=proto)
            sup_imgs = load_support_imgs(spec, device)
            if sup_imgs:
                n_sup += 1
            model.set_cond(delta_img, sup_imgs)
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
            rows.append(
                {
                    "font": stem,
                    "char": ch,
                    "l1": float(np.mean(np.abs(pa - ga))),
                    "gap": spec.get("gap"),
                    "n_support": len(sup_imgs),
                }
            )
    res_b = {
        "method": "stageB_support",
        "ckpt": str(ck_path),
        "step": step,
        "n": len(rows),
        "n_with_support": n_sup,
        "L1_mean": float(np.mean([r["l1"] for r in rows])) if rows else None,
        "seconds": round(time.time() - t0, 1),
        "protocol": res_ft.get("protocol", ""),
        "rows": rows,
    }
    log(f"StageB L1={res_b['L1_mean']:.4f} n={res_b['n']} support_used={n_sup}")

    ref80_data = json.loads(ref80.read_text()) if ref80.exists() else {}
    payload = {
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": res_b["protocol"],
        "ft_cnstyle": ref80_data.get("ft_cnstyle", {}),
        "stageA_formal": ref80_data.get("stageA_formal", {}),
        "stageB": {k: res_b[k] for k in ("L1_mean", "n", "seconds", "step", "ckpt", "protocol", "n_with_support")},
        "results": {"ft_cnstyle": res_ft, "stageB": res_b},
    }
    if ref80_data.get("stageA_formal", {}).get("L1_mean") and res_b["L1_mean"]:
        payload["delta_L1_vs_A"] = ref80_data["stageA_formal"]["L1_mean"] - res_b["L1_mean"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
