#!/usr/bin/env python3
"""E1 smoke: RSI structure source = content encoder(delta image), not E_c(style hanzi)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path("/root/projects/hrfont")
REP = ROOT / "reports/hrfont_overnight"
BANK = ROOT / "data/hrfont/e0_bank"
CKPT = ROOT / "runs/ft_cnstyle/global_step_25000"
if not (CKPT / "unet.pth").exists():
    CKPT = ROOT / "code/FontDiffuser/ckpt"


def log(msg: str) -> None:
    REP.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with (REP / "e1_smoke.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_rgb(path: Path, size: int) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    return tfm(Image.open(path).convert("RGB")).unsqueeze(0)


def main() -> None:
    sys.path.insert(0, str(ROOT / "code/FontDiffuser"))
    from configs.fontdiffuser import get_parser
    from src import FontDiffuserModel, build_content_encoder, build_style_encoder, build_unet

    size = 96
    args = get_parser().parse_args([])
    args.content_image_size = (size, size)
    args.style_image_size = (size, size)
    args.resolution = size
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log(f"E1 smoke device={device} ckpt={CKPT}")

    unet = build_unet(args=args)
    se = build_style_encoder(args=args)
    ce = build_content_encoder(args=args)
    unet.load_state_dict(torch.load(CKPT / "unet.pth", map_location="cpu"))
    se.load_state_dict(torch.load(CKPT / "style_encoder.pth", map_location="cpu"))
    ce.load_state_dict(torch.load(CKPT / "content_encoder.pth", map_location="cpu"))
    model = FontDiffuserModel(unet=unet, style_encoder=se, content_encoder=ce).to(device)
    model.eval()

    orig_forward = model.forward

    def delta_forward(x_t, timesteps, style_images, content_images, content_encoder_downsample_size, delta_images=None):
        # Official: RSI uses E_c(style_images). We swap in E_c(delta_images).
        if delta_images is None:
            delta_images = content_images
        style_img_feature, _, _ = model.style_encoder(style_images)
        b, c, h, w = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(b, h * w, c)
        content_img_feature, content_residual_features = model.content_encoder(content_images)
        content_residual_features.append(content_img_feature)
        style_content_feature, style_content_res_features = model.content_encoder(delta_images)
        style_content_res_features.append(style_content_feature)
        input_hidden_states = [
            style_img_feature,
            content_residual_features,
            style_hidden_states,
            style_content_res_features,
        ]
        out = model.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        return out[0]

    model.forward = delta_forward  # type: ignore

    # Prefer real E0 glyphs; fall back to random if bank not ready.
    style = content = delta = None
    bank96 = BANK / "r96"
    if bank96.exists():
        fonts = [p for p in bank96.iterdir() if p.is_dir() and not p.name.startswith("_")]
        if fonts:
            sty = next((fonts[0] / f"u{ord(ch):04X}.png" for ch in "永和书风"), None)
            for ch in "永和书风":
                p = fonts[0] / f"u{ord(ch):04X}.png"
                if p.exists():
                    style = load_rgb(p, size)
                    break
            for ch in "aAoO口":
                p = fonts[0] / f"u{ord(ch):04X}.png"
                if p.exists():
                    content = load_rgb(p, size)
                    delta = content
                    break
    if style is None:
        style = torch.randn(1, 3, size, size)
        content = torch.randn(1, 3, size, size)
        delta = torch.randn(1, 3, size, size)
        log("E1 smoke using random tensors (bank not ready)")

    x = torch.randn(1, 3, size, size, device=device)
    t = torch.tensor([50], device=device)
    with torch.no_grad():
        pred = model.forward(
            x,
            t,
            style_images=style.to(device),
            content_images=content.to(device),
            content_encoder_downsample_size=getattr(args, "content_encoder_downsample_size", 3),
            delta_images=delta.to(device),
        )
    log(f"E1 smoke pred shape={tuple(pred.shape)} finite={bool(torch.isfinite(pred).all())}")
    (REP / "e1_smoke.json").write_text(
        json.dumps(
            {
                "ok": bool(torch.isfinite(pred).all()),
                "shape": list(pred.shape),
                "device": device,
                "ckpt": str(CKPT),
                "note": "RSI structure = E_c(delta), style still E_s(R)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log("E1 smoke complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"E1 smoke FAILED: {type(e).__name__}: {e}")
        (REP / "e1_smoke.json").write_text(json.dumps({"ok": False, "error": str(e)}), encoding="utf-8")
        raise
