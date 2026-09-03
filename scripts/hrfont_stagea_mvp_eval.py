#!/usr/bin/env python3
"""Unified paired evaluation for Stage A MVP control and feature-delta arms."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("control", "delta"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--char-codes", type=str, default="")
    parser.add_argument("--pred-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


ARGS = parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")

import numpy as np
import torch
import torch.nn as nn
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms

ROOT = Path("/root/projects/hrfont")
DATA = ROOT / "data/fontdiffuser"
META = json.loads((ROOT / "data/retrain_v2/meta.json").read_text(encoding="utf-8"))
META_U = json.loads((ROOT / "data/unified_v1/meta.json").read_text(encoding="utf-8"))
FT = ROOT / "runs/ft_cnstyle/global_step_25000"
GT_ROOT = ROOT / "data/unified_v1/renders/128/gt_latin"
OUT = ARGS.output or ARGS.checkpoint.parent / "metrics.json"
SIZE = 96
STYLE_CHAR = "永"
P1_CHARS = list(META_U["L_p1"])

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "code/FontDiffuser"))
from hrfont_delta_feature import (  # noqa: E402
    alpha_topk,
    build_style_prototypes,
    feature_delta,
    image_path,
)

TFM = transforms.Compose(
    [
        transforms.Resize((SIZE, SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def load_image(path: Path) -> torch.Tensor:
    return TFM(Image.open(path).convert("RGB"))


def load_pt(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(
        image.convert("L").resize((SIZE, SIZE), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0


class ControlledDPM(nn.Module):
    def __init__(self, unet, style_encoder, content_encoder, arm: str):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.arm = arm
        self.delta_features: list[torch.Tensor] | None = None

    @property
    def device(self):
        return next(self.parameters()).device

    def set_delta(self, features: list[torch.Tensor]) -> None:
        self.delta_features = features

    def forward(self, x_t, timesteps, cond, content_encoder_downsample_size, version):
        content_images, style_images = cond[0], cond[1]
        style_feat, _, _ = self.style_encoder(style_images)
        batch, channels, height, width = style_feat.shape
        style_hidden = style_feat.permute(0, 2, 3, 1).reshape(
            batch, height * width, channels
        )
        content_feat, content_res = self.content_encoder(content_images)
        content_all = list(content_res) + [content_feat]
        if self.arm == "control":
            struct_feat, struct_res = self.content_encoder(style_images)
            structure_all = list(struct_res) + [struct_feat]
        else:
            if self.delta_features is None:
                raise RuntimeError("delta features not set")
            blank_content = content_images.mean(dim=(1, 2, 3)) > 0.95
            blank_style = style_images.mean(dim=(1, 2, 3)) > 0.95
            unconditional = (blank_content & blank_style).view(batch, 1, 1, 1)
            structure_all = []
            for value in self.delta_features:
                if value.shape[0] == 1 and batch > 1:
                    value = value.expand(batch, -1, -1, -1)
                if value.shape[0] != batch:
                    raise RuntimeError(
                        f"delta batch mismatch: {value.shape[0]} != {batch}"
                    )
                structure_all.append(value.masked_fill(unconditional, 0))
        hidden = [style_feat, content_all, style_hidden, structure_all]
        return self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=hidden,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )[0]


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
    args.algorithm_type = "dpmsolver++"
    args.num_inference_steps = 20
    args.order = 2
    args.method = "multistep"
    return args


def main() -> None:
    if not ARGS.checkpoint.exists():
        raise FileNotFoundError(ARGS.checkpoint)
    device = torch.device("cuda:0")
    args = make_args()

    from src import build_content_encoder, build_style_encoder, build_unet
    from src.dpm_solver.pipeline_dpm_solver import FontDiffuserDPMPipeline
    from src import build_ddpm_scheduler

    checkpoint = load_pt(ARGS.checkpoint)
    if checkpoint.get("arm") != ARGS.arm:
        raise RuntimeError(
            f"checkpoint arm={checkpoint.get('arm')} but requested {ARGS.arm}"
        )
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    unet.load_state_dict(checkpoint["unet"])
    style_encoder.load_state_dict(load_pt(FT / "style_encoder.pth"))
    content_encoder.load_state_dict(load_pt(FT / "content_encoder.pth"))
    unet.to(device).eval()
    style_encoder.to(device).eval()
    content_encoder.to(device).eval()

    train_fonts = list(META["train_fonts"])
    test_fonts = list(META["test_fonts"])
    train_prototypes = build_style_prototypes(
        style_encoder, DATA, "train", train_fonts, load_image, device
    )
    test_prototypes = build_style_prototypes(
        style_encoder, DATA, "val", test_fonts, load_image, device
    )
    model = ControlledDPM(unet, style_encoder, content_encoder, ARGS.arm)
    pipeline = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=build_ddpm_scheduler(args),
        guidance_type="classifier-free",
        guidance_scale=7.5,
    )
    model.to(device).eval()

    jobs = [(font, ch) for font in test_fonts for ch in P1_CHARS]
    if ARGS.char_codes:
        selected_chars = {
            chr(int(code.strip(), 16))
            for code in ARGS.char_codes.split(",")
            if code.strip()
        }
        jobs = [(font, ch) for font, ch in jobs if ch in selected_chars]
    if ARGS.limit:
        jobs = jobs[: ARGS.limit]
    rows: list[dict] = []
    started = time.time()
    for index, (font, ch) in enumerate(jobs):
        content_path = image_path(DATA, "val", "ContentImage", "", ch)
        style_path = image_path(DATA, "val", "StyleImage", font, STYLE_CHAR)
        target_path = image_path(DATA, "val", "TargetImage", font, ch)
        gt_path = target_path
        if not gt_path.exists():
            gt_name = ch if ch.isalnum() and ord(ch) < 128 else f"u{ord(ch):04X}"
            gt_path = GT_ROOT / font / f"{gt_name}.png"
        if not (content_path.exists() and style_path.exists() and gt_path.exists()):
            continue
        content = load_image(content_path).unsqueeze(0).to(device)
        style = load_image(style_path).unsqueeze(0).to(device)
        if ARGS.arm == "delta":
            candidates = [
                candidate
                for candidate in train_fonts
                if image_path(DATA, "train", "TargetImage", candidate, ch).exists()
            ]
            if candidates:
                alpha = alpha_topk(test_prototypes[font], train_prototypes, candidates)
                _, delta = feature_delta(
                    content_encoder,
                    content,
                    alpha,
                    data_root=DATA,
                    split="train",
                    ch=ch,
                    load_image=load_image,
                    device=device,
                )
            else:
                content_feat, content_res = content_encoder(content)
                delta = [
                    torch.zeros_like(value)
                    for value in list(content_res) + [content_feat]
                ]
            model.set_delta(delta)
        set_seed(ARGS.seed + index)
        prediction = pipeline.generate(
            content_images=content,
            style_images=style,
            batch_size=1,
            order=2,
            num_inference_step=20,
            content_encoder_downsample_size=3,
            dm_size=(SIZE, SIZE),
            algorithm_type="dpmsolver++",
            method="multistep",
        )[0]
        if ARGS.pred_dir is not None:
            ARGS.pred_dir.mkdir(parents=True, exist_ok=True)
            prediction.save(ARGS.pred_dir / f"{font}__u{ord(ch):04X}.png")
        l1 = float(
            np.mean(
                np.abs(
                    image_array(prediction)
                    - image_array(Image.open(gt_path).convert("RGB"))
                )
            )
        )
        rows.append({"font": font, "char": ch, "l1": l1})
        if (index + 1) % 50 == 0:
            print(
                f"{time.strftime('%H:%M:%S')} {ARGS.arm} "
                f"{index + 1}/{len(jobs)} mean_l1={np.mean([r['l1'] for r in rows]):.4f}",
                flush=True,
            )

    by_font = {}
    for font in test_fonts:
        values = [row["l1"] for row in rows if row["font"] == font]
        if values:
            by_font[font] = {"n": len(values), "L1_mean": float(np.mean(values))}
    payload = {
        "experiment": f"A-MVP-{ARGS.arm.upper()}",
        "arm": ARGS.arm,
        "checkpoint": str(ARGS.checkpoint),
        "step": int(checkpoint["step"]),
        "protocol": "pre-rendered inputs/GT, Demo-8 P1, strict 1-shot 永, DPM++20, CFG7.5",
        "seed": ARGS.seed,
        "n": len(rows),
        "L1_mean": float(np.mean([row["l1"] for row in rows])) if rows else None,
        "seconds": round(time.time() - started, 1),
        "by_font": by_font,
        "rows": rows,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(OUT.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT)
    print(f"wrote {OUT} L1={payload['L1_mean']} n={payload['n']}", flush=True)


if __name__ == "__main__":
    main()
