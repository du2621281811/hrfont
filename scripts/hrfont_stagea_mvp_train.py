#!/usr/bin/env python3
"""Stage A MVP paired training: official Chinese-RSI control vs feature-delta RSI."""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("control", "delta"))
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=500)
    return parser.parse_args()


ARGS = parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path("/root/projects/hrfont")
DATA = ROOT / "data/fontdiffuser"
META = json.loads((ROOT / "data/retrain_v2/meta.json").read_text(encoding="utf-8"))
META_U = json.loads((ROOT / "data/unified_v1/meta.json").read_text(encoding="utf-8"))
FT = ROOT / "runs/ft_cnstyle/global_step_25000"
OUT = ARGS.output_dir or ROOT / "runs/stagea_mvp" / f"{ARGS.arm}_seed{ARGS.seed}"
SIZE = 96
STYLE_CHAR = "永"
DROP_CFG = 0.1
DROP_STRUCTURE = 0.25
PERCEPTUAL_COEF = 0.01
OFFSET_COEF = 0.5
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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{ARGS.arm}] {message}", flush=True)


def reset_rsi(unet: nn.Module) -> int:
    from src.modules.attention import OffsetRefStrucInter

    count = 0
    for module in unet.modules():
        if not isinstance(module, OffsetRefStrucInter):
            continue
        count += 1
        for child in module.modules():
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(child.weight)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
            elif isinstance(child, (nn.LayerNorm, nn.GroupNorm)):
                if child.weight is not None:
                    nn.init.ones_(child.weight)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
    return count


class PairedStageASet(Dataset):
    def __init__(self) -> None:
        self.fonts = list(META["train_fonts"])
        self.pairs: list[tuple[str, str]] = []
        for font in self.fonts:
            style_path = image_path(DATA, "train", "StyleImage", font, STYLE_CHAR)
            if not style_path.exists():
                continue
            for ch in P1_CHARS:
                content_path = image_path(DATA, "train", "ContentImage", "", ch)
                target_path = image_path(DATA, "train", "TargetImage", font, ch)
                if content_path.exists() and target_path.exists():
                    self.pairs.append((font, ch))
        if not self.pairs:
            raise RuntimeError("empty controlled Stage A dataset")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        font, ch = self.pairs[index]
        content = load_image(image_path(DATA, "train", "ContentImage", "", ch))
        target = load_image(image_path(DATA, "train", "TargetImage", font, ch))
        style = load_image(image_path(DATA, "train", "StyleImage", font, STYLE_CHAR))
        return {
            "font": font,
            "char": ch,
            "content": content,
            "style": style,
            "target": target,
            "target_01": (target * 0.5 + 0.5).clamp(0, 1),
        }


def collate(batch: list[dict]) -> dict:
    return {
        "font": [item["font"] for item in batch],
        "char": [item["char"] for item in batch],
        "content": torch.stack([item["content"] for item in batch]),
        "style": torch.stack([item["style"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "target_01": torch.stack([item["target_01"] for item in batch]),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    for name in ("unet.pth", "style_encoder.pth", "content_encoder.pth"):
        if not (FT / name).exists():
            raise FileNotFoundError(FT / name)
    if OUT.exists() and (OUT / "final.pt").exists():
        raise SystemExit(f"completed output already exists: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    torch.cuda.manual_seed_all(ARGS.seed)

    from configs.fontdiffuser import get_parser
    from src import build_content_encoder, build_ddpm_scheduler, build_style_encoder, build_unet
    from src.criterion import ContentPerceptualLoss
    from utils import normalize_mean_std, reNormalize_img, x0_from_epsilon

    fd_args = get_parser().parse_args([])
    fd_args.resolution = SIZE
    fd_args.content_image_size = (SIZE, SIZE)
    fd_args.style_image_size = (SIZE, SIZE)
    fd_args.unet_channels = (64, 128, 256, 512)
    fd_args.channel_attn = True
    fd_args.content_encoder_downsample_size = 3
    fd_args.content_start_channel = 64
    fd_args.style_start_channel = 64
    fd_args.beta_scheduler = "scaled_linear"

    device = torch.device("cuda:0")
    dataset = PairedStageASet()
    generator = torch.Generator()
    generator.manual_seed(ARGS.seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
    )

    unet = build_unet(args=fd_args)
    style_encoder = build_style_encoder(args=fd_args)
    content_encoder = build_content_encoder(args=fd_args)
    unet.load_state_dict(load_pt(FT / "unet.pth"))
    style_encoder.load_state_dict(load_pt(FT / "style_encoder.pth"))
    content_encoder.load_state_dict(load_pt(FT / "content_encoder.pth"))
    reset_count = reset_rsi(unet)

    for parameter in list(style_encoder.parameters()) + list(content_encoder.parameters()):
        parameter.requires_grad_(False)
    if hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()
    unet.to(device).train()
    style_encoder.to(device).eval()
    content_encoder.to(device).eval()

    prototypes = build_style_prototypes(
        style_encoder,
        DATA,
        "train",
        dataset.fonts,
        load_image,
        device,
    )
    if len(prototypes) != len(dataset.fonts):
        raise RuntimeError(f"single-ref prototype coverage {len(prototypes)}/{len(dataset.fonts)}")

    perceptual = ContentPerceptualLoss().to(device).eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad_(False)
    scheduler = build_ddpm_scheduler(fd_args)
    n_timesteps = int(getattr(scheduler.config, "num_train_timesteps", 1000))
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5, weight_decay=0.01)

    config = {
        "experiment": f"A-MVP-{ARGS.arm.upper()}",
        "arm": ARGS.arm,
        "seed": ARGS.seed,
        "max_steps": ARGS.max_steps,
        "data_root": str(DATA),
        "pairs": len(dataset),
        "fonts": len(dataset.fonts),
        "chars": len(P1_CHARS),
        "reference": STYLE_CHAR,
        "init": str(FT),
        "encoders_frozen": True,
        "rsi_reinitialized": reset_count,
        "drop_cfg": DROP_CFG,
        "drop_structure": DROP_STRUCTURE,
        "feature_mix": ARGS.arm == "delta",
        "top_m": 3 if ARGS.arm == "delta" else None,
        "tau": 0.07 if ARGS.arm == "delta" else None,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(OUT / "config.json", config)

    state = {"step": 0, "stop": False}
    last_path = OUT / "last.pt"
    if last_path.exists():
        blob = load_pt(last_path)
        unet.load_state_dict(blob["unet"])
        optimizer.load_state_dict(blob["optimizer"])
        state["step"] = int(blob["step"])
        log(f"resumed step={state['step']}")

    def status(extra: dict) -> None:
        write_json(
            OUT / "status.json",
            {
                "experiment": config["experiment"],
                "state": "running" if state["step"] < ARGS.max_steps else "completed",
                "step": state["step"],
                "max_steps": ARGS.max_steps,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                **extra,
            },
        )

    def save(path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "step": state["step"],
                "arm": ARGS.arm,
                "seed": ARGS.seed,
                "unet": unet.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
            },
            tmp,
        )
        tmp.replace(path)

    def stop_handler(_signum, _frame) -> None:
        state["stop"] = True

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        autocast = lambda: torch.amp.autocast("cuda", dtype=torch.float16)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        autocast = lambda: torch.cuda.amp.autocast(dtype=torch.float16)

    log(f"start pairs={len(dataset)} max_steps={ARGS.max_steps} gpu={ARGS.gpu}")
    status({"message": "started"})
    started = time.time()
    while state["step"] < ARGS.max_steps and not state["stop"]:
        for batch in loader:
            if state["step"] >= ARGS.max_steps or state["stop"]:
                break
            content = batch["content"].to(device, non_blocking=True)
            style = batch["style"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            target_01 = batch["target_01"].to(device, non_blocking=True)
            font, ch = batch["font"][0], batch["char"][0]

            cfg_drop = random.random() < DROP_CFG
            structure_drop = random.random() < DROP_STRUCTURE
            if cfg_drop:
                content = torch.ones_like(content)
                style = torch.ones_like(style)

            noise = torch.randn_like(target)
            timestep = torch.randint(0, n_timesteps, (1,), device=device).long()
            noisy_target = scheduler.add_noise(target, noise, timestep)

            with torch.no_grad():
                style_feat, _, _ = style_encoder(style)
                b, channels, height, width = style_feat.shape
                style_hidden = style_feat.permute(0, 2, 3, 1).reshape(
                    b, height * width, channels
                )
                if ARGS.arm == "control":
                    content_feat, content_res = content_encoder(content)
                    struct_feat, struct_res = content_encoder(style)
                    content_all = list(content_res) + [content_feat]
                    structure_all = list(struct_res) + [struct_feat]
                else:
                    candidates = [
                        candidate
                        for candidate in dataset.fonts
                        if candidate != font
                        and image_path(DATA, "train", "TargetImage", candidate, ch).exists()
                    ]
                    alpha = alpha_topk(prototypes[font], prototypes, candidates)
                    content_all, structure_all = feature_delta(
                        content_encoder,
                        content,
                        alpha,
                        data_root=DATA,
                        split="train",
                        ch=ch,
                        load_image=load_image,
                        device=device,
                    )
                if structure_drop or (cfg_drop and ARGS.arm == "delta"):
                    structure_all = [torch.zeros_like(value) for value in structure_all]
                hidden = [style_feat, content_all, style_hidden, structure_all]

            with autocast():
                prediction, offset_sum = unet(
                    noisy_target,
                    timestep,
                    encoder_hidden_states=hidden,
                    content_encoder_downsample_size=3,
                )
                diff_loss = F.mse_loss(prediction.float(), noise.float())
                offset_loss = (
                    offset_sum / 2
                    if torch.is_tensor(offset_sum)
                    else torch.zeros((), device=device)
                )
                pred_x0 = x0_from_epsilon(
                    scheduler, prediction.float(), noisy_target, timestep
                )
                perceptual_loss = perceptual.calculate_loss(
                    generated_images=normalize_mean_std(reNormalize_img(pred_x0)),
                    target_images=normalize_mean_std(target_01),
                    device=device,
                )
                loss = (
                    diff_loss
                    + PERCEPTUAL_COEF * perceptual_loss
                    + OFFSET_COEF * offset_loss
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            state["step"] += 1

            if state["step"] % 20 == 0 or state["step"] == 1:
                elapsed = time.time() - started
                log(
                    f"step={state['step']} loss={float(loss):.4f} "
                    f"diff={float(diff_loss):.4f} dt={elapsed:.0f}s"
                )
                status(
                    {
                        "loss": float(loss),
                        "diff_loss": float(diff_loss),
                        "elapsed_seconds": round(elapsed, 1),
                    }
                )
            if state["step"] % ARGS.save_every == 0:
                save(last_path)

    save(last_path)
    if state["step"] >= ARGS.max_steps:
        save(OUT / "final.pt")
        status({"message": "completed", "elapsed_seconds": round(time.time() - started, 1)})
        log(f"completed step={state['step']}")
    else:
        status({"message": "stopped", "elapsed_seconds": round(time.time() - started, 1)})
        log(f"stopped step={state['step']}")


if __name__ == "__main__":
    main()
