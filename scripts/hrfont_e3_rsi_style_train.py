#!/usr/bin/env python3
"""E3 control: same formal recipe as Stage A but RSI structure = Ec(style), not Δ.

Isolates gain from Δ wiring vs retrain alone. Init ft@25k, reinit RSI offset, 10k default.
Output: runs/e3_rsi_style_control/
"""
from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path("/root/projects/hrfont")
BANK = ROOT / "data/hrfont/e0_bank"
CONTENT_DIR = ROOT / "data/fontdiffuser/train/ContentImage"
CKPT = ROOT / "runs/ft_cnstyle/global_step_25000"
OUT = ROOT / "runs/e3_rsi_style_control"
REP = ROOT / "reports/hrfont_overnight"
REF8 = list("永和书风骨韵天地")
SIZE = 96
TAU = 0.07
TOP_M = 3
DROP_CFG = 0.1
MAX_STEPS = int(os.environ.get("HRFONT_E3_MAX", "10000"))
PERCEPTUAL_COEF = 0.01
OFFSET_COEF = 0.5


def _pick_gpu() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    cands = []
    for line in out.strip().splitlines():
        idx, free = [x.strip() for x in line.split(",")]
        i, fr = int(idx), int(free)
        if i == 1:
            continue
        cands.append((fr, i))
    cands.sort(reverse=True)
    if not cands or cands[0][0] < 8000:
        raise SystemExit(f"no GPU >=8GB free: {cands}")
    return cands[0][1]


if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pick_gpu())

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

META = json.loads((BANK / "meta.json").read_text())
META_U = json.loads((ROOT / "data/unified_v1/meta.json").read_text())
P1_CHARS = list(META_U["L_p1"])


def log(msg: str) -> None:
    REP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [E3] {msg}"
    print(line, flush=True)
    with (REP / "e3_rsi_style.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


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


def denorm_to_01(t: torch.Tensor) -> torch.Tensor:
    return (t * 0.5 + 0.5).clamp(0, 1)


def _load_pt(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def reinit_rsi_offset_modules(unet: nn.Module) -> int:
    from src.modules.attention import OffsetRefStrucInter

    n = 0
    for mod in unet.modules():
        if not isinstance(mod, OffsetRefStrucInter):
            continue
        n += 1
        for m in mod.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    return n


class E3Set(Dataset):
    """Same pairs as Stage A formal; RSI train target uses Ec(style) not Δ."""

    def __init__(self):
        demo = set(META.get("demo8", []))
        self.train = [f for f in META["train_fonts"] if f not in demo]
        self.chars = [c for c in P1_CHARS if (CONTENT_DIR / f"u{ord(c):04X}.jpg").exists()]
        self.content_cache = {ch: load_im(CONTENT_DIR / f"u{ord(ch):04X}.jpg") for ch in self.chars}
        pairs = []
        for f in self.train:
            if not any(png(f, r).exists() for r in REF8):
                continue
            for ch in self.chars:
                if png(f, ch).exists():
                    pairs.append((f, ch))
        self.pairs = pairs
        log(f"dataset pairs={len(pairs)} fonts={len(self.train)} chars={len(self.chars)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        font, ch = self.pairs[idx]
        r = random.choice([r for r in REF8 if png(font, r).exists()])
        return {
            "content": self.content_cache[ch].clone(),
            "style": load_im(png(font, r)),
            "target": load_im(png(font, ch)),
            "target_01": denorm_to_01(load_im(png(font, ch))),
        }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch], 0) for k in batch[0].keys()}


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("cuda required")
    if not (CKPT / "unet.pth").exists():
        raise SystemExit(f"missing {CKPT}")

    device = torch.device("cuda:0")
    log(f"E3 RSI←Ec(style) control max={MAX_STEPS} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    cfg = {
        "experiment": "E3_rsi_style_control",
        "rsi_structure": "Ec(style_image) — official ft wiring",
        "init": str(CKPT),
        "max_steps": MAX_STEPS,
        "drop_cfg": DROP_CFG,
        "content": "DejaVu ContentImage",
        "style": "bank r96 random ref8",
    }
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    sys.path.insert(0, str(ROOT / "code/FontDiffuser"))
    from configs.fontdiffuser import get_parser
    from src import build_content_encoder, build_ddpm_scheduler, build_style_encoder, build_unet
    from src.criterion import ContentPerceptualLoss
    from utils import normalize_mean_std, reNormalize_img, x0_from_epsilon

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

    ds = E3Set()
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate)

    unet = build_unet(args=args)
    se = build_style_encoder(args=args)
    ce = build_content_encoder(args=args)
    unet.load_state_dict(_load_pt(CKPT / "unet.pth"))
    se.load_state_dict(_load_pt(CKPT / "style_encoder.pth"))
    ce.load_state_dict(_load_pt(CKPT / "content_encoder.pth"))
    n_reinit = reinit_rsi_offset_modules(unet)
    log(f"reinit OffsetRefStrucInter: {n_reinit}")
    for p in list(se.parameters()) + list(ce.parameters()):
        p.requires_grad_(False)
    if hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    unet.to(device).train()
    se.to(device).eval()
    ce.to(device).eval()
    perceptual = ContentPerceptualLoss().to(device)
    for p in perceptual.parameters():
        p.requires_grad_(False)
    sched = build_ddpm_scheduler(args)
    n_t = int(getattr(sched.config, "num_train_timesteps", 1000))
    opt = torch.optim.AdamW((p for p in unet.parameters() if p.requires_grad), lr=1e-5, weight_decay=0.01)

    state = {"step": 0}

    def save(tag: str) -> None:
        blob = {"step": state["step"], "unet": unet.state_dict(), "config": cfg}
        torch.save(blob, OUT / "last.pt")
        if state["step"] >= MAX_STEPS or state["step"] % 5000 == 0:
            torch.save(blob, OUT / f"step_{state['step']}.pt")
        log(f"saved ({tag}) step={state['step']}")

    signal.signal(signal.SIGTERM, lambda *_: (save("sigterm"), sys.exit(0)))

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        amp_ctx = lambda: torch.amp.autocast("cuda", dtype=torch.float16)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        amp_ctx = lambda: torch.cuda.amp.autocast(dtype=torch.float16)

    t0 = time.time()
    while state["step"] < MAX_STEPS:
        for batch in dl:
            if state["step"] >= MAX_STEPS:
                break
            content = batch["content"].to(device)
            style = batch["style"].to(device)
            target = batch["target"].to(device)
            target_01 = batch["target_01"].to(device)
            if random.random() < DROP_CFG:
                content = torch.ones_like(content)
                style = torch.ones_like(style)
            noise = torch.randn_like(target)
            t = torch.randint(0, n_t, (1,), device=device).long()
            xt = sched.add_noise(target, noise, t)
            with torch.no_grad():
                style_feat, _, _ = se(style)
                b, c, h, w = style_feat.shape
                style_hidden = style_feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
                content_feat, content_res = ce(content)
                struct_feat, struct_res = ce(style)
            content_res = list(content_res) + [content_feat]
            struct_res = list(struct_res) + [struct_feat]
            hidden = [style_feat, content_res, style_hidden, struct_res]
            with amp_ctx():
                out = unet(xt, t, encoder_hidden_states=hidden, content_encoder_downsample_size=3)
                diff = F.mse_loss(out[0].float(), noise.float())
                offset_loss = out[1] / 2 if torch.is_tensor(out[1]) else torch.zeros((), device=device)
                pred_x0 = x0_from_epsilon(sched, out[0].float(), xt, t)
                percep = perceptual.calculate_loss(
                    normalize_mean_std(reNormalize_img(pred_x0)),
                    normalize_mean_std(target_01),
                    device=device,
                )
                loss = diff + PERCEPTUAL_COEF * percep + OFFSET_COEF * offset_loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            state["step"] += 1
            if state["step"] % 500 == 0:
                log(f"step={state['step']} loss={float(loss):.4f} dt={time.time()-t0:.0f}s")
                save("periodic")
    save("done")
    log("finished")


if __name__ == "__main__":
    main()
