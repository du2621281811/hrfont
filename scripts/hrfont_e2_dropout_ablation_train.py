#!/usr/bin/env python3
"""Dropout ablation: short Stage A runs varying CFG / Δ drop (GPU 2/3).

Same recipe as formal v2 except dropout knobs. Default 10k steps for fair compare.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
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
ABL_ROOT = ROOT / "runs/dropout_ablation"
REP = ROOT / "reports/hrfont_overnight/dropout_ablation"
REF8 = list("永和书风骨韵天地")
SIZE = 96
TAU = 0.07
TOP_M = 3
PERCEPTUAL_COEF = 0.01
OFFSET_COEF = 0.5
META_U = json.loads((ROOT / "data/unified_v1/meta.json").read_text())
P1_CHARS = list(META_U["L_p1"])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="e.g. R1_d01")
    ap.add_argument("--drop-cfg", type=float, default=0.1)
    ap.add_argument("--drop-delta", type=float, default=0.25)
    ap.add_argument(
        "--drop-mode",
        choices=["both", "style_only"],
        default="both",
        help="both=official CFG blank content+style; style_only=v1 overnight",
    )
    ap.add_argument("--max-steps", type=int, default=10_000)
    return ap.parse_args()


ARGS = parse_args()
OUT = ABL_ROOT / ARGS.run_id
DROP_CFG = ARGS.drop_cfg
DROP_DELTA = ARGS.drop_delta
DROP_MODE = ARGS.drop_mode
MAX_STEPS = ARGS.max_steps


def log(msg: str) -> None:
    REP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{ARGS.run_id}] {time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with (REP / "ablation.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

META = json.loads((BANK / "meta.json").read_text())


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


class StageASet(Dataset):
    def __init__(self):
        demo = set(META.get("demo8", []))
        self.train = [f for f in META["train_fonts"] if f not in demo]
        self.chars = [c for c in P1_CHARS if (CONTENT_DIR / f"u{ord(c):04X}.jpg").exists()]
        cache = _load_pt(BANK / "cache/ec_es_r96.pt")
        self.proto = cache["style_proto"]
        self.content_cache = {
            ch: load_im(CONTENT_DIR / f"u{ord(ch):04X}.jpg") for ch in self.chars
        }
        pairs = []
        for f in self.train:
            if not any(png(f, r).exists() for r in REF8):
                continue
            for ch in self.chars:
                if not png(f, ch).exists():
                    continue
                others = [g for g in self.train if g != f and g in self.proto and png(g, ch).exists()]
                if others:
                    pairs.append((f, ch))
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def _alpha(self, font: str, others: list[str]) -> list[tuple[str, float]]:
        q = self.proto[font].float().flatten()
        scores = [(g, float(torch.cosine_similarity(q[None], self.proto[g].float().flatten()[None]).item())) for g in others]
        scores.sort(key=lambda x: -x[1])
        top = scores[:TOP_M]
        logits = torch.tensor([s / TAU for _, s in top])
        w = torch.softmax(logits, dim=0)
        return [(top[i][0], float(w[i])) for i in range(len(top))]

    def __getitem__(self, idx):
        font, ch = self.pairs[idx]
        r = random.choice([r for r in REF8 if png(font, r).exists()])
        others = [g for g in self.train if g != font and g in self.proto and png(g, ch).exists()]
        mix = self._alpha(font, others)
        delta = None
        for g, w in mix:
            im = load_im(png(g, ch))
            delta = im * w if delta is None else delta + im * w
        return {
            "content": self.content_cache[ch].clone(),
            "style": load_im(png(font, r)),
            "target": load_im(png(font, ch)),
            "target_01": denorm_to_01(load_im(png(font, ch))),
            "delta": delta,
        }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch], 0) for k in batch[0].keys()}


def apply_cfg_dropout(content, style):
    if random.random() >= DROP_CFG:
        return content, style
    if DROP_MODE == "both":
        return torch.ones_like(content), torch.ones_like(style)
    return content, torch.ones_like(style)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("cuda required")
    device = torch.device("cuda:0")
    log(
        f"start drop_cfg={DROP_CFG} drop_delta={DROP_DELTA} mode={DROP_MODE} "
        f"max={MAX_STEPS} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    meta = {
        "run_id": ARGS.run_id,
        "drop_cfg": DROP_CFG,
        "drop_delta": DROP_DELTA,
        "drop_mode": DROP_MODE,
        "max_steps": MAX_STEPS,
        "content": "DejaVu",
    }
    (OUT / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

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

    ds = StageASet()
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate)

    unet = build_unet(args=args)
    se = build_style_encoder(args=args)
    ce = build_content_encoder(args=args)
    unet.load_state_dict(_load_pt(CKPT / "unet.pth"))
    se.load_state_dict(_load_pt(CKPT / "style_encoder.pth"))
    ce.load_state_dict(_load_pt(CKPT / "content_encoder.pth"))
    reinit_rsi_offset_modules(unet)
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

    state = {"step": 0, "stop": False}

    def save(tag: str) -> None:
        blob = {"step": state["step"], "unet": unet.state_dict(), "config": meta}
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
            delta_img = batch["delta"].to(device)
            content, style = apply_cfg_dropout(content, style)
            drop_delta = random.random() < DROP_DELTA
            noise = torch.randn_like(target)
            t = torch.randint(0, n_t, (1,), device=device).long()
            xt = sched.add_noise(target, noise, t)
            with torch.no_grad():
                style_feat, _, _ = se(style)
                b, c, h, w = style_feat.shape
                style_hidden = style_feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
                content_feat, content_res = ce(content)
                mix_feat, mix_res = ce(delta_img)
            content_res = list(content_res) + [content_feat]
            mix_res = list(mix_res) + [mix_feat]
            delta_res = [torch.zeros_like(x) for x in mix_res] if drop_delta else [m - c_ for m, c_ in zip(mix_res, content_res)]
            hidden = [style_feat, content_res, style_hidden, delta_res]
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
