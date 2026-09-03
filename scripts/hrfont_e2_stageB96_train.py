#!/usr/bin/env python3
"""Stage B 96 — train SupportAdapter only; freeze Stage A UNet + encoders.

§6.2 B + §4.5.4: gap-gated support, 20% CFG drop support.
Init: best Stage A ckpt from runs/e2_stageA96_formal/
Stop: reports/hrfont_overnight/STOP_STAGE_B
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
STAGEA_OUT = ROOT / "runs/e2_stageA96_formal"
OUT = ROOT / "runs/e2_stageB96"
REP = ROOT / "reports/hrfont_overnight"
REF8 = list("永和书风骨韵天地")
SIZE = 96
TOP_M = 3
DROP_SUPPORT = 0.2
MAX_STEPS = int(os.environ.get("HRFONT_STAGE_B_MAX", "30000"))
LR = 1e-4
STAGEA_CKPT = Path(os.environ.get("HRFONT_STAGEA_CKPT", str(STAGEA_OUT / "last.pt")))


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

sys.path.insert(0, str(ROOT / "scripts"))
from hrfont_support_adapter import SupportAdapter
from hrfont_support_utils import alpha_top_m, gap_for_char, mmr_select, pick_support

META = json.loads((BANK / "meta.json").read_text(encoding="utf-8"))
META_U = json.loads((ROOT / "data/unified_v1/meta.json").read_text(encoding="utf-8"))
P1_CHARS = list(META_U["L_p1"])


def log(msg: str) -> None:
    REP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with (REP / "e2_stageB.log").open("a", encoding="utf-8") as f:
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


def _load_pt(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class StageBSet(Dataset):
    def __init__(self):
        demo = set(META.get("demo8", []))
        self.train = [f for f in META["train_fonts"] if f not in demo]
        self.chars = [c for c in P1_CHARS if (CONTENT_DIR / f"u{ord(c):04X}.jpg").exists()]
        cache = _load_pt(BANK / "cache/ec_es_r96.pt")
        self.proto = cache["style_proto"]
        self.content_cache: dict[str, torch.Tensor] = {}
        for ch in self.chars:
            self.content_cache[ch] = load_im(CONTENT_DIR / f"u{ord(ch):04X}.jpg")
        pairs = []
        for f in self.train:
            if not any(png(f, r).exists() for r in REF8):
                continue
            for ch in self.chars:
                if not png(f, ch).exists():
                    continue
                others = [g for g in self.train if g != f and g in self.proto and png(g, ch).exists()]
                if not others:
                    continue
                pairs.append((f, ch))
        self.pairs = pairs
        log(f"StageB dataset pairs={len(pairs)} gamma-gated support")

    def __len__(self):
        return len(self.pairs)

    def _alpha_mix_delta(self, font: str, ch: str) -> torch.Tensor:
        others = [g for g in self.train if g != font and g in self.proto and png(g, ch).exists()]
        mix = alpha_top_m(font, others)
        delta = None
        for g, w in mix:
            im = load_im(png(g, ch))
            delta = im * w if delta is None else delta + im * w
        return delta

    def __getitem__(self, idx):
        font, ch = self.pairs[idx]
        refs = [r for r in REF8 if png(font, r).exists()]
        r = random.choice(refs)
        spec = pick_support(font, ch, self.train, png)
        support_paths = spec.get("support_imgs") or []
        support_imgs = [load_im(p) for p in support_paths[: TOP_M * 3]]
        return {
            "content": self.content_cache[ch].clone(),
            "style": load_im(png(font, r)),
            "target": load_im(png(font, ch)),
            "delta": self._alpha_mix_delta(font, ch),
            "support": support_imgs,
            "gap": spec.get("gap", 0.0),
            "n_support": len(support_imgs),
        }


def collate(batch):
    return {
        "content": torch.stack([b["content"] for b in batch], 0),
        "style": torch.stack([b["style"] for b in batch], 0),
        "target": torch.stack([b["target"] for b in batch], 0),
        "delta": torch.stack([b["delta"] for b in batch], 0),
        "support": [b["support"] for b in batch],
        "gap": [b["gap"] for b in batch],
        "n_support": [b["n_support"] for b in batch],
    }


def pool_ec(feat, res_list) -> torch.Tensor:
    """GAP each scale then concat -> (D,) vector."""
    parts = []
    for x in list(res_list) + [feat]:
        if x.dim() == 4:
            parts.append(x.mean(dim=(2, 3)).squeeze(0))
        else:
            parts.append(x.flatten())
    return torch.cat(parts, dim=0)


def build_hidden(se, ce, content, style, delta_img, support_imgs, support_adapter, drop_support: bool):
    with torch.no_grad():
        style_feat, _, _ = se(style)
        b, c, h, w = style_feat.shape
        style_hidden = style_feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
        content_feat, content_res = ce(content)
        content_res = list(content_res) + [content_feat]
        mix_feat, mix_res = ce(delta_img)
        mix_res = list(mix_res) + [mix_feat]
        delta_res = [m - c_ for m, c_ in zip(mix_res, content_res)]

    style_hidden = style_hidden.detach()
    if (not drop_support) and support_imgs and support_imgs[0]:
        vecs = []
        for im in support_imgs[0]:
            im = im.unsqueeze(0).to(content.device)
            with torch.no_grad():
                f, rs = ce(im)
                vec = pool_ec(f, list(rs))
            vecs.append(vec.unsqueeze(0))
        if vecs:
            sup = torch.cat(vecs, dim=0).unsqueeze(0)
            sup_tok = support_adapter(sup)
            style_hidden = torch.cat([style_hidden, sup_tok], dim=1)

    return [style_feat, content_res, style_hidden, delta_res]


def write_status(extra: dict) -> None:
    st = {
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train": "e2_stageB96",
        "stageA_init": str(STAGEA_CKPT),
        "max_steps": MAX_STEPS,
        **extra,
    }
    (REP / "STATUS_STAGE_B.json").write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    if not STAGEA_CKPT.exists():
        raise SystemExit(f"missing Stage A ckpt {STAGEA_CKPT}")
    device = torch.device("cuda:0")
    log(f"Stage B init={STAGEA_CKPT} max={MAX_STEPS} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    sys.path.insert(0, str(ROOT / "code/FontDiffuser"))
    from configs.fontdiffuser import get_parser
    from src import build_content_encoder, build_ddpm_scheduler, build_style_encoder, build_unet

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

    ds = StageBSet()
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate)

    unet = build_unet(args=args)
    se = build_style_encoder(args=args)
    ce = build_content_encoder(args=args)
    blob = _load_pt(STAGEA_CKPT)
    unet.load_state_dict(blob["unet"])

    for p in unet.parameters():
        p.requires_grad_(False)
    for p in se.parameters():
        p.requires_grad_(False)
    for p in ce.parameters():
        p.requires_grad_(False)
    unet.eval()
    se.eval()
    ce.eval()
    ce.to(device)
    se.to(device)

    with torch.no_grad():
        f, rs = ce(torch.randn(1, 3, SIZE, SIZE, device=device))
        in_dim = pool_ec(f, list(rs)).numel()
    with torch.no_grad():
        style_dim = se(torch.randn(1, 3, SIZE, SIZE, device=device))[0].shape[1]
    support_adapter = SupportAdapter(in_dim, style_dim).to(device)

    sched = build_ddpm_scheduler(args)
    n_t = int(getattr(sched.config, "num_train_timesteps", 1000))
    opt = torch.optim.AdamW(support_adapter.parameters(), lr=LR, weight_decay=0.01)

    state = {"step": 0}
    ck = OUT / "last.pt"
    if ck.exists():
        bb = _load_pt(ck)
        if "support_adapter" in bb:
            support_adapter.load_state_dict(bb["support_adapter"])
        if "opt" in bb:
            try:
                opt.load_state_dict(bb["opt"])
            except Exception:
                pass
        state["step"] = int(bb.get("step", 0))
        log(f"resume Stage B step={state['step']}")

    ce.to(device)
    se.to(device)
    unet.to(device)
    support_adapter.train()
    unet.to(device)

    stop_file = REP / "STOP_STAGE_B"

    def save(tag: str) -> None:
        blob = {
            "step": state["step"],
            "unet": unet.state_dict(),
            "support_adapter": support_adapter.state_dict(),
            "opt": opt.state_dict(),
            "stageA_ckpt": str(STAGEA_CKPT),
            "config": {"drop_support": DROP_SUPPORT, "max_steps": MAX_STEPS},
        }
        torch.save(blob, OUT / "last.pt")
        if state["step"] % 5000 == 0 or state["step"] >= MAX_STEPS:
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
        if stop_file.exists() or (REP / "STOP_ALL").exists():
            save("STOP")
            return
        for batch in dl:
            if state["step"] >= MAX_STEPS or stop_file.exists():
                break
            content = batch["content"].to(device)
            style = batch["style"].to(device)
            target = batch["target"].to(device)
            delta = batch["delta"].to(device)
            drop_sup = random.random() < DROP_SUPPORT
            # Low-gap (n_support==0) → build_hidden naturally omits support (same as Stage A).
            # CFG 20% drop → train without support tokens (uncond branch), still update noise loss.

            noise = torch.randn_like(target)
            t = torch.randint(0, n_t, (1,), device=device).long()
            xt = sched.add_noise(target, noise, t)

            hidden = build_hidden(
                se, ce, content, style, delta, batch["support"], support_adapter, drop_sup
            )
            with amp_ctx():
                out = unet(xt, t, encoder_hidden_states=hidden, content_encoder_downsample_size=3)
                loss = F.mse_loss(out[0].float(), noise.float())
            opt.zero_grad(set_to_none=True)
            if loss.requires_grad:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(support_adapter.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            # No support / CFG drop: frozen UNet path only — forward step, no adapter grad.

            state["step"] += 1
            if state["step"] % 100 == 0:
                write_status({"step": state["step"], "loss": float(loss.item())})
            if state["step"] % 500 == 0:
                save("interval")
                log(f"step={state['step']} loss={loss.item():.4f} elapsed={(time.time()-t0)/3600:.2f}h")

    save("done")
    log("Stage B complete")


if __name__ == "__main__":
    main()
