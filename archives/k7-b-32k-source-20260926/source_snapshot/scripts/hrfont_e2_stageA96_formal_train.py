#!/usr/bin/env python3
"""Stage A 96 — formal v2 (idea validation vs ft_cnstyle).

Aligns with ft_cnstyle / retrain_v2 PROTOCOL for fair comparison:
  - Content = DejaVu (same ContentImage/source as ft_cnstyle)
  - Loss = noise MSE + 0.01 VGG perceptual + 0.5 offset (official coeffs)
  - CFG train dropout: content+style @ 0.1; Δ branch zero @ 0.25
  - Re-init RSI OffsetRefStrucInter modules (Δ wiring; §5.4b)
  - Same init: ft_cnstyle@25k; frozen encoders; Δ-RSI hidden states

Output: runs/e2_stageA96_formal/
Stop:   reports/hrfont_overnight/STOP_FORMAL
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
OUT = ROOT / "runs/e2_stageA96_formal"
REP = ROOT / "reports/hrfont_overnight"
REF8 = list("永和书风骨韵天地")
SIZE = 96
TAU = 0.07
TOP_M = 3
DROP_CFG = 0.1
DROP_DELTA = 0.25
MAX_STEPS = 80_000
PERCEPTUAL_COEF = 0.01
OFFSET_COEF = 0.5
MAX_CKPT_KEEP = 4
OOM_EXIT = 42


def _pick_gpu_id() -> int:
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
        raise SystemExit(f"no GPU with >=8GB free among {{0,2,3}}: {cands}")
    return cands[0][1]


if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pick_gpu_id())

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
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with (REP / "e2_stageA_formal.log").open("a", encoding="utf-8") as f:
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
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
    return n


class StageAFormalSet(Dataset):
    """P1 Latin targets; DejaVu content (ft_cnstyle); bank Δ mix; Demo-8 excluded."""

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
        log(
            f"formal dataset pairs={len(self.pairs)} fonts={len(self.train)} "
            f"p1_chars={len(self.chars)} content=DejaVu/ContentImage demo8_excluded={len(demo)}"
        )

    def __len__(self):
        return len(self.pairs)

    def _alpha(self, font: str, others: list[str]) -> list[tuple[str, float]]:
        q = self.proto[font].float().flatten()
        scores = []
        for g in others:
            v = self.proto[g].float().flatten()
            scores.append((g, float(F.cosine_similarity(q[None], v[None]).item())))
        scores.sort(key=lambda x: -x[1])
        top = scores[:TOP_M]
        logits = torch.tensor([s / TAU for _, s in top])
        w = torch.softmax(logits, dim=0)
        return [(top[i][0], float(w[i])) for i in range(len(top))]

    def __getitem__(self, idx):
        font, ch = self.pairs[idx]
        refs = [r for r in REF8 if png(font, r).exists()]
        r = random.choice(refs)
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
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], 0) for k in keys}


def latest_ckpt() -> Path | None:
    last = OUT / "last.pt"
    if last.exists():
        return last
    cks = sorted(OUT.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return cks[-1] if cks else None


def prune_ckpts() -> None:
    cks = sorted(OUT.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    keep = set()
    for p in cks[-MAX_CKPT_KEEP:]:
        keep.add(p)
    for p in cks:
        if int(p.stem.split("_")[1]) % 5000 == 0:
            keep.add(p)
    for p in cks:
        if p not in keep:
            try:
                p.unlink()
            except OSError:
                pass


def write_status(extra: dict) -> None:
    gpu = ""
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception:
        pass
    cks = sorted(OUT.glob("step_*.pt"))
    st = {
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train": "e2_stageA96_formal",
        "protocol": "DejaVu content + VGG + offset reinit + CFG0.1",
        "train_ckpt": str(cks[-1]) if cks else (str(OUT / "last.pt") if (OUT / "last.pt").exists() else None),
        "n_ckpt": len(cks),
        "max_steps": MAX_STEPS,
        "gpu": gpu.splitlines() if gpu else [],
        **extra,
    }
    (REP / "STATUS_FORMAL.json").write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")


def _opt_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def main() -> None:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    log(f"formal v2 CUDA_VISIBLE_DEVICES={vis}")
    if not torch.cuda.is_available():
        raise SystemExit("cuda not available")
    if not (CKPT / "unet.pth").exists():
        raise SystemExit(f"missing ckpt {CKPT}")
    if not CONTENT_DIR.is_dir():
        raise SystemExit(f"missing DejaVu ContentImage {CONTENT_DIR}")

    sys.path.insert(0, str(ROOT / "code/ours/FontDiffuser"))
    from configs.fontdiffuser import get_parser
    from src import build_content_encoder, build_ddpm_scheduler, build_style_encoder, build_unet
    from src.criterion import ContentPerceptualLoss
    from utils import normalize_mean_std, reNormalize_img, x0_from_epsilon

    device = torch.device("cuda:0")
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

    ds = StageAFormalSet()
    if len(ds) < 10:
        raise SystemExit("dataset too small")
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate, pin_memory=True)

    unet = build_unet(args=args)
    se = build_style_encoder(args=args)
    ce = build_content_encoder(args=args)
    unet.load_state_dict(_load_pt(CKPT / "unet.pth"))
    se.load_state_dict(_load_pt(CKPT / "style_encoder.pth"))
    ce.load_state_dict(_load_pt(CKPT / "content_encoder.pth"))
    n_reinit = reinit_rsi_offset_modules(unet)
    log(f"reinit OffsetRefStrucInter modules: {n_reinit}")

    se.eval()
    ce.eval()
    for p in list(se.parameters()) + list(ce.parameters()):
        p.requires_grad_(False)
    try:
        if hasattr(unet, "enable_gradient_checkpointing"):
            unet.enable_gradient_checkpointing()
            log("gradient checkpointing on")
    except Exception as e:
        log(f"gradient checkpointing skipped: {e}")

    perceptual = ContentPerceptualLoss().to(device)
    for p in perceptual.parameters():
        p.requires_grad_(False)

    sched = build_ddpm_scheduler(args)
    n_t = int(getattr(sched.config, "num_train_timesteps", getattr(sched, "num_train_timesteps", 1000)))
    opt = torch.optim.AdamW((p for p in unet.parameters() if p.requires_grad), lr=1e-5, weight_decay=0.01)

    state = {"step": 0, "stop": False}
    ck = latest_ckpt()
    if ck is not None:
        blob = _load_pt(ck)
        unet.load_state_dict(blob["unet"])
        if "opt" in blob:
            try:
                opt.load_state_dict(blob["opt"])
                _opt_to_device(opt, device)
            except Exception as e:
                log(f"opt state load skipped: {e}")
        state["step"] = int(blob.get("step", 0))
        log(f"resume {ck} step={state['step']}")

    unet.to(device)
    se.to(device)
    ce.to(device)

    def save_now(tag: str) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        blob = {"step": state["step"], "unet": unet.state_dict(), "opt": opt.state_dict()}
        tmp = OUT / "last.pt.tmp"
        torch.save(blob, tmp)
        tmp.replace(OUT / "last.pt")
        step = state["step"]
        if step > 0 and step % 500 == 0:
            path = OUT / f"step_{step}.pt"
            torch.save(blob, path)
            prune_ckpts()
            log(f"saved {path}")
        else:
            log(f"saved last.pt ({tag})")

    def _handle(signum, _frame):
        log(f"signal {signum} — save and exit")
        state["stop"] = True
        try:
            save_now(f"signal{signum}")
        except Exception as e:
            log(f"save on signal failed: {e}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        amp_ctx = lambda: torch.amp.autocast("cuda", dtype=torch.float16)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        amp_ctx = lambda: torch.cuda.amp.autocast(dtype=torch.float16)

    log(f"formal train start pairs={len(ds)} step={state['step']} max={MAX_STEPS}")
    write_status({"step": state["step"], "phase": "start"})
    unet.train()
    se.eval()
    ce.eval()
    oom_streak = 0
    t0 = time.time()
    stop_file = REP / "STOP_FORMAL"

    while not state["stop"] and state["step"] < MAX_STEPS:
        if stop_file.exists():
            log("STOP_FORMAL — save and exit")
            save_now("STOP")
            return
        for batch in dl:
            if state["stop"] or stop_file.exists() or state["step"] >= MAX_STEPS:
                save_now("stop-loop")
                return
            try:
                content = batch["content"].to(device, non_blocking=True)
                style = batch["style"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                target_01 = batch["target_01"].to(device, non_blocking=True)
                delta_img = batch["delta"].to(device, non_blocking=True)

                if random.random() < DROP_CFG:
                    content = torch.ones_like(content)
                    style = torch.ones_like(style)
                drop_delta = random.random() < DROP_DELTA

                noise = torch.randn_like(target)
                t = torch.randint(0, n_t, (target.shape[0],), device=device).long()
                xt = sched.add_noise(target, noise, t)

                with torch.no_grad():
                    style_feat, _, _ = se(style)
                    b, c, h, w = style_feat.shape
                    style_hidden = style_feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
                    content_feat, content_res = ce(content)
                    mix_feat, mix_res = ce(delta_img)
                content_res = list(content_res) + [content_feat]
                mix_res = list(mix_res) + [mix_feat]
                if drop_delta:
                    delta_res = [torch.zeros_like(x) for x in mix_res]
                else:
                    delta_res = [m - c_ for m, c_ in zip(mix_res, content_res)]
                hidden = [style_feat, content_res, style_hidden, delta_res]

                with amp_ctx():
                    out = unet(
                        xt,
                        t,
                        encoder_hidden_states=hidden,
                        content_encoder_downsample_size=args.content_encoder_downsample_size,
                    )
                    noise_pred = out[0]
                    offset_sum = out[1]
                    diff = F.mse_loss(noise_pred.float(), noise.float())
                    offset_loss = offset_sum / 2
                    if not torch.is_tensor(offset_loss):
                        offset_loss = torch.zeros((), device=device)

                    pred_x0 = x0_from_epsilon(sched, noise_pred.float(), xt, t)
                    pred_01 = reNormalize_img(pred_x0)
                    percep = perceptual.calculate_loss(
                        generated_images=normalize_mean_std(pred_01),
                        target_images=normalize_mean_std(target_01),
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
                oom_streak = 0
                step = state["step"]
                if step % 20 == 0:
                    mem = torch.cuda.max_memory_allocated() / 1024**2
                    log(
                        f"step={step} loss={float(loss):.4f} diff={float(diff):.4f} "
                        f"percep={float(percep):.4f} mem={mem:.0f}MiB dt={time.time()-t0:.0f}s"
                    )
                    torch.cuda.reset_peak_memory_stats()
                    write_status({"step": step, "loss": float(loss), "mem_max_miB": round(mem, 1)})
                if step % 100 == 0:
                    save_now("periodic")
                    torch.cuda.empty_cache()
                if step % 500 == 0:
                    save_now("step500")
            except torch.cuda.OutOfMemoryError:
                oom_streak += 1
                log(f"OOM streak={oom_streak}")
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                time.sleep(8)
                if oom_streak >= 5:
                    sys.exit(OOM_EXIT)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_streak += 1
                    torch.cuda.empty_cache()
                    time.sleep(8)
                    if oom_streak >= 5:
                        sys.exit(OOM_EXIT)
                    continue
                log(f"RuntimeError: {e}")
                torch.cuda.empty_cache()
                continue

    save_now("max_steps")
    log(f"done step={state['step']}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        log(f"FATAL {type(e).__name__}: {e}")
        write_status({"fatal": f"{type(e).__name__}: {e}"})
        raise
