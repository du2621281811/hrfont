#!/usr/bin/env python3
"""Representative F0/F3 eval: E1 stratified protocol on test16.

Does not change sampling RNG across shards/GPUs: each (method, font, char)
calls accelerate.set_seed(3407) immediately before DPM.

GPU3 is F2 training — refuse it. Prefer CUDA_VISIBLE_DEVICES pointing at an
idle card (GPU2). Parallelism = one process per method, skip-existing PNGs.

Writes reports/f03_test16_strat/ only.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path("/root/projects/hrfont")
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
SPLIT = ROOT / "manifests/split_v3_228_16_16.json"
TEST_STEMS = ROOT / "manifests/pipeline_v3_test_stems.txt"
OUT = ROOT / "reports/f03_test16_strat"
REF8 = list("永和书风骨韵天地")
SEED = 3407
STRATIFIED = (
    list("0123456789")
    + list("AGMQRWBCO")
    + list("aodpqbegcilnu")
    + list("àéüāě")
    + list("あかさん")
    + list("アカン")
    + list("ㄅㄆㄚ")
)
BUCKET_ORDER = [
    "digit",
    "latin_upper",
    "latin_lower",
    "latin_ext",
    "hiragana",
    "katakana",
    "bopomofo",
]
METHODS = {
    "P1": {
        "label": "官方",
        "kind": "image",
        "variant": ROOT / "code/variants/cn2west_ft_v2/FontDiffuser",
        "ckpt": ROOT / "code/official/FontDiffuser/ckpt",
    },
    "F0_100k": {
        "label": "F0@100k",
        "kind": "image",
        "variant": ROOT / "code/variants/cn2west_f0_rsifree/FontDiffuser",
        "ckpt": ROOT / "runs/F0-RSIFREE-FT-A-S3407/global_step_100000",
    },
    "F3_80k": {
        "label": "F3@80k",
        "kind": "f3",
        "variant": ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser",
        "ckpt": ROOT / "runs/F3-JOINT-DS-A-S3407/global_step_80000",
    },
}
F3_VARIANT = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"
F3_RUN = ROOT / "runs/F3-JOINT-DS-A-S3407"
TIMELINE_STEPS = [5000, 10000, 20000, 30000, 40000, 50000, 60000, 75000, 80000]
# All 16 test fonts; 16 chars covering every script so 9 steps stay readable.
TIMELINE_CHARS = list("08AGQaegàěあさアンㄅㄚ")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cp_of(ch: str) -> str:
    return f"u{ord(ch):04X}"


def fonts() -> list[str]:
    return [l.strip() for l in TEST_STEMS.read_text().splitlines() if l.strip()]


def script_bucket(ch: str) -> str:
    if ch.isdigit():
        return "digit"
    name = unicodedata.name(ch, "")
    if "HIRAGANA" in name:
        return "hiragana"
    if "KATAKANA" in name:
        return "katakana"
    if "BOPOMOFO" in name:
        return "bopomofo"
    if "LATIN" in name and ch.isupper():
        return "latin_upper"
    if "LATIN" in name and ch.islower() and ord(ch) > 127:
        return "latin_ext"
    if "LATIN" in name and ch.islower():
        return "latin_lower"
    return "other"


def content_path(ch: str) -> Path:
    for sp in ("test", "val", "train"):
        p = DATA / sp / "ContentImage" / f"{cp_of(ch)}.png"
        if p.is_file():
            return p
    raise FileNotFoundError(ch)


def gt_path(stem: str, ch: str) -> Path | None:
    p = DATA / "test" / "TargetImage" / stem / f"{stem}+{cp_of(ch)}.png"
    return p if p.is_file() else None


def pick_style_path(stem: str) -> Path:
    d = DATA / "test" / "StyleImage" / stem
    for ch in REF8:
        p = d / f"{stem}+{cp_of(ch)}.png"
        if p.is_file():
            return p
    pngs = sorted(d.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(d)
    return pngs[0]


def pred_path(mid: str, stem: str, ch: str) -> Path:
    return OUT / "preds" / mid / "test" / stem / f"test__{stem}__{cp_of(ch)}__s{SEED}.png"


def refuse_f2_gpu() -> None:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis.strip() == "3":
        raise SystemExit("refusing CUDA_VISIBLE_DEVICES=3 (F2 is training there)")


def parse_shard(s: str) -> tuple[int, int]:
    i, n = s.split("/")
    i, n = int(i), int(n)
    if not (0 <= i < n):
        raise ValueError(s)
    return i, n


def shard_list(items: list[str], spec: str) -> list[str]:
    i, n = parse_shard(spec)
    return [x for k, x in enumerate(items) if k % n == i]


def update_status(patch: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "status.json"
    lock_p = OUT / "status.lock"
    with lock_p.open("a+") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        cur = {}
        if path.is_file():
            try:
                cur = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cur = {}
        methods = cur.setdefault("methods", {})
        if "method" in patch:
            mid = patch.pop("method")
            rec = methods.setdefault(mid, {})
            rec.update(patch)
            rec["updated_at"] = utc_now()
        else:
            cur.update(patch)
        cur["updated_at"] = utc_now()
        path.write_text(json.dumps(cur, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def log(mid: str, msg: str) -> None:
    line = f"[{utc_now()}] [{mid}] {msg}"
    print(line, flush=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    with (OUT / "logs" / f"{mid}.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def to_tensor96(img: Image.Image, device: str):
    import torch
    import torchvision.transforms as T

    if img.size != (96, 96):
        img = img.resize((96, 96))
    t = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    return t(img.convert("RGB"))[None].to(device)


def load_image_pipe(variant: Path, ckpt_dir: Path, device: str):
    sys.path.insert(0, str(variant))
    import torch
    from src import (
        FontDiffuserDPMPipeline,
        FontDiffuserModelDPM,
        build_content_encoder,
        build_ddpm_scheduler,
        build_style_encoder,
        build_unet,
    )

    args = SimpleNamespace(
        resolution=96,
        unet_channels=(64, 128, 256, 512),
        style_image_size=(96, 96),
        content_image_size=(96, 96),
        content_encoder_downsample_size=3,
        channel_attn=True,
        content_start_channel=64,
        style_start_channel=64,
        beta_scheduler="scaled_linear",
        model_type="noise",
    )
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    unet.load_state_dict(torch.load(ckpt_dir / "unet.pth", map_location="cpu", weights_only=True))
    style_encoder.load_state_dict(
        torch.load(ckpt_dir / "style_encoder.pth", map_location="cpu", weights_only=True)
    )
    content_encoder.load_state_dict(
        torch.load(ckpt_dir / "content_encoder.pth", map_location="cpu", weights_only=True)
    )
    model = FontDiffuserModelDPM(
        unet=unet, style_encoder=style_encoder, content_encoder=content_encoder
    ).to(device)
    model.eval()
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=build_ddpm_scheduler(args),
        version="V3",
        model_type="noise",
        guidance_type="classifier-free",
        guidance_scale=7.5,
    )
    return pipe, args


def sample_image(pipe, args, content_img, style_img, device: str) -> Image.Image:
    import torch
    from accelerate.utils import set_seed

    set_seed(SEED)
    with torch.no_grad():
        images = pipe.generate(
            content_images=to_tensor96(content_img, device),
            style_images=to_tensor96(style_img, device),
            batch_size=1,
            order=2,
            num_inference_step=20,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
            t_start=None,
            t_end=None,
            dm_size=args.content_image_size,
            algorithm_type="dpmsolver++",
            skip_type="time_uniform",
            method="multistep",
            correcting_x0_fn=None,
        )
    im = images[0]
    if isinstance(im, Image.Image):
        return im.convert("RGB")
    arr = im.detach().cpu()
    if arr.ndim == 4:
        arr = arr[0]
    arr = ((arr.clamp(-1, 1) + 1) * 0.5 * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def write_sidecar(mid: str, stem: str, ch: str, style_p: Path) -> None:
    g = gt_path(stem, ch)
    meta = {
        "method": mid,
        "split": "test",
        "font": stem,
        "char": ch,
        "cp": cp_of(ch),
        "bucket": script_bucket(ch),
        "seed": SEED,
        "style_path": str(style_p.relative_to(ROOT)),
        "content_path": str(content_path(ch).relative_to(ROOT)),
        "gt_path": str(g.relative_to(ROOT)) if g else None,
    }
    pred_path(mid, stem, ch).with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def generate_image_method(mid: str, device: str, stems: list[str], overwrite: bool) -> None:
    spec = METHODS[mid]
    n_total = len(stems) * len(STRATIFIED)
    update_status(
        {
            "method": mid,
            "phase": "running",
            "done": 0,
            "skipped": 0,
            "total": n_total,
            "label": spec["label"],
        }
    )
    log(mid, f"load {spec['ckpt']}")
    pipe, pargs = load_image_pipe(spec["variant"], spec["ckpt"], device)
    done = skipped = 0
    t0 = time.time()
    for stem in stems:
        style_p = pick_style_path(stem)
        style_img = Image.open(style_p).convert("RGB")
        for ch in STRATIFIED:
            out_p = pred_path(mid, stem, ch)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            if out_p.is_file() and not overwrite:
                skipped += 1
                continue
            pred = sample_image(pipe, pargs, Image.open(content_path(ch)).convert("RGB"), style_img, device)
            pred.save(out_p)
            write_sidecar(mid, stem, ch, style_p)
            done += 1
            if (done + skipped) % 20 == 0:
                rate = done / max(1e-6, time.time() - t0)
                eta = (n_total - done - skipped) / max(rate, 1e-6)
                update_status(
                    {
                        "method": mid,
                        "phase": "running",
                        "done": done,
                        "skipped": skipped,
                        "total": n_total,
                        "rate_per_s": round(rate, 3),
                        "eta_s": int(eta),
                        "last": f"{stem} {ch}",
                    }
                )
                log(mid, f"done={done} skipped={skipped}/{n_total} rate={rate:.2f}/s eta={eta/60:.1f}m last={stem} {ch}")
    elapsed = time.time() - t0
    update_status(
        {
            "method": mid,
            "phase": "done",
            "done": done,
            "skipped": skipped,
            "total": n_total,
            "elapsed_s": round(elapsed, 1),
        }
    )
    log(mid, f"finished done={done} skipped={skipped} elapsed={elapsed:.1f}s")


def make_dpm_adapter(fd, torch):
    class DPMAdapter(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.fd = inner

        def forward(self, x_t, timesteps, cond, content_encoder_downsample_size, version):
            noise, _ = self.fd(
                x_t,
                timesteps,
                content_images=cond[0],
                content_encoder_downsample_size=content_encoder_downsample_size,
                style_features=cond[2],
                structure_features=cond[3],
                content_features=cond[4],
                support_tokens=cond[5],
            )
            return noise

    return DPMAdapter(fd)


def cat_cond(uncond, cond):
    out = []
    for u, c in zip(uncond, cond):
        if u is None and c is None:
            out.append(None)
        elif isinstance(u, list):
            out.append([torch_cat(a, b) for a, b in zip(u, c)])
        else:
            out.append(torch_cat(u, c))
    return out


def torch_cat(a, b):
    import torch

    return torch.cat([a, b], dim=0)


def generate_f3(device: str, stems: list[str], overwrite: bool) -> None:
    import torch
    from accelerate.utils import set_seed

    mid = "F3_80k"
    n_total = len(stems) * len(STRATIFIED)
    update_status({"method": mid, "phase": "loading", "done": 0, "skipped": 0, "total": n_total, "label": "F3@80k"})
    variant = METHODS[mid]["variant"]
    ckpt = METHODS[mid]["ckpt"]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(variant))
    os.chdir(variant)
    import train as T
    from scripts.hrfont_feature_cache import EcCache, EsCache
    from scripts.hrfont_support_adapter import SupportAdapter
    from src.build import (
        build_content_encoder,
        build_ddpm_scheduler,
        build_style_encoder,
        build_unet,
    )
    from src.dpm_solver.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP
    from src.model import FontDiffuserModel

    log(mid, "load Es/Ec caches + LibraryEs")
    es = EsCache(ROOT / "artifacts/f0/es_spatial_f0")
    ec = EcCache(ROOT / "artifacts/f0/ec_multiscale_f0")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_fonts = sorted(split["stems"]["train"])
    library = T._LibraryEs(es, train_fonts, T._style_chars_from_cache(es))
    bank = T._load_support_bank(str(ROOT / "artifacts/f0/support_bank.json"), SimpleNamespace(support=True))
    cfg = SimpleNamespace(
        rsi_source="delta",
        delta_enabled=True,
        delta_tau=0.07,
        delta_eps_alpha=0.01,
        delta_k_max=10,
        delta_k_top=10,
        delta_mode="topk",
        seed=SEED,
        support=True,
        support_k=8,
        style_start_channel=64,
    )
    args = SimpleNamespace(
        resolution=96,
        unet_channels=(64, 128, 256, 512),
        style_image_size=(96, 96),
        content_image_size=(96, 96),
        content_encoder_downsample_size=3,
        channel_attn=True,
        content_start_channel=64,
        style_start_channel=64,
        beta_scheduler="scaled_linear",
    )
    fd = FontDiffuserModel(
        unet=build_unet(args),
        style_encoder=build_style_encoder(args),
        content_encoder=build_content_encoder(args),
    )
    fd.unet.load_state_dict(torch.load(ckpt / "unet.pth", map_location="cpu", weights_only=True))
    fd.style_encoder.load_state_dict(
        torch.load(ckpt / "style_encoder.pth", map_location="cpu", weights_only=True)
    )
    fd.content_encoder.load_state_dict(
        torch.load(ckpt / "content_encoder.pth", map_location="cpu", weights_only=True)
    )
    adapter = SupportAdapter(sum(T.EC_SCALE_CHANNELS), cfg.style_start_channel * 16)
    adapter.load_state_dict(torch.load(ckpt / "support_adapter.pth", map_location="cpu", weights_only=True))
    fd.support_adapter = adapter
    T._ban_encoder_forward(fd)
    device_t = torch.device(device)
    fd.to(device_t).eval()
    model = make_dpm_adapter(fd, torch).to(device_t).eval()
    scheduler = build_ddpm_scheduler(args)
    noise_schedule = NoiseScheduleVP(schedule="discrete", betas=scheduler.betas)
    keep = torch.zeros(1, dtype=torch.bool, device=device_t)

    def pack_one(font: str, ch: str):
        samples = {
            "split": ["test"],
            "font_stem": [font],
            "char_cp": [cp_of(ch)],
            "ref_chars": [[f"u{ord(c):04X}" for c in "永和书风骨韵天地"]],
        }
        style, queries = T._style_conditions(es, samples, device_t)
        structure = T._structure_features(es, ec, library, samples, queries, cfg, keep, device_t)
        content = T._content_features(ec, samples, keep, device_t)
        vecs = []
        for scp in list(bank.get(cp_of(ch), []))[: cfg.support_k]:
            try:
                feats = [x.to(device_t) for x in ec.features("style", font, scp)]
            except KeyError:
                continue
            vecs.append(T._pool_ec(feats))
        pooled = torch.stack(vecs, dim=0).unsqueeze(0) if vecs else None
        return {"style": style, "structure": structure, "content": content, "support_pooled": pooled}

    def sample_one(packed):
        set_seed(SEED)
        img = torch.zeros(1, 3, 96, 96, device=device_t)
        style = packed["style"]
        structure = packed["structure"]
        content = packed["content"]
        pooled = packed["support_pooled"]
        support = fd.support_adapter(pooled) if pooled is not None else None
        cond = [img, img, style, structure, content, support]
        uncond = [
            torch.ones_like(img),
            torch.ones_like(img),
            torch.zeros_like(style),
            [torch.zeros_like(x) for x in structure],
            [torch.zeros_like(x) for x in content],
            torch.zeros_like(support) if support is not None else None,
        ]

        def get_t_input(t_continuous):
            return (t_continuous - 1.0 / noise_schedule.total_N) * 1000.0

        def model_fn(x, t_continuous):
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t_continuous, t_continuous], dim=0)
            t_input = get_t_input(t_in)
            c_in = cat_cond(uncond, cond)
            noise_uncond, noise = model(
                x_in,
                t_input,
                c_in,
                content_encoder_downsample_size=3,
                version="V3",
            ).chunk(2)
            return noise_uncond + 7.5 * (noise - noise_uncond)

        solver = DPM_Solver(model_fn=model_fn, noise_schedule=noise_schedule, algorithm_type="dpmsolver++")
        x = torch.randn(1, 3, 96, 96, device=device_t)
        with torch.no_grad():
            x = solver.sample(x=x, steps=20, order=2, skip_type="time_uniform", method="multistep")
        x = (x / 2 + 0.5).clamp(0, 1)[0].detach().cpu().permute(1, 2, 0).numpy()
        return Image.fromarray((x * 255).round().astype("uint8"))

    done = skipped = 0
    t0 = time.time()
    update_status({"method": mid, "phase": "running", "done": 0, "skipped": 0, "total": n_total})
    for stem in stems:
        style_p = pick_style_path(stem)
        for ch in STRATIFIED:
            out_p = pred_path(mid, stem, ch)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            if out_p.is_file() and not overwrite:
                skipped += 1
                continue
            packed = pack_one(stem, ch)
            pred = sample_one(packed)
            pred.save(out_p)
            write_sidecar(mid, stem, ch, style_p)
            done += 1
            if (done + skipped) % 10 == 0:
                rate = done / max(1e-6, time.time() - t0)
                eta = (n_total - done - skipped) / max(rate, 1e-6)
                update_status(
                    {
                        "method": mid,
                        "phase": "running",
                        "done": done,
                        "skipped": skipped,
                        "total": n_total,
                        "rate_per_s": round(rate, 3),
                        "eta_s": int(eta),
                        "last": f"{stem} {ch}",
                    }
                )
                log(mid, f"done={done} skipped={skipped}/{n_total} rate={rate:.2f}/s eta={eta/60:.1f}m last={stem} {ch}")
    elapsed = time.time() - t0
    update_status(
        {
            "method": mid,
            "phase": "done",
            "done": done,
            "skipped": skipped,
            "total": n_total,
            "elapsed_s": round(elapsed, 1),
        }
    )
    log(mid, f"finished done={done} skipped={skipped} elapsed={elapsed:.1f}s")


def f3_mid(step: int) -> str:
    return "F3_80k" if step == 80000 else f"F3_{step}"


def generate_f3_timeline(device: str, overwrite: bool) -> None:
    """Sample F3 named ckpts. Conditions come from frozen F0 caches (shared across steps)."""
    import torch
    from accelerate.utils import set_seed

    stems = fonts()
    chars = TIMELINE_CHARS
    n_total = len(stems) * len(chars) * len(TIMELINE_STEPS)
    update_status(
        {
            "method": "F3_timeline",
            "phase": "loading",
            "done": 0,
            "skipped": 0,
            "total": n_total,
            "label": "F3 过程",
        }
    )
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(F3_VARIANT))
    os.chdir(F3_VARIANT)
    import train as T
    from scripts.hrfont_feature_cache import EcCache, EsCache
    from scripts.hrfont_support_adapter import SupportAdapter
    from src.build import (
        build_content_encoder,
        build_ddpm_scheduler,
        build_style_encoder,
        build_unet,
    )
    from src.dpm_solver.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP
    from src.model import FontDiffuserModel

    log("F3_timeline", "load Es/Ec caches")
    es = EsCache(ROOT / "artifacts/f0/es_spatial_f0")
    ec = EcCache(ROOT / "artifacts/f0/ec_multiscale_f0")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_fonts = sorted(split["stems"]["train"])
    library = T._LibraryEs(es, train_fonts, T._style_chars_from_cache(es))
    bank = T._load_support_bank(str(ROOT / "artifacts/f0/support_bank.json"), SimpleNamespace(support=True))
    cfg = SimpleNamespace(
        rsi_source="delta", delta_enabled=True, delta_tau=0.07, delta_eps_alpha=0.01,
        delta_k_max=10, delta_k_top=10, delta_mode="topk", seed=SEED, support=True,
        support_k=8, style_start_channel=64,
    )
    args = SimpleNamespace(
        resolution=96, unet_channels=(64, 128, 256, 512),
        style_image_size=(96, 96), content_image_size=(96, 96),
        content_encoder_downsample_size=3, channel_attn=True,
        content_start_channel=64, style_start_channel=64, beta_scheduler="scaled_linear",
    )
    device_t = torch.device(device)
    fd = FontDiffuserModel(
        unet=build_unet(args),
        style_encoder=build_style_encoder(args),
        content_encoder=build_content_encoder(args),
    )
    adapter = SupportAdapter(sum(T.EC_SCALE_CHANNELS), cfg.style_start_channel * 16)
    fd.support_adapter = adapter
    T._ban_encoder_forward(fd)
    fd.to(device_t).eval()
    model = make_dpm_adapter(fd, torch).to(device_t).eval()
    scheduler = build_ddpm_scheduler(args)
    noise_schedule = NoiseScheduleVP(schedule="discrete", betas=scheduler.betas)
    keep = torch.zeros(1, dtype=torch.bool, device=device_t)

    def pack_one(font: str, ch: str):
        samples = {
            "split": ["test"], "font_stem": [font], "char_cp": [cp_of(ch)],
            "ref_chars": [[f"u{ord(c):04X}" for c in "永和书风骨韵天地"]],
        }
        style, queries = T._style_conditions(es, samples, device_t)
        structure = T._structure_features(es, ec, library, samples, queries, cfg, keep, device_t)
        content = T._content_features(ec, samples, keep, device_t)
        vecs = []
        for scp in list(bank.get(cp_of(ch), []))[: cfg.support_k]:
            try:
                feats = [x.to(device_t) for x in ec.features("style", font, scp)]
            except KeyError:
                continue
            vecs.append(T._pool_ec(feats))
        pooled = torch.stack(vecs, dim=0).unsqueeze(0) if vecs else None
        return {"style": style, "structure": structure, "content": content, "support_pooled": pooled}

    log("F3_timeline", f"precompute conditions fonts={len(stems)} chars={len(chars)}")
    packed_map = {}
    for stem in stems:
        for ch in chars:
            packed_map[(stem, ch)] = pack_one(stem, ch)

    done = skipped = 0
    t0 = time.time()
    for step in TIMELINE_STEPS:
        mid = f3_mid(step)
        ckpt = F3_RUN / f"global_step_{step}"
        log("F3_timeline", f"load {ckpt}")
        fd.unet.load_state_dict(torch.load(ckpt / "unet.pth", map_location="cpu", weights_only=True))
        fd.style_encoder.load_state_dict(
            torch.load(ckpt / "style_encoder.pth", map_location="cpu", weights_only=True)
        )
        fd.content_encoder.load_state_dict(
            torch.load(ckpt / "content_encoder.pth", map_location="cpu", weights_only=True)
        )
        adapter.load_state_dict(torch.load(ckpt / "support_adapter.pth", map_location="cpu", weights_only=True))
        adapter.to(device_t)
        fd.to(device_t).eval()

        def sample_one(p):
            set_seed(SEED)
            img = torch.zeros(1, 3, 96, 96, device=device_t)
            support = fd.support_adapter(p["support_pooled"]) if p["support_pooled"] is not None else None
            cond = [img, img, p["style"], p["structure"], p["content"], support]
            uncond = [
                torch.ones_like(img), torch.ones_like(img), torch.zeros_like(p["style"]),
                [torch.zeros_like(x) for x in p["structure"]],
                [torch.zeros_like(x) for x in p["content"]],
                torch.zeros_like(support) if support is not None else None,
            ]

            def get_t_input(t_continuous):
                return (t_continuous - 1.0 / noise_schedule.total_N) * 1000.0

            def model_fn(x, t_continuous):
                x_in = torch.cat([x, x], dim=0)
                t_in = torch.cat([t_continuous, t_continuous], dim=0)
                noise_uncond, noise = model(
                    x_in, get_t_input(t_in), cat_cond(uncond, cond),
                    content_encoder_downsample_size=3, version="V3",
                ).chunk(2)
                return noise_uncond + 7.5 * (noise - noise_uncond)

            solver = DPM_Solver(model_fn=model_fn, noise_schedule=noise_schedule, algorithm_type="dpmsolver++")
            x = torch.randn(1, 3, 96, 96, device=device_t)
            with torch.no_grad():
                x = solver.sample(x=x, steps=20, order=2, skip_type="time_uniform", method="multistep")
            x = (x / 2 + 0.5).clamp(0, 1)[0].detach().cpu().permute(1, 2, 0).numpy()
            return Image.fromarray((x * 255).round().astype("uint8"))

        for stem in stems:
            style_p = pick_style_path(stem)
            for ch in chars:
                out_p = pred_path(mid, stem, ch)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                if out_p.is_file() and not overwrite:
                    skipped += 1
                    continue
                pred = sample_one(packed_map[(stem, ch)])
                pred.save(out_p)
                write_sidecar(mid, stem, ch, style_p)
                done += 1
                if (done + skipped) % 20 == 0:
                    rate = done / max(1e-6, time.time() - t0)
                    eta = (n_total - done - skipped) / max(rate, 1e-6)
                    update_status(
                        {
                            "method": "F3_timeline",
                            "phase": "running",
                            "done": done,
                            "skipped": skipped,
                            "total": n_total,
                            "rate_per_s": round(rate, 3),
                            "eta_s": int(eta),
                            "last": f"{step} {stem} {ch}",
                        }
                    )
                    log("F3_timeline", f"step={step} done={done} skipped={skipped}/{n_total} rate={rate:.2f}/s eta={eta/60:.1f}m")
    elapsed = time.time() - t0
    update_status(
        {
            "method": "F3_timeline",
            "phase": "done",
            "done": done,
            "skipped": skipped,
            "total": n_total,
            "elapsed_s": round(elapsed, 1),
        }
    )
    log("F3_timeline", f"finished done={done} skipped={skipped} elapsed={elapsed:.1f}s")
    write_timeline_html()


def write_timeline_html() -> None:
    steps = TIMELINE_STEPS
    mids = [f3_mid(s) for s in steps]
    items = []
    for stem in fonts():
        for ch in TIMELINE_CHARS:
            preds = {f3_mid(s): f"preds/{f3_mid(s)}/test/{stem}/test__{stem}__{cp_of(ch)}__s{SEED}.png" for s in steps}
            items.append(
                {
                    "font": stem,
                    "char": ch,
                    "cp": cp_of(ch),
                    "bucket": script_bucket(ch),
                    "content": f"refs/content/{cp_of(ch)}.png",
                    "style": f"refs/style/{stem}.png",
                    "gt": f"refs/gt/{stem}+{cp_of(ch)}.png",
                    "f0": f"preds/F0_100k/test/{stem}/test__{stem}__{cp_of(ch)}__s{SEED}.png",
                    "preds": preds,
                }
            )
    payload = {
        "fonts": fonts(),
        "chars": TIMELINE_CHARS,
        "steps": steps,
        "mids": mids,
        "items": items,
    }
    html = f"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<meta http-equiv="refresh" content="25"/>
<title>F3 训练过程 · test16</title>
<style>
:root{{--bg:#eef1f5;--card:#fff;--line:#d5dbe3;--muted:#5c6570;--ink:#1a1a1a;--accent:#1f4a6f}}
*{{box-sizing:border-box}} body{{margin:0;font:14px/1.45 system-ui,"Noto Sans SC",sans-serif;background:var(--bg);color:var(--ink)}}
header{{background:var(--card);border-bottom:1px solid var(--line);padding:14px 18px}}
h1{{margin:0;font-size:1.2rem}} .meta{{color:var(--muted);font-size:12px;margin-top:4px}}
main{{max-width:1400px;margin:16px auto;padding:0 14px 48px}}
.note{{background:#fff8e8;border:1px solid #e6d7a8;padding:8px 10px;margin:10px 0;font-size:12px}}
select{{padding:6px 8px;border:1px solid var(--line);background:#fff;margin-right:6px}}
.grid{{overflow-x:auto}}
table.g th,table.g td{{border:1px solid var(--line);padding:3px;text-align:center;vertical-align:bottom;font-size:11px;color:var(--muted)}}
table.g img{{width:72px;height:72px;image-rendering:pixelated;display:block;background:#fff}}
table.g td.ch{{font:700 15px/1.2 ui-serif,serif;color:var(--ink)}}
a{{color:var(--accent)}}
pre{{white-space:pre-wrap}}
</style></head><body>
<header>
  <h1>F3 从 5k 到 80k 的变化</h1>
  <div class="meta">test16 套字体 · 16 个跨语种字符 · 同一 DPM++20 / CFG7.5 / seed 3407 · <a href="./">返回终点评测</a></div>
</header>
<main>
<div class="note">左两列是 GT 与父模型 F0@100k。后面每一列是 F3 的一个 checkpoint。条件始终是 cache Δ + Support×8，只有 UNet/RSI 头在变。5k 刚过 warmup。75k 是 trainer val 最好，80k 是终点。</div>
<section><pre id="prog" class="meta">读取进度…</pre></section>
<p>
  <label>字体 <select id="font"></select></label>
  <label>语种 <select id="bucket"><option value="">全部</option></select></label>
</p>
<div class="grid" id="sheet"></div>
</main>
<script>
const DATA = {json.dumps(payload, ensure_ascii=False)};
const BNAME = {{digit:'数字', latin_upper:'拉丁大写', latin_lower:'拉丁小写', latin_ext:'拉丁扩展', hiragana:'平假名', katakana:'片假名', bopomofo:'注音'}};
const fontSel = document.getElementById('font');
const bucketSel = document.getElementById('bucket');
DATA.fonts.forEach(f => {{ const o=document.createElement('option'); o.value=f; o.textContent=f; fontSel.appendChild(o); }});
[...new Set(DATA.items.map(i=>i.bucket))].forEach(b => {{ const o=document.createElement('option'); o.value=b; o.textContent=BNAME[b]||b; bucketSel.appendChild(o); }});
function render(){{
  const font=fontSel.value, bucket=bucketSel.value;
  const items=DATA.items.filter(it=>it.font===font && (!bucket || it.bucket===bucket));
  const heads=['字','Content','GT','F0@100k'].concat(DATA.steps.map(s=>'F3@'+(s/1000)+'k'));
  let h='<table class="g"><thead><tr>'+heads.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>';
  for (const it of items){{
    h += '<tr><td class="ch">'+it.char+'</td>';
    h += '<td><img src="'+it.content+'"/></td><td><img src="'+it.gt+'"/></td><td><img src="'+it.f0+'"/></td>';
    for (const mid of DATA.mids){{
      h += '<td><img src="'+it.preds[mid]+'" onerror="this.style.opacity=.2"/></td>';
    }}
    h += '</tr>';
  }}
  document.getElementById('sheet').innerHTML = h+'</tbody></table>';
}}
fontSel.onchange=render; bucketSel.onchange=render; render();
async function tick(){{
  try {{
    const s = await (await fetch('status.json?t='+Date.now(),{{cache:'no-store'}})).json();
    const v = (s.methods||{{}}).F3_timeline || {{}};
    const tot=v.total||0, d=(v.done||0)+(v.skipped||0);
    const pct = tot? Math.round(100*d/tot):0;
    let t = (v.phase||'')+' '+d+'/'+tot+' ('+pct+'%)';
    if (v.eta_s) t += '  ETA '+Math.round(v.eta_s/60)+' min';
    if (v.last) t += '  '+v.last;
    document.getElementById('prog').textContent = t;
  }} catch(e) {{ document.getElementById('prog').textContent = String(e); }}
}}
tick(); setInterval(tick, 8000);
</script>
</body></html>
"""
    (OUT / "timeline.html").write_text(html, encoding="utf-8")
    print("wrote", OUT / "timeline.html", "items", len(items))


def cmd_timeline(args: argparse.Namespace) -> None:
    refuse_f2_gpu()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    write_timeline_html()
    if args.html_only:
        return
    generate_f3_timeline(args.device, args.overwrite)


def cmd_generate(args: argparse.Namespace) -> None:
    refuse_f2_gpu()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    stems = shard_list(fonts(), args.shard)
    protocol = {
        "split": "test",
        "n_fonts": 16,
        "fonts": fonts(),
        "chars": STRATIFIED,
        "chars_n": len(STRATIFIED),
        "seed": SEED,
        "sampler": "dpmsolver++ 20 CFG7.5 order2 multistep",
        "style": "ref8 first available, prefer 永",
        "note": "Same protocol as E1 formal stratified. Val16 is not used. Per-item set_seed(3407).",
        "methods": {k: {"label": v["label"], "kind": v["kind"], "ckpt": str(v["ckpt"])} for k, v in METHODS.items()},
    }
    (OUT / "PROTOCOL.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_status({"protocol": "E1-stratified test16×47 seed3407", "phase": "generate"})
    mid = args.method
    if METHODS[mid]["kind"] == "f3":
        generate_f3(args.device, stems, args.overwrite)
    else:
        generate_image_method(mid, args.device, stems, args.overwrite)


def to_gray01(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    c1, c2 = 0.01**2, 0.03**2
    mu_a, mu_b = a.mean(), b.mean()
    sig_a, sig_b = a.var(), b.var()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (sig_a + sig_b + c2)))


def ink_coverage(a: np.ndarray, thr: float = 0.92) -> float:
    return float((a < thr).mean())


def agg(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}

    def mean(key: str):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(sum(vals) / len(vals)) if vals else None

    return {
        "n": len(rows),
        "L1_mean": mean("L1"),
        "SSIM_mean": mean("SSIM"),
        "coverage_mean": mean("coverage"),
        "LPIPS_mean": mean("LPIPS"),
        "blank_rate": float(sum(1 for r in rows if (r.get("coverage") or 0) < 0.005) / len(rows)),
    }


def cmd_metrics(args: argparse.Namespace) -> None:
    refuse_f2_gpu()
    lpips_fn = None
    if args.lpips:
        import torch
        import lpips

        lpips_fn = lpips.LPIPS(net="alex").to(args.device).eval()
    rows = []
    for mid in METHODS:
        for png in (OUT / "preds" / mid).rglob("*.png"):
            meta_p = png.with_suffix(".json")
            if not meta_p.is_file():
                continue
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            gtp = gt_path(meta["font"], meta["char"])
            if gtp is None:
                continue
            pred = Image.open(png).convert("RGB")
            gt = Image.open(gtp).convert("RGB")
            pa, ga = to_gray01(pred), to_gray01(gt)
            rec = {
                **meta,
                "L1": float(np.abs(pa - ga).mean()),
                "SSIM": ssim(pa, ga),
                "coverage": ink_coverage(pa),
                "LPIPS": None,
                "rel": str(png.relative_to(OUT)),
            }
            if lpips_fn is not None:
                import torch

                def to_n11(im: Image.Image):
                    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
                    t = torch.from_numpy(arr).permute(2, 0, 1)[None] * 2 - 1
                    return t.to(args.device)

                with torch.no_grad():
                    rec["LPIPS"] = float(lpips_fn(to_n11(pred), to_n11(gt)).item())
            rows.append(rec)
    report = {
        "computed_at": utc_now(),
        "n_total": len(rows),
        "caveat": "L1/SSIM/LPIPS vs GT are diagnostics, not the paper style claim. E12 still gated.",
        "methods": {},
    }
    by_m = defaultdict(list)
    for r in rows:
        by_m[r["method"]].append(r)
    for mid, rs in by_m.items():
        by_b = defaultdict(list)
        for r in rs:
            by_b[r["bucket"]].append(r)
        report["methods"][mid] = {
            "label": METHODS[mid]["label"],
            "overall": agg(rs),
            "by_bucket": {b: agg(by_b[b]) for b in BUCKET_ORDER if b in by_b},
        }
    (OUT / "metrics_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "metrics_items.json").write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    update_status({"phase": "metrics_done", "n_metric_rows": len(rows)})
    print(json.dumps({m: report["methods"][m]["overall"] for m in report["methods"]}, indent=2))


def copy_refs() -> None:
    ref = OUT / "refs"
    (ref / "content").mkdir(parents=True, exist_ok=True)
    (ref / "gt").mkdir(parents=True, exist_ok=True)
    (ref / "style").mkdir(parents=True, exist_ok=True)
    for ch in STRATIFIED:
        Image.open(content_path(ch)).convert("RGB").resize((96, 96)).save(ref / "content" / f"{cp_of(ch)}.png")
    for stem in fonts():
        Image.open(pick_style_path(stem)).convert("RGB").resize((96, 96)).save(ref / "style" / f"{stem}.png")
        for ch in STRATIFIED:
            g = gt_path(stem, ch)
            dst = ref / "gt" / f"{stem}+{cp_of(ch)}.png"
            if g:
                Image.open(g).convert("RGB").resize((96, 96)).save(dst)


def cmd_gallery(_args: argparse.Namespace) -> None:
    copy_refs()
    metrics = {}
    mp = OUT / "metrics_summary.json"
    if mp.is_file():
        metrics = json.loads(mp.read_text(encoding="utf-8"))
    items = []
    for stem in fonts():
        for ch in STRATIFIED:
            items.append(
                {
                    "font": stem,
                    "char": ch,
                    "cp": cp_of(ch),
                    "bucket": script_bucket(ch),
                    "content": f"refs/content/{cp_of(ch)}.png",
                    "style": f"refs/style/{stem}.png",
                    "gt": f"refs/gt/{stem}+{cp_of(ch)}.png",
                    "preds": {
                        mid: f"preds/{mid}/test/{stem}/test__{stem}__{cp_of(ch)}__s{SEED}.png" for mid in METHODS
                    },
                }
            )
    payload = {
        "generated_at": utc_now(),
        "fonts": fonts(),
        "chars": STRATIFIED,
        "buckets": BUCKET_ORDER,
        "methods": [{"id": k, "label": v["label"]} for k, v in METHODS.items()],
        "metrics": metrics,
        "items": items,
        "n": len(items),
    }
    (OUT / "browse_index.json").write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    write_html(payload)
    update_status({"phase": "gallery_done", "n_items": len(items)})
    print("wrote", OUT / "index.html", "items", len(items))


def _fmt(x, n=4) -> str:
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else "—"


def write_html(payload: dict) -> None:
    metrics = payload.get("metrics") or {}
    rows = []
    methods = payload["methods"]
    mids = [m["id"] for m in methods]
    if metrics.get("methods"):
        rows.append("<table><thead><tr><th>方法</th><th>n</th><th>L1↓</th><th>SSIM↑</th><th>LPIPS↓</th></tr></thead><tbody>")
        for mid in mids:
            o = metrics["methods"].get(mid, {}).get("overall", {})
            rows.append(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    METHODS[mid]["label"], o.get("n"),
                    _fmt(o.get("L1_mean")), _fmt(o.get("SSIM_mean")), _fmt(o.get("LPIPS_mean")),
                )
            )
        rows.append("</tbody></table>")
        rows.append("<h3>按语种</h3><table><thead><tr><th>语种</th>")
        for mid in mids:
            rows.append(f"<th>{METHODS[mid]['label']} L1</th><th>SSIM</th>")
        rows.append("</tr></thead><tbody>")
        for b in BUCKET_ORDER:
            cells = [b]
            for mid in mids:
                bb = metrics["methods"].get(mid, {}).get("by_bucket", {}).get(b, {})
                cells.append(_fmt(bb.get("L1_mean"), 3))
                cells.append(_fmt(bb.get("SSIM_mean"), 3))
            rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        rows.append("</tbody></table>")
    metric_html = "".join(rows)
    html = f"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<meta http-equiv="refresh" content="20"/>
<title>F0 / F3 · test16 × 47 分层评测</title>
<style>
:root{{--bg:#eef1f5;--card:#fff;--line:#d5dbe3;--muted:#5c6570;--ink:#1a1a1a;--accent:#1f4a6f}}
*{{box-sizing:border-box}} body{{margin:0;font:14px/1.45 system-ui,"Noto Sans SC",sans-serif;background:var(--bg);color:var(--ink)}}
header{{background:var(--card);border-bottom:1px solid var(--line);padding:14px 18px}}
h1{{margin:0;font-size:1.2rem}} h2{{font-size:1.05rem;margin:0 0 8px}} h3{{font-size:.95rem}}
.meta,.cap{{color:var(--muted);font-size:12px}}
main{{max-width:1280px;margin:16px auto;padding:0 14px 48px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:14px;margin:14px 0}}
.note{{background:#fff8e8;border:1px solid #e6d7a8;padding:8px 10px;margin:10px 0;font-size:12px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}}
th,td{{border-bottom:1px solid var(--line);padding:5px 7px;text-align:left}}
.bar{{height:8px;background:#e4e9ef;margin-top:4px}} .bar>i{{display:block;height:100%;background:var(--accent)}}
.grid{{overflow-x:auto}}
table.g th,table.g td{{border:1px solid var(--line);padding:3px;text-align:center;vertical-align:bottom;font-size:11px;color:var(--muted)}}
table.g img{{width:72px;height:72px;image-rendering:pixelated;display:block;background:#fff}}
table.g td.ch{{font:700 15px/1.2 ui-serif,serif;color:var(--ink)}}
select{{padding:6px 8px;border:1px solid var(--line);background:#fff;margin-right:6px}}
.pill{{display:inline-block;border:1px solid var(--line);padding:1px 8px;font-size:12px;margin-right:6px}}
</style></head><body>
<header>
  <h1>官方 · F0@100k · F3@80k　测试集分层评测</h1>
  <div class="meta">test16 套字体 × 47 个分层字符（数字 / 拉丁大小写 / 拉丁扩展 / 平假名 / 片假名 / 注音）· DPM++20 CFG7.5 seed3407 · 与 E1 正式评测同一口径</div>
</header>
<main>
<div class="note">这是补全后的代表性格。旧看板 4 字体 × AaG0e 只是训练中途探针，请以本页为准。像素指标相对 GT，不是风格终局；E12 仍未过门。验证集不在这里（val 只用于选 checkpoint）。<br/><b><a href="delta_retrieve/">Δ 检索诊断（汉字依据 vs 取回的西文）→</a></b>
<br/><b><a href="timeline.html">看 F3 从 5k 到 80k 的变化过程 →</a></b></div>
<section class="card" id="progress"><h2>出图进度</h2><pre id="prog" class="meta">读取 status.json …</pre></section>
<section class="card"><h2>诊断指标（相对 GT）</h2>{metric_html or "<p class='meta'>指标将在三路出图完成后计算。</p>"}
<p class="cap">L1 越低、SSIM 越高、LPIPS 越低通常越贴近像素真值。跨文字风格主张仍要等独立评测器。</p>
</section>
<section class="card">
  <h2>对照图</h2>
  <p>
    <label>字体 <select id="font"></select></label>
    <label>语种 <select id="bucket"><option value="">全部</option></select></label>
  </p>
  <div class="grid" id="sheet"></div>
</section>
<p class="cap">F3 走训练条件：cache Δ + 同字体 Support×8，不是再跑官方图像 RSI。官方与 F0 走 Content + Style「永」。每张图独立 set_seed(3407)，分卡/分字体不改变结果。</p>
</main>
<script>
const DATA = {json.dumps({"fonts": payload["fonts"], "chars": payload["chars"], "buckets": payload["buckets"], "methods": payload["methods"], "items": payload["items"]}, ensure_ascii=False)};
const fontSel = document.getElementById('font');
const bucketSel = document.getElementById('bucket');
DATA.fonts.forEach(f => {{ const o=document.createElement('option'); o.value=f; o.textContent=f; fontSel.appendChild(o); }});
const BNAME = {{digit:'数字', latin_upper:'拉丁大写', latin_lower:'拉丁小写', latin_ext:'拉丁扩展', hiragana:'平假名', katakana:'片假名', bopomofo:'注音'}};
DATA.buckets.forEach(b => {{ const o=document.createElement('option'); o.value=b; o.textContent=BNAME[b]||b; bucketSel.appendChild(o); }});
function render(){{
  const font = fontSel.value;
  const bucket = bucketSel.value;
  const items = DATA.items.filter(it => it.font===font && (!bucket || it.bucket===bucket));
  const heads = ['字','Content','Style','GT'].concat(DATA.methods.map(m=>m.label));
  let h = '<table class="g"><thead><tr>'+heads.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>';
  for (const it of items){{
    h += '<tr><td class="ch">'+it.char+'</td>';
    h += '<td><img src="'+it.content+'"/></td>';
    h += '<td><img src="'+it.style+'"/></td>';
    h += '<td><img src="'+it.gt+'"/></td>';
    for (const m of DATA.methods){{
      h += '<td><img src="'+it.preds[m.id]+'" onerror="this.style.opacity=.25"/></td>';
    }}
    h += '</tr>';
  }}
  h += '</tbody></table>';
  document.getElementById('sheet').innerHTML = h;
}}
fontSel.onchange = render; bucketSel.onchange = render; render();
async function tick(){{
  try {{
    const s = await (await fetch('status.json?t='+Date.now(), {{cache:'no-store'}})).json();
    const ms = s.methods||{{}};
    let t = '更新 '+ (s.updated_at||'') + ' · 阶段 ' + (s.phase||'') + '\\n';
    for (const [k,v] of Object.entries(ms)){{
      const tot = v.total||752;
      const d = (v.done||0)+(v.skipped||0);
      const pct = tot? Math.round(100*d/tot):0;
      t += k+' '+ (v.phase||'') + ' '+d+'/'+tot+' ('+pct+'%)';
      if (v.eta_s) t += '  ETA '+Math.round(v.eta_s/60)+' min';
      if (v.rate_per_s) t += '  '+v.rate_per_s+'/s';
      t += '\\n';
    }}
    document.getElementById('prog').textContent = t;
  }} catch(e) {{ document.getElementById('prog').textContent = String(e); }}
}}
tick(); setInterval(tick, 8000);
</script>
</body></html>
"""
    (OUT / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--method", required=True, choices=list(METHODS))
    g.add_argument("--device", default="cuda:0")
    g.add_argument("--shard", default="0/1")
    g.add_argument("--overwrite", action="store_true")
    g.set_defaults(func=cmd_generate)
    m = sub.add_parser("metrics")
    m.add_argument("--device", default="cuda:0")
    m.add_argument("--lpips", action="store_true")
    m.set_defaults(func=cmd_metrics)
    gal = sub.add_parser("gallery")
    gal.set_defaults(func=cmd_gallery)
    p = sub.add_parser("progress-page")
    p.set_defaults(func=cmd_gallery)
    t = sub.add_parser("timeline")
    t.add_argument("--device", default="cuda:0")
    t.add_argument("--overwrite", action="store_true")
    t.add_argument("--html-only", action="store_true")
    t.set_defaults(func=cmd_timeline)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
