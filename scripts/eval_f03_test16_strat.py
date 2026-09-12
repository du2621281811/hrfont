#!/usr/bin/env python3
"""Representative F0/F3 eval: E1 stratified protocol on test16.

Does not change sampling RNG across shards/GPUs: each (method, font, char)
calls accelerate.set_seed(3407) immediately before DPM.

GPU1 holds F2-P training — refuse it. Prefer CUDA_VISIBLE_DEVICES pointing at an
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
REF8_CPS = [f"u{ord(c):04X}" for c in REF8]
STYLE1_CPS = [f"u{ord('永'):04X}"]
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
    "E1_100k": {
        "label": "E1@100k",
        "kind": "image",
        "variant": ROOT / "code/variants/cn2west_ft_v2/FontDiffuser",
        "ckpt": ROOT / "runs/E1-FTV2-A-S3407/global_step_100000",
    },
    "F0_100k": {
        "label": "F0@100k",
        "kind": "image",
        "variant": ROOT / "code/variants/cn2west_f0_rsifree/FontDiffuser",
        "ckpt": ROOT / "runs/F0-RSIFREE-FT-A-S3407/global_step_100000",
    },
    "F1_30000": {
        "label": "F1@30k",
        "kind": "f1",
        "variant": ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser",
        "ckpt": ROOT / "runs/F1-OFFRSI-A-S3407/global_step_30000",
    },
    "F1_80000": {
        "label": "F1@80k",
        "kind": "f1",
        "variant": ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser",
        "ckpt": ROOT / "runs/F1-OFFRSI-A-S3407/global_step_80000",
    },
    "F3_80k": {
        "label": "F3@80k",
        "kind": "f3",
        "variant": ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser",
        "ckpt": ROOT / "runs/F3-JOINT-DS-A-S3407/global_step_80000",
        "style_oneshot": False,
    },
    "F2_75000": {
        "label": "F2@75k",
        "kind": "f2",
        "variant": ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser",
        "ckpt": ROOT / "runs/F2-DELTARSI-A-S3407/global_step_75000",
        "style_oneshot": False,
    },
    "F2_80000": {
        "label": "F2@80k",
        "kind": "f2",
        "variant": ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser",
        "ckpt": ROOT / "runs/F2-DELTARSI-A-S3407/global_step_80000",
        "style_oneshot": False,
    },
    "F2_75000_s1": {
        "label": "F2@75k style1",
        "kind": "f2",
        "variant": ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser",
        "ckpt": ROOT / "runs/F2-DELTARSI-A-S3407/global_step_75000",
        "style_oneshot": True,
    },
    "F2_80000_s1": {
        "label": "F2@80k style1",
        "kind": "f2",
        "variant": ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser",
        "ckpt": ROOT / "runs/F2-DELTARSI-A-S3407/global_step_80000",
        "style_oneshot": True,
    },
    "F2_40000": {
        "label": "F2@40k",
        "kind": "f2",
        "variant": ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser",
        "ckpt": ROOT / "runs/F2-DELTARSI-A-S3407/global_step_40000",
        "style_oneshot": False,
    },
    "F2P_40000": {
        "label": "F2-P@40k",
        "kind": "f2",
        "variant": ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser",
        "ckpt": ROOT / "runs/f2_pattn_s3407/global_step_40000",
        "style_oneshot": False,
        "style_pattn": True,
    },
    "F3bP_40000": {
        "label": "F3b-P@40k",
        "kind": "f3",
        "variant": ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser",
        "ckpt": ROOT / "runs/f3b_pattn_s3407/global_step_40000",
        "style_oneshot": False,
        "style_pattn": True,
        "support_bank": ROOT / "artifacts/f0/support_bank_f3b_topology.json",
    },
    "F3_80k_s1": {
        "label": "F3@80k style1",
        "kind": "f3",
        "variant": ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser",
        "ckpt": ROOT / "runs/F3-JOINT-DS-A-S3407/global_step_80000",
        "style_oneshot": True,
    },
    "F2_80000_k1": {
        "label": "F2@80k 1-shot",
        "kind": "f2",
        "variant": ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser",
        "ckpt": ROOT / "runs/F2-DELTARSI-A-S3407/global_step_80000",
        "style_oneshot": True,
        "delta_oneshot": True,
    },
    "F2P_40000_k1": {
        "label": "F2-P@40k 1-shot",
        "kind": "f2",
        "variant": ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser",
        "ckpt": ROOT / "runs/f2_pattn_s3407/global_step_40000",
        "style_oneshot": True,
        "delta_oneshot": True,
        "style_pattn": True,
    },
    "F3_80k_k1": {
        "label": "F3@80k 1-shot",
        "kind": "f3",
        "variant": ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser",
        "ckpt": ROOT / "runs/F3-JOINT-DS-A-S3407/global_step_80000",
        "style_oneshot": True,
        "delta_oneshot": True,
    },
    "F3b_80000": {
        "label": "F3b@80k",
        "kind": "f3",
        "variant": ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser",
        "ckpt": ROOT / "runs/f3b_topology_ownfont_s3407/global_step_80000",
        "style_oneshot": False,
        "support_bank": ROOT / "artifacts/f0/support_bank_f3b_topology.json",
    },
    "F3b_80000_k1": {
        "label": "F3b@80k 1-shot",
        "kind": "f3",
        "variant": ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser",
        "ckpt": ROOT / "runs/f3b_topology_ownfont_s3407/global_step_80000",
        "style_oneshot": True,
        "delta_oneshot": True,
        "support_bank": ROOT / "artifacts/f0/support_bank_f3b_topology.json",
    },
    "F3bP_40000_k1": {
        "label": "F3b-P@40k 1-shot",
        "kind": "f3",
        "variant": ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser",
        "ckpt": ROOT / "runs/f3b_pattn_s3407/global_step_40000",
        "style_oneshot": True,
        "delta_oneshot": True,
        "style_pattn": True,
        "support_bank": ROOT / "artifacts/f0/support_bank_f3b_topology.json",
    },
}
F1_RUN = ROOT / "runs/F1-OFFRSI-A-S3407"
F3_VARIANT = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"
F3_RUN = ROOT / "runs/F3-JOINT-DS-A-S3407"
F2_RUN = ROOT / "runs/F2-DELTARSI-A-S3407"
TIMELINE_STEPS = [5000, 10000, 20000, 30000, 40000, 50000, 60000, 75000, 80000]
# F2 not finished 80k yet; timeline uses whichever named ckpts exist under F2_RUN.
F2_TIMELINE_CANDIDATES = [5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 75000]
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


def refuse_busy_train_gpu() -> None:
    """Refuse a physical GPU that still has an active F2-P/F3b-P train process."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if vis != "1":
        return
    # Training queue completed; only block if a train.py is still attached to GPU1.
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--id=1", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return
    for line in out.splitlines():
        if "python" in line.lower() or "train" in line.lower():
            # Heuristic: still refuse when something python-like holds GPU1 during historic P training.
            raise SystemExit("refusing CUDA_VISIBLE_DEVICES=1 (busy train/eval process detected)")


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


def pack_style_delta(
    T,
    es,
    ec,
    library,
    font: str,
    ch: str,
    cfg,
    keep,
    device,
    style_oneshot: bool,
    style_pattn: bool = False,
    delta_oneshot: bool = False,
):
    """Style Es uses k=1「永」 when style_oneshot.

    Historical *_s1 arms keep Δ on ref8 (`delta_oneshot=False`).
    True 1-shot compare arms set `delta_oneshot=True` so Es and Δ both see 永.
    """
    base = {
        "split": ["test"],
        "font_stem": [font],
        "char_cp": [cp_of(ch)],
    }
    style_refs = STYLE1_CPS if style_oneshot else REF8_CPS
    delta_refs = STYLE1_CPS if delta_oneshot else REF8_CPS
    style, _, style_seq, style_mask = T._style_conditions(
        es, {**base, "ref_chars": [style_refs]}, device, style_pattn=style_pattn
    )
    _, queries, *_ = T._style_conditions(es, {**base, "ref_chars": [delta_refs]}, device, style_pattn=False)
    structure = T._structure_features(
        es, ec, library, {**base, "ref_chars": [delta_refs]}, queries, cfg, keep, device
    )
    content = T._content_features(ec, {**base, "ref_chars": [delta_refs]}, keep, device)
    return style, structure, content, style_seq, style_mask


def sidecar_style_extra(mid: str) -> dict:
    spec = METHODS[mid]
    oneshot = bool(spec.get("style_oneshot"))
    delta_oneshot = bool(spec.get("delta_oneshot"))
    delta_k = 1 if delta_oneshot else 8
    return {
        "style_k": 1 if oneshot else 8,
        "delta_k": delta_k,
        "style_chars": "永" if oneshot else "".join(REF8),
        "delta_chars": "永" if delta_oneshot else "".join(REF8),
        "style_oneshot": oneshot,
        "delta_oneshot": delta_oneshot,
    }


def support_table(bank) -> dict:
    """_load_support_bank used to return a flat cp→list map; now wraps {table, meta}."""
    if isinstance(bank, dict) and isinstance(bank.get("table"), dict):
        return bank["table"]
    return bank


def write_sidecar(mid: str, stem: str, ch: str, style_p: Path, extra: dict | None = None) -> None:
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
    if extra:
        meta.update(extra)
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


def make_dpm_adapter(fd, torch, style_pattn: bool = False):
    class DPMAdapter(torch.nn.Module):
        def __init__(self, inner, style_pattn: bool):
            super().__init__()
            self.fd = inner
            self.style_pattn = style_pattn

        def forward(self, x_t, timesteps, cond, content_encoder_downsample_size, version):
            kw = dict(
                content_images=cond[0],
                content_encoder_downsample_size=content_encoder_downsample_size,
                style_features=cond[2],
                structure_features=cond[3],
                content_features=cond[4],
                support_tokens=cond[5],
            )
            if self.style_pattn:
                kw["style_seq_tokens"] = cond[6]
                kw["style_seq_mask"] = cond[7]
            noise, _ = self.fd(x_t, timesteps, **kw)
            return noise

    return DPMAdapter(fd, style_pattn=style_pattn)


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


def generate_f3(device: str, stems: list[str], overwrite: bool, mid: str = "F3_80k") -> None:
    import torch
    from accelerate.utils import set_seed

    spec = METHODS[mid]
    style_oneshot = bool(spec.get("style_oneshot"))
    style_pattn = bool(spec.get("style_pattn"))
    delta_oneshot = bool(spec.get("delta_oneshot"))
    bank_path = Path(spec.get("support_bank") or (ROOT / "artifacts/f0/support_bank.json"))
    n_total = len(stems) * len(STRATIFIED)
    update_status({"method": mid, "phase": "loading", "done": 0, "skipped": 0, "total": n_total, "label": spec["label"]})
    variant = spec["variant"]
    ckpt = spec["ckpt"]
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

    log(mid, f"load Es/Ec caches + LibraryEs (style_pattn={style_pattn}; bank={bank_path.name})")
    es = EsCache(ROOT / "artifacts/f0/es_spatial_f0")
    ec = EcCache(ROOT / "artifacts/f0/ec_multiscale_f0")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_fonts = sorted(split["stems"]["train"])
    library = T._LibraryEs(es, train_fonts, T._style_chars_from_cache(es))
    bank = support_table(
        T._load_support_bank(
            str(bank_path),
            SimpleNamespace(support=True, support_k=8),
        )
    )
    log(mid, f"support table keys={len(bank)} style_oneshot={style_oneshot}")
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
    model = make_dpm_adapter(fd, torch, style_pattn=style_pattn).to(device_t).eval()
    scheduler = build_ddpm_scheduler(args)
    noise_schedule = NoiseScheduleVP(schedule="discrete", betas=scheduler.betas)
    keep = torch.zeros(1, dtype=torch.bool, device=device_t)

    def pack_one(font: str, ch: str):
        style, structure, content, style_seq, style_mask = pack_style_delta(
            T,
            es,
            ec,
            library,
            font,
            ch,
            cfg,
            keep,
            device_t,
            style_oneshot,
            style_pattn=style_pattn,
            delta_oneshot=delta_oneshot,
        )
        vecs = []
        for scp in list(bank.get(cp_of(ch), []))[: cfg.support_k]:
            try:
                feats = [x.to(device_t) for x in ec.features("style", font, scp)]
            except KeyError:
                continue
            vecs.append(T._pool_ec(feats))
        pooled = torch.stack(vecs, dim=0).unsqueeze(0) if vecs else None
        return {
            "style": style,
            "structure": structure,
            "content": content,
            "support_pooled": pooled,
            "style_seq": style_seq,
            "style_mask": style_mask,
        }

    def sample_one(packed):
        set_seed(SEED)
        img = torch.zeros(1, 3, 96, 96, device=device_t)
        style = packed["style"]
        structure = packed["structure"]
        content = packed["content"]
        style_seq = packed["style_seq"]
        style_mask = packed["style_mask"]
        pooled = packed["support_pooled"]
        support = fd.support_adapter(pooled) if pooled is not None else None
        if style_pattn:
            cond = [img, img, style, structure, content, support, style_seq, style_mask]
            uncond = [
                torch.ones_like(img),
                torch.ones_like(img),
                torch.zeros_like(style),
                [torch.zeros_like(x) for x in structure],
                [torch.zeros_like(x) for x in content],
                torch.zeros_like(support) if support is not None else None,
                torch.zeros_like(style_seq),
                style_mask.clone(),
            ]
        else:
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
            write_sidecar(mid, stem, ch, style_p, extra=sidecar_style_extra(mid))
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


def f2_mid(step: int) -> str:
    return f"F2_{step}" if step != 75000 else "F2_75000"


def f2_timeline_steps() -> list[int]:
    out = []
    for step in F2_TIMELINE_CANDIDATES:
        ckpt = F2_RUN / f"global_step_{step}"
        if (ckpt / "unet.pth").is_file():
            out.append(step)
    return out


def generate_f2(device: str, stems: list[str], overwrite: bool, mid: str = "F2_75000") -> None:
    """Δ only, no Support. style_oneshot=True → Es from 永 only; Δ still ref8.
    style_pattn=True → F2-P per-ref style seq on up-path cross-attn."""
    import torch
    from accelerate.utils import set_seed

    spec = METHODS[mid]
    style_oneshot = bool(spec.get("style_oneshot"))
    style_pattn = bool(spec.get("style_pattn"))
    delta_oneshot = bool(spec.get("delta_oneshot"))
    n_total = len(stems) * len(STRATIFIED)
    update_status(
        {"method": mid, "phase": "loading", "done": 0, "skipped": 0, "total": n_total, "label": spec["label"]}
    )
    variant = spec["variant"]
    ckpt = spec["ckpt"]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(variant))
    os.chdir(variant)
    import train as T
    from scripts.hrfont_feature_cache import EcCache, EsCache
    from src.build import (
        build_content_encoder,
        build_ddpm_scheduler,
        build_style_encoder,
        build_unet,
    )
    from src.dpm_solver.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP
    from src.model import FontDiffuserModel

    log(mid, f"load Es/Ec caches + LibraryEs (F2: delta, no support; style_pattn={style_pattn})")
    es = EsCache(ROOT / "artifacts/f0/es_spatial_f0")
    ec = EcCache(ROOT / "artifacts/f0/ec_multiscale_f0")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_fonts = sorted(split["stems"]["train"])
    library = T._LibraryEs(es, train_fonts, T._style_chars_from_cache(es))
    cfg = SimpleNamespace(
        rsi_source="delta",
        delta_enabled=True,
        delta_tau=0.07,
        delta_eps_alpha=0.01,
        delta_k_max=10,
        delta_k_top=10,
        delta_mode="topk",
        seed=SEED,
        support=False,
        support_k=0,
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
    T._ban_encoder_forward(fd)
    device_t = torch.device(device)
    fd.to(device_t).eval()
    model = make_dpm_adapter(fd, torch, style_pattn=style_pattn).to(device_t).eval()
    scheduler = build_ddpm_scheduler(args)
    noise_schedule = NoiseScheduleVP(schedule="discrete", betas=scheduler.betas)
    keep = torch.zeros(1, dtype=torch.bool, device=device_t)

    def pack_one(font: str, ch: str):
        style, structure, content, style_seq, style_mask = pack_style_delta(
            T,
            es,
            ec,
            library,
            font,
            ch,
            cfg,
            keep,
            device_t,
            style_oneshot,
            style_pattn=style_pattn,
            delta_oneshot=delta_oneshot,
        )
        return {
            "style": style,
            "structure": structure,
            "content": content,
            "style_seq": style_seq,
            "style_mask": style_mask,
        }

    def sample_one(packed):
        set_seed(SEED)
        img = torch.zeros(1, 3, 96, 96, device=device_t)
        style = packed["style"]
        structure = packed["structure"]
        content = packed["content"]
        style_seq = packed["style_seq"]
        style_mask = packed["style_mask"]
        if style_pattn:
            cond = [img, img, style, structure, content, None, style_seq, style_mask]
            uncond = [
                torch.ones_like(img),
                torch.ones_like(img),
                torch.zeros_like(style),
                [torch.zeros_like(x) for x in structure],
                [torch.zeros_like(x) for x in content],
                None,
                torch.zeros_like(style_seq),
                style_mask.clone(),
            ]
        else:
            cond = [img, img, style, structure, content, None]
            uncond = [
                torch.ones_like(img),
                torch.ones_like(img),
                torch.zeros_like(style),
                [torch.zeros_like(x) for x in structure],
                [torch.zeros_like(x) for x in content],
                None,
            ]

        def get_t_input(t_continuous):
            return (t_continuous - 1.0 / noise_schedule.total_N) * 1000.0

        def model_fn(x, t_continuous):
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t_continuous, t_continuous], dim=0)
            noise_uncond, noise = model(
                x_in,
                get_t_input(t_in),
                cat_cond(uncond, cond),
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
            pred = sample_one(pack_one(stem, ch))
            pred.save(out_p)
            write_sidecar(mid, stem, ch, style_p, extra=sidecar_style_extra(mid))
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


def generate_f1(device: str, stems: list[str], overwrite: bool, mid: str = "F1_80000") -> None:
    """F1: official RSI on F0, no Support. 1-shot style (永) for Mode A fairness vs F0/E1."""
    import torch
    from accelerate.utils import set_seed

    spec = METHODS[mid]
    n_total = len(stems) * len(STRATIFIED)
    update_status(
        {"method": mid, "phase": "loading", "done": 0, "skipped": 0, "total": n_total, "label": spec["label"]}
    )
    variant = spec["variant"]
    ckpt = spec["ckpt"]
    if not (ckpt / "unet.pth").is_file():
        raise FileNotFoundError(f"F1 ckpt missing: {ckpt}")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(variant))
    os.chdir(variant)
    import train as T
    from scripts.hrfont_feature_cache import EcCache, EsCache
    from src.build import (
        build_content_encoder,
        build_ddpm_scheduler,
        build_style_encoder,
        build_unet,
    )
    from src.dpm_solver.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP
    from src.model import FontDiffuserModel

    log(mid, f"load Es/Ec + LibraryEs (F1: official RSI, no support) ckpt={ckpt.name}")
    es = EsCache(ROOT / "artifacts/f0/es_spatial_f0")
    ec = EcCache(ROOT / "artifacts/f0/ec_multiscale_f0")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_fonts = sorted(split["stems"]["train"])
    library = T._LibraryEs(es, train_fonts, T._style_chars_from_cache(es))
    cfg = SimpleNamespace(
        rsi_source="official",
        delta_enabled=False,
        delta_tau=0.07,
        delta_eps_alpha=0.01,
        delta_k_max=10,
        delta_k_top=10,
        delta_mode="topk",
        seed=SEED,
        support=False,
        support_k=0,
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
    T._ban_encoder_forward(fd)
    device_t = torch.device(device)
    fd.to(device_t).eval()
    model = make_dpm_adapter(fd, torch).to(device_t).eval()
    scheduler = build_ddpm_scheduler(args)
    noise_schedule = NoiseScheduleVP(schedule="discrete", betas=scheduler.betas)
    keep = torch.zeros(1, dtype=torch.bool, device=device_t)
    # 1-shot: official structure uses refs[0]; style Es also only 永 → fair vs F0/E1 Mode A
    ref_one = [f"u{ord('永'):04X}"]

    def pack_one(font: str, ch: str):
        samples = {
            "split": ["test"],
            "font_stem": [font],
            "char_cp": [cp_of(ch)],
            "ref_chars": [ref_one],
        }
        style, queries, *_ = T._style_conditions(es, samples, device_t)
        structure = T._structure_features(es, ec, library, samples, queries, cfg, keep, device_t)
        content = T._content_features(ec, samples, keep, device_t)
        return {"style": style, "structure": structure, "content": content}

    def sample_one(packed):
        set_seed(SEED)
        img = torch.zeros(1, 3, 96, 96, device=device_t)
        style = packed["style"]
        structure = packed["structure"]
        content = packed["content"]
        cond = [img, img, style, structure, content, None]
        uncond = [
            torch.ones_like(img),
            torch.ones_like(img),
            torch.zeros_like(style),
            [torch.zeros_like(x) for x in structure],
            [torch.zeros_like(x) for x in content],
            None,
        ]

        def get_t_input(t_continuous):
            return (t_continuous - 1.0 / noise_schedule.total_N) * 1000.0

        def model_fn(x, t_continuous):
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t_continuous, t_continuous], dim=0)
            noise_uncond, noise = model(
                x_in,
                get_t_input(t_in),
                cat_cond(uncond, cond),
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
            pred = sample_one(pack_one(stem, ch))
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
                log(
                    mid,
                    f"done={done} skipped={skipped}/{n_total} rate={rate:.2f}/s eta={eta/60:.1f}m last={stem} {ch}",
                )
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
    bank = support_table(
        T._load_support_bank(
            str(ROOT / "artifacts/f0/support_bank.json"),
            SimpleNamespace(support=True, support_k=8),
        )
    )
    log("F3_timeline", f"support table keys={len(bank)}")
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
        style, queries, *_ = T._style_conditions(es, samples, device_t)
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


def generate_f2_timeline(
    device: str,
    overwrite: bool,
    stems: list[str] | None = None,
    steps: list[int] | None = None,
    status_key: str = "F2_timeline",
) -> None:
    """Sample F2 named ckpts with train-matched Δ conditions (no Support).

    Font/step sharding is result-safe: each (font,char,step) calls set_seed(3407).
    """
    import torch
    from accelerate.utils import set_seed

    steps = list(steps) if steps is not None else f2_timeline_steps()
    if not steps:
        raise SystemExit(f"no F2 checkpoints under {F2_RUN}")
    stems = list(stems) if stems is not None else fonts()
    chars = TIMELINE_CHARS
    n_total = len(stems) * len(chars) * len(steps)
    update_status(
        {
            "method": status_key,
            "phase": "loading",
            "done": 0,
            "skipped": 0,
            "total": n_total,
            "label": f"F2 过程:{status_key}",
            "stems": stems,
            "steps": steps,
        }
    )
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(F3_VARIANT))
    os.chdir(F3_VARIANT)
    import train as T
    from scripts.hrfont_feature_cache import EcCache, EsCache
    from src.build import (
        build_content_encoder,
        build_ddpm_scheduler,
        build_style_encoder,
        build_unet,
    )
    from src.dpm_solver.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP
    from src.model import FontDiffuserModel

    log(status_key, f"load Es/Ec caches; steps={steps} stems={len(stems)}")
    es = EsCache(ROOT / "artifacts/f0/es_spatial_f0")
    ec = EcCache(ROOT / "artifacts/f0/ec_multiscale_f0")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_fonts = sorted(split["stems"]["train"])
    library = T._LibraryEs(es, train_fonts, T._style_chars_from_cache(es))
    cfg = SimpleNamespace(
        rsi_source="delta",
        delta_enabled=True,
        delta_tau=0.07,
        delta_eps_alpha=0.01,
        delta_k_max=10,
        delta_k_top=10,
        delta_mode="topk",
        seed=SEED,
        support=False,
        support_k=0,
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
    device_t = torch.device(device)
    fd = FontDiffuserModel(
        unet=build_unet(args),
        style_encoder=build_style_encoder(args),
        content_encoder=build_content_encoder(args),
    )
    T._ban_encoder_forward(fd)
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
        style, queries, *_ = T._style_conditions(es, samples, device_t)
        structure = T._structure_features(es, ec, library, samples, queries, cfg, keep, device_t)
        content = T._content_features(ec, samples, keep, device_t)
        return {"style": style, "structure": structure, "content": content}

    log(status_key, f"precompute conditions fonts={len(stems)} chars={len(chars)}")
    packed_map = {}
    for stem in stems:
        for ch in chars:
            packed_map[(stem, ch)] = pack_one(stem, ch)

    done = skipped = 0
    t0 = time.time()
    for step in steps:
        mid = f2_mid(step)
        ckpt = F2_RUN / f"global_step_{step}"
        log(status_key, f"load {ckpt}")
        fd.unet.load_state_dict(torch.load(ckpt / "unet.pth", map_location="cpu", weights_only=True))
        fd.style_encoder.load_state_dict(
            torch.load(ckpt / "style_encoder.pth", map_location="cpu", weights_only=True)
        )
        fd.content_encoder.load_state_dict(
            torch.load(ckpt / "content_encoder.pth", map_location="cpu", weights_only=True)
        )
        fd.to(device_t).eval()

        def sample_one(p):
            set_seed(SEED)
            img = torch.zeros(1, 3, 96, 96, device=device_t)
            cond = [img, img, p["style"], p["structure"], p["content"], None]
            uncond = [
                torch.ones_like(img),
                torch.ones_like(img),
                torch.zeros_like(p["style"]),
                [torch.zeros_like(x) for x in p["structure"]],
                [torch.zeros_like(x) for x in p["content"]],
                None,
            ]

            def get_t_input(t_continuous):
                return (t_continuous - 1.0 / noise_schedule.total_N) * 1000.0

            def model_fn(x, t_continuous):
                x_in = torch.cat([x, x], dim=0)
                t_in = torch.cat([t_continuous, t_continuous], dim=0)
                noise_uncond, noise = model(
                    x_in,
                    get_t_input(t_in),
                    cat_cond(uncond, cond),
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
                            "method": status_key,
                            "phase": "running",
                            "done": done,
                            "skipped": skipped,
                            "total": n_total,
                            "rate_per_s": round(rate, 3),
                            "eta_s": int(eta),
                            "last": f"{step} {stem} {ch}",
                        }
                    )
                    log(
                        status_key,
                        f"step={step} done={done} skipped={skipped}/{n_total} rate={rate:.2f}/s eta={eta/60:.1f}m",
                    )
    elapsed = time.time() - t0
    update_status(
        {
            "method": status_key,
            "phase": "done",
            "done": done,
            "skipped": skipped,
            "total": n_total,
            "elapsed_s": round(elapsed, 1),
        }
    )
    log(status_key, f"finished done={done} skipped={skipped} elapsed={elapsed:.1f}s")
    write_f2_timeline_html()


def write_f2_timeline_html() -> None:
    steps = f2_timeline_steps()
    mids = [f2_mid(s) for s in steps]
    items = []
    for stem in fonts():
        for ch in TIMELINE_CHARS:
            preds = {
                f2_mid(s): f"preds/{f2_mid(s)}/test/{stem}/test__{stem}__{cp_of(ch)}__s{SEED}.png" for s in steps
            }
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
                    "f3": f"preds/F3_80k/test/{stem}/test__{stem}__{cp_of(ch)}__s{SEED}.png",
                    "preds": preds,
                }
            )
    payload = {
        "fonts": fonts(),
        "chars": TIMELINE_CHARS,
        "steps": steps,
        "mids": mids,
        "items": items,
        "note": "mid-train F2; 80k not included until DONE",
    }
    html = f"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<meta http-equiv="refresh" content="25"/>
<title>F2 训练过程 · test16</title>
<style>
:root{{--bg:#eef1f5;--card:#fff;--line:#d5dbe3;--muted:#5c6570;--ink:#1a1a1a;--accent:#1f4a6f}}
*{{box-sizing:border-box}} body{{margin:0;font:14px/1.45 system-ui,"Noto Sans SC",sans-serif;background:var(--bg);color:var(--ink)}}
header{{background:var(--card);border-bottom:1px solid var(--line);padding:14px 18px}}
h1{{margin:0;font-size:1.2rem}} .meta{{color:var(--muted);font-size:12px;margin-top:4px}}
main{{max-width:1500px;margin:16px auto;padding:0 14px 48px}}
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
  <h1>F2 从 5k 到 75k 的变化（中间结果）</h1>
  <div class="meta">协议同 F3 时间线：test16 × 16 跨语种字 · DPM++20 / CFG7.5 / seed 3407 · <a href="./">终点评测</a> · <a href="timeline.html">F3 过程</a></div>
</header>
<main>
<div class="note">左列为 GT / F0@100k / F3@80k 对照；其后每一列是 F2 checkpoint。条件是 <b>cache Δ only</b>（无 Support），与训练 <code>--no-support</code> 一致。75k 是当前续训前最后落盘点；80k 完成后可再补一列。像素观感诊断，不是风格终局。</div>
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
  const heads=['字','Content','GT','F0@100k','F3@80k'].concat(DATA.steps.map(s=>'F2@'+(s/1000)+'k'));
  let h='<table class="g"><thead><tr>'+heads.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>';
  for (const it of items){{
    h += '<tr><td class="ch">'+it.char+'</td>';
    h += '<td><img src="'+it.content+'"/></td><td><img src="'+it.gt+'"/></td><td><img src="'+it.f0+'"/></td><td><img src="'+it.f3+'" onerror="this.style.opacity=.2"/></td>';
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
    const ms = s.methods||{{}};
    const keys = Object.keys(ms).filter(k => k==='F2_timeline' || k.startsWith('F2_timeline_'));
    let t = '';
    if (!keys.length) t = 'waiting…';
    for (const k of keys){{
      const v = ms[k]||{{}};
      const tot=v.total||0, d=(v.done||0)+(v.skipped||0);
      const pct = tot? Math.round(100*d/tot):0;
      t += k+' '+(v.phase||'')+' '+d+'/'+tot+' ('+pct+'%)';
      if (v.eta_s) t += ' ETA '+Math.round(v.eta_s/60)+'m';
      if (v.last) t += ' '+v.last;
      t += '\\n';
    }}
    document.getElementById('prog').textContent = t;
  }} catch(e) {{ document.getElementById('prog').textContent = String(e); }}
}}
tick(); setInterval(tick, 8000);
</script>
</body></html>
"""
    (OUT / "timeline_f2.html").write_text(html, encoding="utf-8")
    print("wrote", OUT / "timeline_f2.html", "items", len(items), "steps", steps)


def cmd_timeline(args: argparse.Namespace) -> None:
    refuse_busy_train_gpu()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    arm = getattr(args, "arm", "F3").upper()
    if arm == "F2":
        write_f2_timeline_html()
        if args.html_only:
            return
        stems = shard_list(fonts(), args.shard)
        steps = f2_timeline_steps()
        if getattr(args, "steps", None):
            want = {int(x) for x in args.steps.split(",") if x.strip()}
            steps = [s for s in steps if s in want]
        status_key = f"F2_timeline_{args.shard.replace('/', 'of')}"
        generate_f2_timeline(
            args.device,
            args.overwrite,
            stems=stems,
            steps=steps,
            status_key=status_key,
        )
    else:
        write_timeline_html()
        if args.html_only:
            return
        generate_f3_timeline(args.device, args.overwrite)


def cmd_generate(args: argparse.Namespace) -> None:
    refuse_busy_train_gpu()
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
        "style": "ref8 first available, prefer 永; F2/F3 *_s1: Es from 永 only, Δ still ref8",
        "note": "Same protocol as E1 formal stratified. Val16 is not used. Per-item set_seed(3407). style_oneshot splits Es k from Δ retrieve k.",
        "methods": {
            k: {
                "label": v["label"],
                "kind": v["kind"],
                "ckpt": str(v["ckpt"]),
                "style_oneshot": bool(v.get("style_oneshot", False)),
            }
            for k, v in METHODS.items()
        },
    }
    (OUT / "PROTOCOL.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_status({"protocol": "E1-stratified test16×47 seed3407", "phase": "generate"})
    mid = args.method
    kind = METHODS[mid]["kind"]
    if kind == "f3":
        generate_f3(args.device, stems, args.overwrite, mid=mid)
    elif kind == "f2":
        generate_f2(args.device, stems, args.overwrite, mid=mid)
    elif kind == "f1":
        generate_f1(args.device, stems, args.overwrite, mid=mid)
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
    refuse_busy_train_gpu()
    lpips_fn = None
    if args.lpips:
        import torch
        import lpips

        lpips_fn = lpips.LPIPS(net="alex").to(args.device).eval()
    want = [x.strip() for x in (getattr(args, "only", "") or "").split(",") if x.strip()]
    mids = want if want else list(METHODS)
    for mid in mids:
        if mid not in METHODS:
            raise SystemExit(f"unknown method {mid}")
    rows = []
    for mid in mids:
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
    by_m = defaultdict(list)
    for r in rows:
        by_m[r["method"]].append(r)
    new_methods = {}
    for mid, rs in by_m.items():
        by_b = defaultdict(list)
        for r in rs:
            by_b[r["bucket"]].append(r)
        new_methods[mid] = {
            "label": METHODS[mid]["label"],
            "overall": agg(rs),
            "by_bucket": {b: agg(by_b[b]) for b in BUCKET_ORDER if b in by_b},
        }
    summary_p = OUT / "metrics_summary.json"
    items_p = OUT / "metrics_items.json"
    if want and summary_p.is_file():
        report = json.loads(summary_p.read_text(encoding="utf-8"))
        report.setdefault("methods", {})
        report["methods"].update(new_methods)
        report["computed_at"] = utc_now()
        report["caveat"] = (
            "L1/SSIM/LPIPS vs GT are diagnostics, not the paper style claim. "
            "Mode D (*_s1) is style-k ablation with Δ still ref8."
        )
        if items_p.is_file():
            old_rows = json.loads(items_p.read_text(encoding="utf-8"))
            drop = set(want)
            rows = [r for r in old_rows if r.get("method") not in drop] + rows
        report["n_total"] = len(rows)
    else:
        report = {
            "computed_at": utc_now(),
            "n_total": len(rows),
            "caveat": "L1/SSIM/LPIPS vs GT are diagnostics, not the paper style claim. E12 still gated.",
            "methods": new_methods,
        }
    (OUT / "metrics_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "metrics_items.json").write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    update_status({"phase": "metrics_done", "n_metric_rows": len(rows), "scored": mids})
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
    """Refresh browse_index refs; prefer fair-axes HTML rebuild (keeps E1 if present)."""
    copy_refs()
    # Prefer dedicated rebuild: preserves E1 columns + Mode A/B/C UI.
    rebuild = ROOT / "scripts/rebuild_glyph_board_fair_axes.py"
    if rebuild.is_file() and (OUT / "browse_index.json").is_file():
        import runpy

        runpy.run_path(str(rebuild), run_name="__main__")
        update_status({"phase": "gallery_done", "ui": "fair_axes_v1"})
        return
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
    """Delegate to fair-axes board builder (Mode A/B/C). Keeps payload on disk first."""
    (OUT / "browse_index.json").write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    rebuild = ROOT / "scripts/rebuild_glyph_board_fair_axes.py"
    if rebuild.is_file():
        import runpy

        runpy.run_path(str(rebuild), run_name="__main__")
        return
    # Fallback: minimal stub if rebuild script missing
    (OUT / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>f03 board</title>"
        "<p>Missing scripts/rebuild_glyph_board_fair_axes.py — cannot render fair-axis board.</p>\n",
        encoding="utf-8",
    )


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
    m.add_argument("--only", default="", help="comma-separated method ids; merge into existing metrics_summary")
    m.set_defaults(func=cmd_metrics)
    gal = sub.add_parser("gallery")
    gal.set_defaults(func=cmd_gallery)
    p = sub.add_parser("progress-page")
    p.set_defaults(func=cmd_gallery)
    t = sub.add_parser("timeline")
    t.add_argument("--arm", choices=["F2", "F3", "f2", "f3"], default="F3")
    t.add_argument("--device", default="cuda:0")
    t.add_argument("--shard", default="0/1", help="font shard i/n; result-safe with per-item seed")
    t.add_argument("--steps", default="", help="optional comma steps e.g. 5000,10000")
    t.add_argument("--overwrite", action="store_true")
    t.add_argument("--html-only", action="store_true")
    t.set_defaults(func=cmd_timeline)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
