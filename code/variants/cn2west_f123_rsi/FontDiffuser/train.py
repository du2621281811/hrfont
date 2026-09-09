#!/usr/bin/env python3
"""Train HR-Font F1/F2/F3 matched arms (identity-safe RSI, cache-only Es/Ec).

Arms differ in exactly one factor each:
  F1  rsi_source=official, support=off
  F2  rsi_source=delta,    support=off   (F1 vs F2 isolates the Delta source)
  F3  rsi_source=delta,    support=on    (F2 vs F3 isolates Support)
Everything else -- init, caches, source_drop, CFG drop, batch order, RNG, schedule --
is shared code, so the arms are matched by construction rather than by convention.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from torchvision import transforms
from tqdm.auto import tqdm

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from scripts.hrfont_delta_v2 import DeltaConfig, compute_alpha
from scripts.hrfont_feature_cache import EcCache, EsCache, mix_cached_delta
from scripts.hrfont_support_adapter import SupportAdapter

from configs.fontdiffuser import get_parser
from dataset.collate_fn import CollateFN
from dataset.font_dataset import FontDataset
from src import (ContentPerceptualLoss, FontDiffuserModel, build_content_encoder,
                 build_ddpm_scheduler, build_style_encoder, build_unet)
from utils import normalize_mean_std, reNormalize_img, save_args_to_yaml, x0_from_epsilon


# (rsi_source, support) that each arm is allowed to run with.
ARM_SPEC = {
    "F1": ("official", False),
    "F2": ("delta", False),
    "F3": ("delta", True),
    "F3b": ("delta", True),
}
# Ec multi-scale channel widths (see scripts/hrfont_feature_cache.py EC_SHAPES).
EC_SCALE_CHANNELS = (3, 64, 128, 256, 256)


def get_args():
    args = get_parser().parse_args()
    args.style_image_size = (args.style_image_size, args.style_image_size)
    args.content_image_size = (args.content_image_size, args.content_image_size)
    return args


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ban_encoder_forward(model):
    def hook(_module, _inputs):
        raise RuntimeError("cache-only: Es/Ec forward is forbidden")
    model.style_encoder.register_forward_pre_hook(hook)
    model.content_encoder.register_forward_pre_hook(hook)


def _verify_caches(args):
    es_dir = Path(args.es_cache_path)
    ec_dir = Path(args.ec_cache_path)
    es_man = json.loads((es_dir / "manifest.json").read_text(encoding="utf-8"))
    ec_man = json.loads((ec_dir / "manifest.json").read_text(encoding="utf-8"))
    init = Path(args.phase_1_ckpt_dir)
    es_sha = _sha256_file(init / "style_encoder.pth")
    ec_sha = _sha256_file(init / "content_encoder.pth")
    if es_man.get("es_checkpoint_sha256") != es_sha:
        raise RuntimeError("D-A1: Es cache SHA does not match init style_encoder")
    if ec_man.get("ec_checkpoint_sha256") != ec_sha:
        raise RuntimeError("D-A1: Ec cache SHA does not match init content_encoder")
    es_prog = json.loads((es_dir / "progress.json").read_text(encoding="utf-8"))
    ec_prog = json.loads((ec_dir / "progress.json").read_text(encoding="utf-8"))
    if es_prog.get("done") != es_prog.get("total") or ec_prog.get("done") != ec_prog.get("total"):
        raise RuntimeError("encoder cache is incomplete")
    return es_sha, ec_sha


class _LibraryEs:
    """Keep train228 pooled Es in RAM so alpha does not mmap 228 fonts per sample."""

    def __init__(self, es_cache: EsCache, train_fonts: list[str], style_chars: list[str]):
        self.fonts = train_fonts
        self.font_index = {font: i for i, font in enumerate(train_fonts)}
        self.char_index = {cp: i for i, cp in enumerate(style_chars)}
        rows = []
        for font in train_fonts:
            idx = [es_cache._row("train", font, cp) for cp in style_chars]
            rows.append(np.array(es_cache.pooled[idx], copy=True))
        self.table = torch.from_numpy(np.stack(rows)).float()  # [228, 338, D]

    def prototypes(self, rchars: list[str], device) -> torch.Tensor:
        idx = [self.char_index[cp] for cp in rchars]
        return self.table[:, idx, :].to(device)


def _style_chars_from_cache(es_cache: EsCache) -> list[str]:
    first_font = None
    ordered = []
    for key in es_cache.table.keys:
        _, split, font, cp = key.split("|")
        if split != "train":
            continue
        if first_font is None:
            first_font = font
        if font != first_font:
            break
        ordered.append(cp)
    if len(ordered) != 338:
        raise RuntimeError(f"expected 338 style chars in Es cache, got {len(ordered)}")
    return ordered


def _style_conditions(es_cache: EsCache, samples, device):
    maps, queries = [], []
    for split, font, refs in zip(samples["split"], samples["font_stem"], samples["ref_chars"]):
        spatial = torch.stack([es_cache.spatial_tensor(split, font, cp) for cp in refs], 0)
        maps.append(spatial.mean(0))
        queries.append(torch.stack([
            F.normalize(es_cache.pooled_tensor(split, font, cp), dim=0) for cp in refs
        ]))
    return torch.stack(maps).to(device), queries


def _write_heartbeat(output_dir: Path, **payload):
    payload = {"ts": __import__("datetime").datetime.now().isoformat(timespec="seconds"), **payload}
    (output_dir / "heartbeat.json").write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _content_features(ec_cache: EcCache, samples, cfg_mask, device):
    rows = []
    for dropped, cp in zip(cfg_mask, samples["char_cp"]):
        feats = [tensor.to(device) for tensor in ec_cache.features("content", "", cp)]
        if bool(dropped):
            feats = [torch.zeros_like(item) for item in feats]
        rows.append(feats)
    return [torch.cat([row[scale] for row in rows]) for scale in range(len(rows[0]))]


def _structure_features(es_cache, ec_cache, library: _LibraryEs, samples, queries, cfg, source_draw, device):
    """Build the RSI structure condition, applying source_drop identically per source.

    Both branches honour `source_draw`; dropping only the delta branch (as the old
    Stage-A code did) makes the official arm see a different conditioning
    distribution and silently breaks the F1 vs F2 comparison.
    """
    fonts = samples["font_stem"]
    chars = samples["char_cp"]
    ref_chars = samples["ref_chars"]
    if cfg.rsi_source == "official":
        rows = []
        for dropped, font, refs in zip(source_draw, fonts, ref_chars):
            feats = [tensor.to(device) for tensor in ec_cache.features("style", font, refs[0])]
            if bool(dropped):
                feats = [torch.zeros_like(item) for item in feats]
            rows.append(feats)
        return [torch.cat([row[scale] for row in rows]) for scale in range(len(rows[0]))]
    if not cfg.delta_enabled:
        return None
    alpha_cfg = DeltaConfig(tau=cfg.delta_tau, eps_alpha=cfg.delta_eps_alpha,
                            k_max=cfg.delta_k_max, k_top=cfg.delta_k_top,
                            mode=cfg.delta_mode, rng_seed=cfg.seed)
    sample_features = []
    example = None
    train_fonts = library.fonts
    for dropped, font, cp, rchars, query in zip(source_draw, fonts, chars, ref_chars, queries):
        prototypes = library.prototypes(rchars, device)
        exclude = library.font_index.get(font)
        indices, weights, _ = compute_alpha(query.to(device), prototypes, exclude, alpha_cfg)
        if len(indices) == 0:
            raise RuntimeError(f"empty top-K neighborhood font={font} char={cp}")
        neighbors = [ec_cache.features("target", train_fonts[i], cp) for i in indices]
        neighbors = [[tensor.to(device) for tensor in feats] for feats in neighbors]
        neutral = [tensor.to(device) for tensor in ec_cache.features("content", "", cp)]
        example = example or neutral
        if bool(dropped):
            sample_features.append([torch.zeros_like(item) for item in example])
            continue
        sample_features.append(mix_cached_delta(neighbors, weights, neutral))
    return [torch.cat([row[scale] for row in sample_features]) for scale in range(len(example))]


def _load_support_bank(path, args) -> dict:
    if not args.support:
        return {}
    if not path:
        raise RuntimeError("--support requires --support_bank")
    bank_path = Path(path)
    if not bank_path.is_file():
        bank_path = REPO / path
    if not bank_path.is_file():
        raise RuntimeError(f"support bank not found: {path}")
    payload = json.loads(bank_path.read_text(encoding="utf-8"))
    table = payload.get("support", payload)
    if not isinstance(table, dict) or not table:
        raise RuntimeError(f"support bank is empty or malformed: {bank_path}")
    meta = {
        "mode": payload.get("mode", "fixed"),
        "sample_random_train": bool(payload.get("sample_random_train", False)),
        "k_train_min": int(payload.get("k_train_min", args.support_k)),
        "k_train_max": int(payload.get("k_train_max", args.support_k)),
    }
    return {"table": {str(k): list(v) for k, v in table.items()}, "meta": meta}


def _pool_ec(feats) -> torch.Tensor:
    """Multi-scale Ec features -> one vector (per-scale spatial mean, concatenated)."""
    return torch.cat([x.mean(dim=(2, 3)).reshape(-1) for x in feats], dim=0)


# Host-side pooled Ec vectors for support gather. Values are CPU float32 [D].
# Science-identical to per-step features()+mean; avoids re-faulting 94G mmap rows.
_SUPPORT_POOL_CACHE: dict[tuple[str, str], torch.Tensor] = {}


def _pooled_style_cached(ec_cache, font: str, scp: str) -> torch.Tensor | None:
    key = (str(font), str(scp))
    hit = _SUPPORT_POOL_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        feats = ec_cache.features("style", font, scp)
    except KeyError:
        return None
    vec = _pool_ec(feats).detach().cpu().contiguous()
    _SUPPORT_POOL_CACHE[key] = vec
    return vec


def _prewarm_support_pool(ec_cache, bank, fonts: list[str]) -> int:
    """Fault unique (font, support_cp) pooled rows once. Returns entries cached."""
    if not isinstance(bank, dict) or "table" not in bank:
        return 0
    scps = sorted({str(scp) for pool in bank["table"].values() for scp in pool})
    if not scps or not fonts:
        return 0
    before = len(_SUPPORT_POOL_CACHE)
    # Batch by font to keep memmap row indices somewhat local.
    for font in fonts:
        items = [(font, scp) for scp in scps]
        try:
            bundled = ec_cache.features_many("style", items)
        except KeyError:
            for font2, scp in items:
                _pooled_style_cached(ec_cache, font2, scp)
            continue
        for (font2, scp), feats in bundled.items():
            key = (str(font2), str(scp))
            if key not in _SUPPORT_POOL_CACHE:
                _SUPPORT_POOL_CACHE[key] = _pool_ec(feats).detach().cpu().contiguous()
    return len(_SUPPORT_POOL_CACHE) - before


def _support_tokens(ec_cache, adapter, bank, samples, support_draw, cfg, device, train: bool = True):
    """[B, K, context_dim] adapter tokens; None when support is off or fully dropped.

    Support glyphs are the target font's OWN glyphs (Ec `style` role) — F3/F3b option B.
    F3b banks may sample a random subset of the preset stroke pool during training.
    """
    if adapter is None:
        return None
    if isinstance(bank, dict) and "table" in bank:
        table = bank["table"]
        meta = bank.get("meta") or {}
    else:
        table = bank or {}
        meta = {}
    sample_random = bool(meta.get("sample_random_train")) and train and getattr(cfg, "arm", "") == "F3b"
    k_min = int(meta.get("k_train_min", cfg.support_k))
    k_max = int(meta.get("k_train_max", cfg.support_k))
    rows, width = [], 0
    for dropped, font, cp in zip(support_draw, samples["font_stem"], samples["char_cp"]):
        if bool(dropped):
            rows.append([])
            continue
        pool = list(table.get(cp, []))
        if sample_random and pool:
            k = int(torch.randint(k_min, k_max + 1, (1,)).item()) if k_max > k_min else cfg.support_k
            k = max(1, min(k, len(pool)))
            # Independent of shared training RNG stream: use a local generator seeded
            # from (seed, font, cp) would be ideal; for speed use torch randperm here
            # after support_draw was already consumed from the shared stream.
            idx = torch.randperm(len(pool))[:k].tolist()
            chosen = [pool[i] for i in idx]
        else:
            chosen = pool[: cfg.support_k]
        vecs = []
        for scp in chosen:
            vec = _pooled_style_cached(ec_cache, font, scp)
            if vec is None:
                continue
            vecs.append(vec.to(device, non_blocking=True))
        rows.append(vecs)
        width = max(width, len(vecs))
    if width == 0:
        return None
    dim = next(v[0].shape[0] for v in rows if v)
    padded = []
    for vecs in rows:
        pad = [torch.zeros(dim, device=device) for _ in range(width - len(vecs))]
        padded.append(torch.stack(vecs + pad))
    return adapter(torch.stack(padded))


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng(state: dict):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_checkpoint(model, directory: Path, optimizer, scheduler, step: int, scaler=None,
                     adapter=None):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.unet.state_dict(), directory / "unet.pth")
    torch.save(model.style_encoder.state_dict(), directory / "style_encoder.pth")
    torch.save(model.content_encoder.state_dict(), directory / "content_encoder.pth")
    torch.save(adapter.state_dict() if adapter is not None else {}, directory / "support_adapter.pth")
    payload = {"step": step, "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "rng": _rng_state()}
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, directory / "trainer_state.pt")


def _load_parent(model, directory: Path, output_dir: Path) -> None:
    """Init from F0 (StyleUpBlockNoRSI): load shared weights, leave RSI modules fresh.

    Writes `parent_load_manifest.json` so newly created keys are auditable rather than
    implied. Anything missing that is NOT an RSI module is a hard error.
    """
    unet_sd = torch.load(directory / "unet.pth", map_location="cpu")
    missing, unexpected = model.unet.load_state_dict(unet_sd, strict=False)
    rsi_prefixes = ("sc_interpreter_offsets", "dcn_deforms", "zero_convs")
    new_keys = [k for k in missing if any(s in k for s in rsi_prefixes)]
    unexplained = [k for k in missing if k not in new_keys]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parent_load_manifest.json").write_text(json.dumps({
        "parent": str(directory),
        "new_rsi_keys": new_keys,
        "unexplained_missing_keys": unexplained,
        "unexpected_keys": list(unexpected),
        "note": "F0 has no RSI modules; new_rsi_keys start fresh with zero_convs at 0.",
    }, indent=2) + "\n", encoding="utf-8")
    if unexplained:
        raise RuntimeError(f"parent init missing non-RSI keys: {unexplained[:20]}")
    if unexpected:
        raise RuntimeError(f"parent init has unexpected keys: {list(unexpected)[:20]}")
    model.style_encoder.load_state_dict(torch.load(directory / "style_encoder.pth", map_location="cpu"))
    model.content_encoder.load_state_dict(torch.load(directory / "content_encoder.pth", map_location="cpu"))


def _load_checkpoint(model, directory: Path, optimizer=None, scheduler=None, scaler=None,
                     restore_rng: bool = False, adapter=None) -> int:
    model.unet.load_state_dict(torch.load(directory / "unet.pth", map_location="cpu"))
    model.style_encoder.load_state_dict(torch.load(directory / "style_encoder.pth", map_location="cpu"))
    model.content_encoder.load_state_dict(torch.load(directory / "content_encoder.pth", map_location="cpu"))
    adapter_path = directory / "support_adapter.pth"
    if adapter is not None and adapter_path.is_file():
        payload = torch.load(adapter_path, map_location="cpu")
        if payload:
            adapter.load_state_dict(payload)
    state_path = directory / "trainer_state.pt"
    if not state_path.is_file():
        return 0
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    if restore_rng:
        _set_rng(state.get("rng") or {})
    return int(state.get("step", state.get("global_step", 0)))


def _record_config(args, es_sha: str, ec_sha: str):
    out = Path(args.output_dir)
    save_args_to_yaml(args, out / f"{args.experience_name}_config.yaml")
    digest = "none"
    if args.config_path:
        src = Path(args.config_path)
        if not src.is_file():
            src = REPO / args.config_path
        copied = out / "input_config.yaml"
        shutil.copy2(src, copied)
        digest = hashlib.sha256(copied.read_bytes()).hexdigest()
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                         capture_output=True, text=True).stdout.strip()
    (out / "run_note.txt").write_text(
        f"git_sha={sha} config_sha256={digest} es_sha={es_sha} ec_sha={ec_sha}\n",
        encoding="utf-8")


def _forward_batch(model, noise_scheduler, perceptual_loss, args, batch, style, structure,
                   content_feats, train: bool, support_tokens=None):
    target = batch["target_image"]
    nonorm_target = batch["nonorm_target_image"]
    bsz = target.shape[0]
    noise = torch.randn_like(target)
    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=target.device).long()
    noisy = noise_scheduler.add_noise(target, noise, timesteps)
    noise_pred, offset_sum = model(
        x_t=noisy, timesteps=timesteps, content_images=batch["content_image"],
        style_features=style, structure_features=structure, content_features=content_feats,
        support_tokens=support_tokens,
        content_encoder_downsample_size=args.content_encoder_downsample_size)
    diffusion = F.mse_loss(noise_pred.float(), noise.float())
    if not train:
        return diffusion, offset_sum.detach()
    x0_norm = x0_from_epsilon(noise_scheduler, noise_pred, noisy, timesteps)
    x0 = reNormalize_img(x0_norm)
    perceptual = perceptual_loss.calculate_loss(
        normalize_mean_std(x0), normalize_mean_std(nonorm_target), target.device)
    loss = diffusion + args.perceptual_coefficient * perceptual + args.offset_coefficient * (offset_sum / 2)
    return loss, diffusion.detach()


@torch.no_grad()
def _run_val(raw, es_cache, ec_cache, library, val_loader, noise_scheduler, args, device,
             bank=None):
    raw.eval()
    losses = []
    for samples in val_loader:
        target = samples["target_image"].to(device)
        samples = {**samples, "target_image": target}
        style, queries = _style_conditions(es_cache, samples, device)
        # Validation never drops: source, CFG and support are all fully on.
        source_draw = torch.zeros(target.shape[0], dtype=torch.bool, device=device)
        cfg_mask = torch.zeros_like(source_draw)
        structure = _structure_features(es_cache, ec_cache, library, samples, queries, args,
                                        source_draw, device)
        content_feats = _content_features(ec_cache, samples, cfg_mask, device)
        support = _support_tokens(ec_cache, getattr(raw, "support_adapter", None), bank or {},
                                  samples, source_draw, args, device, train=False)
        dummy = samples
        dummy["target_image"] = target
        dummy["content_image"] = samples["content_image"].to(device)
        dummy["nonorm_target_image"] = samples["nonorm_target_image"].to(device)
        loss, _ = _forward_batch(raw, noise_scheduler, None, args, dummy, style, structure,
                                 content_feats, train=False, support_tokens=support)
        losses.append(float(loss))
    raw.train()
    raw.style_encoder.eval()
    raw.content_encoder.eval()
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def _parity_gate(raw, es_cache, ec_cache, library, val_loader, noise_scheduler, args, device,
                 bank, output_dir: Path):
    """Assert the freshly-initialised RSI branch is a bit-exact no-op.

    `zero_conv` makes `skip + zero_conv(warped - skip) == skip` exactly, so step 0 must
    reproduce F0's raw-skip forward. A zero-offset DeformConv would NOT satisfy this:
    it still convolves the skip with a learned 3x3 kernel.
    """
    blocks = [b for b in raw.unet.up_blocks if hasattr(b, "rsi_enabled")]
    if not blocks:
        raise RuntimeError("parity gate: no identity-safe RSI blocks found")
    samples = next(iter(val_loader))
    samples = {**samples, "target_image": samples["target_image"].to(device),
               "content_image": samples["content_image"].to(device),
               "nonorm_target_image": samples["nonorm_target_image"].to(device)}
    style, queries = _style_conditions(es_cache, samples, device)
    draw = torch.zeros(samples["target_image"].shape[0], dtype=torch.bool, device=device)
    structure = _structure_features(es_cache, ec_cache, library, samples, queries, args, draw, device)
    content_feats = _content_features(ec_cache, samples, draw, device)
    support = _support_tokens(ec_cache, getattr(raw, "support_adapter", None), bank or {},
                              samples, draw, args, device, train=False)
    noise = torch.randn_like(samples["target_image"])
    steps = torch.zeros(samples["target_image"].shape[0], device=device).long()
    noisy = noise_scheduler.add_noise(samples["target_image"], noise, steps)

    def run():
        pred, _ = raw(x_t=noisy, timesteps=steps, content_images=samples["content_image"],
                      style_features=style, structure_features=structure,
                      content_features=content_feats, support_tokens=support,
                      content_encoder_downsample_size=args.content_encoder_downsample_size)
        return pred

    raw.eval()
    with_rsi = run()
    for b in blocks:
        b.rsi_enabled = False
    without_rsi = run()
    for b in blocks:
        b.rsi_enabled = True
    raw.train()
    raw.style_encoder.eval()
    raw.content_encoder.eval()

    max_abs = float((with_rsi - without_rsi).abs().max())
    gains = [b.rsi_gain() for b in blocks]
    report = {"max_abs_diff": max_abs, "rsi_gain_at_init": gains, "n_blocks": len(blocks),
              "passed": max_abs == 0.0}
    (output_dir / "parity_gate.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError(f"parity gate failed: max_abs_diff={max_abs}")
    print(f"parity gate OK: RSI branch is exactly identity at init (blocks={len(blocks)})", flush=True)


def main():
    args = get_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision,
                              log_with=args.report_to,
                              project_dir=str(Path(args.output_dir) / args.logging_dir))
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    es_sha, ec_sha = _verify_caches(args)
    if accelerator.is_main_process:
        _write_heartbeat(Path(args.output_dir), status="loading_caches", step=0)
    es_cache = EsCache(Path(args.es_cache_path))
    ec_cache = EcCache(Path(args.ec_cache_path))
    split = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    train_fonts = sorted(split.get("stems", split)["train"])
    if accelerator.is_main_process:
        _write_heartbeat(Path(args.output_dir), status="loading_library", step=0)
    library = _LibraryEs(es_cache, train_fonts, _style_chars_from_cache(es_cache))
    if accelerator.is_main_process:
        print(f"library Es table {tuple(library.table.shape)} fonts={len(train_fonts)}", flush=True)

    expected = ARM_SPEC[args.arm]
    if (args.rsi_source, bool(args.support)) != expected:
        raise RuntimeError(
            f"arm {args.arm} requires rsi_source={expected[0]} support={expected[1]}, "
            f"got rsi_source={args.rsi_source} support={bool(args.support)}")
    bank = _load_support_bank(args.support_bank, args)
    if args.support and args.arm == "F3b" and accelerator.is_main_process:
        _write_heartbeat(Path(args.output_dir), status="prewarm_support_pool", step=0)
        n_warm = _prewarm_support_pool(ec_cache, bank, train_fonts)
        print(f"F3b support pool prewarmed entries=+{n_warm} total={len(_SUPPORT_POOL_CACHE)}", flush=True)

    model = FontDiffuserModel(unet=build_unet(args),
                              style_encoder=build_style_encoder(args),
                              content_encoder=build_content_encoder(args))
    if args.support:
        # in_dim = concatenated per-scale means of the 5 Ec scales.
        ec_dim = sum(EC_SCALE_CHANNELS)
        # F3b: standard init (PI). Legacy F3 keeps zero-init final Linear.
        model.support_adapter = SupportAdapter(
            ec_dim, args.style_start_channel * 16, zero_init=(args.arm != "F3b")
        )
    if args.phase_1_ckpt_dir:
        _load_parent(model, Path(args.phase_1_ckpt_dir), Path(args.output_dir))
    if args.freeze_encoders:
        model.style_encoder.requires_grad_(False).eval()
        model.content_encoder.requires_grad_(False).eval()
    if args.encoder_runtime == "cache_only":
        _ban_encoder_forward(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon)
    scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                              num_warmup_steps=args.lr_warmup_steps,
                              num_training_steps=args.max_train_steps)
    noise_scheduler = build_ddpm_scheduler(args)
    perceptual_loss = ContentPerceptualLoss()

    def native(image):
        if image.size != (args.resolution, args.resolution):
            raise ValueError(f"expected native {args.resolution}x{args.resolution}, got {image.size}")
        return image

    normalized = transforms.Compose([native, transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5])])
    dataset = FontDataset(args, "train", [normalized, normalized, normalized], scr=False)
    loader = torch.utils.data.DataLoader(dataset, shuffle=True,
                                         batch_size=args.train_batch_size,
                                         collate_fn=CollateFN(),
                                         generator=torch.Generator().manual_seed(args.seed or 3407))
    val_set = FontDataset(args, "val", [normalized, normalized, normalized], scr=False)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False,
                                             batch_size=args.train_batch_size,
                                             collate_fn=CollateFN())

    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    raw = accelerator.unwrap_model(model)
    adapter = getattr(raw, "support_adapter", None)
    global_step = _load_checkpoint(raw, Path(args.resume_from), optimizer, scheduler,
                                   restore_rng=True, adapter=adapter) if args.resume_from else 0
    if args.parity_check and not args.resume_from:
        _parity_gate(raw, es_cache, ec_cache, library, val_loader, noise_scheduler, args,
                     next(raw.parameters()).device, bank, Path(args.output_dir))
    if accelerator.is_main_process:
        accelerator.init_trackers(args.experience_name)
        _record_config(args, es_sha, ec_sha)

    best = {"step": None, "val": None}
    progress = tqdm(range(args.max_train_steps), initial=global_step,
                    disable=not accelerator.is_local_main_process)
    epochs = math.ceil(args.max_train_steps / max(1, len(loader)))
    stop_file = Path(args.output_dir) / "STOP"
    for _ in range(epochs):
        for samples in loader:
            if global_step >= args.max_train_steps:
                break
            model.train()
            raw.style_encoder.eval()
            raw.content_encoder.eval()
            bsz = samples["target_image"].shape[0]
            with accelerator.accumulate(model):
                with torch.no_grad():
                    style, queries = _style_conditions(es_cache, samples, samples["target_image"].device)
                    # Draw order is fixed across arms so that a given step consumes the
                    # same RNG in F1/F2/F3; support_draw is drawn even when support is
                    # off, otherwise F2 and F3 would desynchronise after step 1.
                    source_draw = torch.rand(bsz, device=style.device) < args.source_drop
                    support_draw = torch.rand(bsz, device=style.device) < args.support_drop
                    structure = _structure_features(es_cache, ec_cache, library, samples, queries,
                                                    args, source_draw, style.device)
                    cfg_mask = torch.rand(bsz, device=style.device) < args.drop_prob
                    content_feats = _content_features(ec_cache, samples, cfg_mask, style.device)
                    style = style.clone()
                    style[cfg_mask] = 0
                # Cache features are frozen, but SupportAdapter must build an
                # autograd graph. Calling it inside no_grad freezes its initial
                # zero output for the entire F3 run.
                support = _support_tokens(ec_cache, adapter, bank, samples, support_draw,
                                          args, style.device, train=True)
                loss, _ = _forward_batch(model, noise_scheduler, perceptual_loss, args, samples,
                                         style, structure, content_feats, train=True,
                                         support_tokens=support)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                loss_value = float(loss.detach().item())
                if not math.isfinite(loss_value):
                    raise RuntimeError(f"non-finite train loss at step {global_step}: {loss_value}")
                accelerator.log({"train_loss": loss_value}, step=global_step)
                if accelerator.is_main_process:
                    if global_step <= 64:
                        with (Path(args.output_dir) / "draw_log.jsonl").open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps({
                                "step": global_step,
                                "fonts": list(samples["font_stem"]),
                                "chars": list(samples["char_cp"]),
                                "refs": list(samples["ref_chars"]),
                                "cfg": [bool(x) for x in cfg_mask.detach().cpu()],
                                "source_draw": [bool(x) for x in source_draw.detach().cpu()],
                                "support_draw": [bool(x) for x in support_draw.detach().cpu()],
                                "loss": loss_value,
                            }, ensure_ascii=False) + "\n")
                    if global_step % args.log_interval == 0 or global_step <= 20:
                        with (Path(args.output_dir) / "train_log.jsonl").open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps({
                                "step": global_step, "loss": loss_value,
                                "lr": scheduler.get_last_lr()[0],
                                # How much of the RSI branch the arm actually uses.
                                "rsi_gain": [b.rsi_gain() for b in raw.unet.up_blocks
                                             if hasattr(b, "rsi_gain")],
                            }) + "\n")
                        _write_heartbeat(Path(args.output_dir), status="running",
                                         step=global_step, loss=loss_value)
                    if global_step % args.state_interval == 0:
                        _save_checkpoint(raw, Path(args.output_dir) / "last_state",
                                         optimizer, scheduler, global_step, adapter=adapter)
                if accelerator.is_main_process and global_step % args.ckpt_interval == 0:
                    _save_checkpoint(raw, Path(args.output_dir) / f"global_step_{global_step}",
                                     optimizer, scheduler, global_step, adapter=adapter)
                    _save_checkpoint(raw, Path(args.output_dir) / "last_state",
                                     optimizer, scheduler, global_step, adapter=adapter)
                    val_loss = _run_val(raw, es_cache, ec_cache, library, val_loader,
                                        noise_scheduler, args, style.device, bank)
                    eligible = global_step >= 10000
                    if eligible and (best["val"] is None or val_loss < best["val"]):
                        best = {"step": global_step, "val": val_loss}
                        _save_checkpoint(raw, Path(args.output_dir) / "best",
                                         optimizer, scheduler, global_step, adapter=adapter)
                    (Path(args.output_dir) / "val_log.jsonl").open("a", encoding="utf-8").write(
                        json.dumps({"step": global_step, "val_loss": val_loss, "best": best}) + "\n")
                    accelerator.log({"val_loss": val_loss}, step=global_step)
                if stop_file.exists():
                    if accelerator.is_main_process:
                        _save_checkpoint(raw, Path(args.output_dir) / "stopped_step",
                                         optimizer, scheduler, global_step, adapter=adapter)
                    return
            if global_step % args.log_interval == 0:
                progress.set_postfix(loss=float(loss.detach()), lr=scheduler.get_last_lr()[0])
        if global_step >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        _save_checkpoint(raw, Path(args.output_dir) / "last_state", optimizer, scheduler, global_step, adapter=adapter)
        (Path(args.output_dir) / "DONE.json").write_text(
            json.dumps({"global_step": global_step, "status": "completed", "best": best}) + "\n",
            encoding="utf-8")
    accelerator.end_training()


if __name__ == "__main__":
    main()
