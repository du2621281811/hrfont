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
from scripts.hrfont_feature_cache import EcCache, EsCache, EsLocalCache, mix_cached_delta
from scripts.hrfont_support_adapter import SupportAdapter

from configs.fontdiffuser import get_parser
from dataset.collate_fn import CollateFN
from dataset.font_dataset import FontDataset
from src import (ContentPerceptualLoss, FontDiffuserModel,
                 TCV2Cache, TCV2Global9Adapter, TCV2Head, build_content_encoder,
                 build_ddpm_scheduler, build_style_encoder, build_unet,
                 pooled_ec_features, cache_fingerprint)
from utils import normalize_mean_std, reNormalize_img, save_args_to_yaml, x0_from_epsilon


# (rsi_source, support) that each arm is allowed to run with.
ARM_SPEC = {
    "F1": ("official", False),
    "F2": ("delta", False),
    "F3": ("delta", True),
    "F3b": ("delta", True),
    "F2P": ("delta", False),
    "F3bP": ("delta", True),
    "F2RL": ("delta", False),
    "F2PRL": ("delta", False),
}
# Arms that replace up-path 9-token mean with per-ref pooled h tokens (DESIGN_F2P_F3BP).
STYLE_PATTN_ARMS = frozenset({"F2P", "F3bP", "F2PRL"})
# Per-ref Es block2 4×4 tokens (SOLUTION_STYLE_WEAKNESS). F2RL keeps global9; F2PRL drops it.
LOCAL_L128_ARMS = frozenset({"F2RL", "F2PRL"})
LOCAL_PROJ_SEED = 3407 * 1009 + 128
N_LOCAL_PER_REF = 16
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


def _clean_map_sha256(directory: str | Path) -> str:
    digest = hashlib.sha256()
    directory = Path(directory)
    for name in ("pairs_train.tsv", "pairs_val.tsv", "pairs_test.tsv", "sample_weights.json"):
        path = directory / name
        if not path.is_file():
            raise RuntimeError(f"clean-map file missing: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _ban_encoder_forward(model):
    def hook(_module, _inputs):
        raise RuntimeError("cache-only: Es/Ec forward is forbidden")
    model.style_encoder.register_forward_pre_hook(hook)
    model.content_encoder.register_forward_pre_hook(hook)


def _verify_caches(args, encoder_dir=None):
    es_dir = Path(args.es_cache_path)
    ec_dir = Path(args.ec_cache_path)
    es_man = json.loads((es_dir / "manifest.json").read_text(encoding="utf-8"))
    ec_man = json.loads((ec_dir / "manifest.json").read_text(encoding="utf-8"))
    anchor = encoder_dir or args.phase_1_ckpt_dir
    if not anchor:
        raise RuntimeError("cache verification requires --phase_1_ckpt_dir or --warm_start_from")
    init = Path(anchor)
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
    if args.arm in LOCAL_L128_ARMS:
        local_dir = Path(args.es_local_cache_path)
        local_man = json.loads((local_dir / "manifest.json").read_text(encoding="utf-8"))
        local_prog = json.loads((local_dir / "progress.json").read_text(encoding="utf-8"))
        if local_prog.get("done") != local_prog.get("total"):
            raise RuntimeError("es local cache is incomplete")
        if local_man.get("es_checkpoint_sha256") != es_sha:
            raise RuntimeError("D-A1: Es local cache SHA does not match init style_encoder")
    if args.tc_enabled:
        tc_dir = Path(args.tc_cache_path)
        tc_man = json.loads((tc_dir / "manifest.json").read_text(encoding="utf-8"))
        if tc_man.get("kind") != "tc_v2_appearance_cache":
            raise RuntimeError("TC cache manifest kind mismatch")
        if int(tc_man.get("appearance_dim", -1)) != 896:
            raise RuntimeError("TC cache appearance dimension mismatch")
        if int(tc_man.get("resolution", -1)) != int(args.resolution):
            raise RuntimeError("TC cache resolution mismatch")
        if tc_man.get("descriptor") != "VGG16 enc_1/2/3 mean+std population":
            raise RuntimeError("TC cache VGG descriptor protocol mismatch")
        if not str(tc_man.get("input_normalize", "")).startswith("RGB / 255 then ImageNet"):
            raise RuntimeError("TC cache input normalization protocol mismatch")
        if int(tc_man.get("train_target_entries", 0)) <= 0:
            raise RuntimeError("TC cache has no train target teachers")
        tc_prog = json.loads((tc_dir / "progress.json").read_text(encoding="utf-8"))
        if tc_prog.get("done") != tc_prog.get("total"):
            raise RuntimeError("TC cache is incomplete")
        split_path = Path(args.split_manifest)
        if not split_path.is_file():
            split_path = REPO / args.split_manifest
        if tc_man.get("split_manifest_sha256") != _sha256_file(split_path):
            raise RuntimeError("TC cache split-manifest binding mismatch")
        if tc_man.get("target_splits") != ["train", "val"]:
            raise RuntimeError("TC cache must not contain test target teachers")
        if not args.v0913_clean_map:
            raise RuntimeError("TC-v2 requires --v0913_clean_map for clean-pair binding")
        clean_path = Path(args.v0913_clean_map)
        if not clean_path.is_dir():
            clean_path = REPO / args.v0913_clean_map
        if tc_man.get("clean_map_sha256") != _clean_map_sha256(clean_path):
            raise RuntimeError("TC cache clean-map binding mismatch")
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


def _style_conditions(es_cache: EsCache, samples, device, style_pattn: bool = False, n_max: int = 8,
                      local_cache: EsLocalCache | None = None):
    """Build down-path style map (+ optional per-ref up-path h tokens / R-L128).

    Down-path always gets the ref-mean spatial map (legacy MCA contract).
    When style_pattn=True, also return padded per-ref pooled h tokens and a
    boolean keep-mask (True=valid) for up-path cross-attention.
    When local_cache is set, also return padded [16*n_max, 256] block2 tokens.
    F2-PRL uses both: up-path = h + L, no mean global9.
    """
    maps, queries = [], []
    seqs, masks = [], []
    local_rows, local_masks = [], []
    for split, font, refs in zip(samples["split"], samples["font_stem"], samples["ref_chars"]):
        spatial = torch.stack([es_cache.spatial_tensor(split, font, cp) for cp in refs], 0)
        maps.append(spatial.mean(0))
        pooled = torch.stack([
            F.normalize(es_cache.pooled_tensor(split, font, cp), dim=0) for cp in refs
        ], 0)
        queries.append(pooled)
        if style_pattn:
            n = int(pooled.shape[0])
            if n > n_max:
                pooled = pooled[:n_max]
                n = n_max
            if n < 1:
                raise RuntimeError(f"style_pattn requires ≥1 ref, got 0 for {font}")
            if n < n_max:
                pad = torch.zeros(n_max - n, pooled.shape[1], dtype=pooled.dtype)
                pooled = torch.cat([pooled, pad], 0)
            mask = torch.zeros(n_max, dtype=torch.bool)
            mask[:n] = True
            seqs.append(pooled)
            masks.append(mask)
        if local_cache is not None:
            toks = torch.cat([local_cache.tokens(split, font, cp) for cp in refs], 0)
            cap = N_LOCAL_PER_REF * n_max
            n_l = int(toks.shape[0])
            if n_l > cap:
                toks = toks[:cap]
                n_l = cap
            if n_l < cap:
                toks = torch.cat([toks, torch.zeros(cap - n_l, toks.shape[1], dtype=toks.dtype)], 0)
            lmask = torch.zeros(cap, dtype=torch.bool)
            lmask[:n_l] = True
            local_rows.append(toks)
            local_masks.append(lmask)
    maps_t = torch.stack(maps).to(device)
    seq_t = torch.stack(seqs).to(device) if style_pattn else None
    mask_t = torch.stack(masks).to(device) if style_pattn else None
    loc_t = torch.stack(local_rows).to(device) if local_cache is not None else None
    loc_m = torch.stack(local_masks).to(device) if local_cache is not None else None
    return maps_t, queries, seq_t, mask_t, loc_t, loc_m


def _attach_local_proj(model) -> None:
    """One shared Linear(256,1024); isolated RNG so F2 streams stay unmatched-safe."""
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(LOCAL_PROJ_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(LOCAL_PROJ_SEED)
    proj = torch.nn.Linear(256, 1024, bias=True)
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)
    model.add_module("local_style_proj", proj)


def _pack_up_style(raw, style, style_seq, style_mask, local_raw, local_mask, cfg_mask):
    """Pack up-path style tokens.

    F2-P: per-ref pooled h. F2-RL128: mean global9 + projected L. F2-PRL: h + L (no G).
    CFG zeros G/P; L is masked off (no Linear bias leak). Down-path still uses mean 3×3.
    """
    if local_raw is None:
        if style_seq is None:
            return None, None
        seq = style_seq
        mask = style_mask
        if cfg_mask is not None and bool(cfg_mask.any()):
            seq = seq.clone()
            seq[cfg_mask] = 0
        return seq, mask
    proj = raw.local_style_proj
    local_tok = proj(local_raw.to(dtype=next(proj.parameters()).dtype))
    if style_seq is not None:
        local_tok = local_tok.to(dtype=style_seq.dtype)
        seq = torch.cat([style_seq, local_tok], dim=1)
        mask = torch.cat([style_mask.to(device=style_seq.device),
                          local_mask.to(device=style_seq.device)], dim=1)
        n_p = style_seq.shape[1]
        if cfg_mask is not None and bool(cfg_mask.any()):
            seq = seq.clone()
            mask = mask.clone()
            seq[cfg_mask, :n_p] = 0
            mask[cfg_mask, n_p:] = False
        return seq, mask
    bsz = style.shape[0]
    g = style.permute(0, 2, 3, 1).reshape(bsz, -1, style.shape[1])
    g_mask = torch.ones(bsz, g.shape[1], dtype=torch.bool, device=style.device)
    local_tok = local_tok.to(dtype=g.dtype)
    seq = torch.cat([g, local_tok], dim=1)
    mask = torch.cat([g_mask, local_mask.to(device=style.device)], dim=1)
    if cfg_mask is not None and bool(cfg_mask.any()):
        mask = mask.clone()
        mask[cfg_mask, g.shape[1]:] = False
    return seq, mask


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
        extra = None
        donor_mask = getattr(cfg, "_v0913_donor_exclude", None)
        if donor_mask is not None:
            extra = donor_mask(cp)
        indices, weights, _ = compute_alpha(
            query.to(device), prototypes, exclude, alpha_cfg, extra_exclude=extra
        )
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


def _tc_conditions(tc_cache, ec_cache, samples, head, adapter, cfg_mask, device,
                    include_target: bool = True):
    """Build TC inputs from frozen cache rows, then run H/adapter with grad.

    Target rows are read only when ``include_target`` is true (training loss).
    The sampler never calls this helper and therefore cannot consume GT rows.
    """
    refs, ec_rows, masks, targets = [], [], [], []
    for split, font, cp, ref_chars in zip(samples["split"], samples["font_stem"],
                                          samples["char_cp"], samples["ref_chars"]):
        tc_cache.validate_split(split)
        row = [tc_cache.stats("ref", split, font, ref_cp) for ref_cp in ref_chars]
        if not row:
            raise RuntimeError(f"TC-v2 requires at least one ref for {font}/{cp}")
        refs.append(torch.stack(row))
        ec_rows.append(_pool_ec(ec_cache.features("content", "", cp)).squeeze(0))
        if include_target:
            targets.append(tc_cache.stats("target", split, font, cp))
    n_max = max(int(x.shape[0]) for x in refs)
    ref_batch = []
    mask_batch = []
    for row in refs:
        pad = torch.zeros(n_max - row.shape[0], row.shape[1], dtype=row.dtype)
        ref_batch.append(torch.cat([row, pad], dim=0))
        mask_batch.append(torch.cat([torch.ones(row.shape[0], dtype=torch.bool),
                                     torch.zeros(n_max - row.shape[0], dtype=torch.bool)]))
    ref_batch = torch.stack(ref_batch).to(device)
    ec_batch = torch.stack(ec_rows).to(device)
    ref_mask = torch.stack(mask_batch).to(device)
    predicted = head(ref_batch, ec_batch, ref_mask)
    residual = adapter(predicted)
    if cfg_mask is not None and bool(cfg_mask.any()):
        residual = residual.clone()
        residual[cfg_mask] = 0
    comp_loss = None
    if include_target:
        target = torch.stack(targets).to(device)
        comp_loss = F.smooth_l1_loss(predicted.float(), target.float())
    return residual, comp_loss, predicted


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
    proj = getattr(model, "local_style_proj", None)
    torch.save(proj.state_dict() if proj is not None else {}, directory / "local_style_proj.pth")
    tc_head = getattr(model, "tc_head", None)
    tc_adapter = getattr(model, "tc_global_adapter", None)
    torch.save(tc_head.state_dict() if tc_head is not None else {}, directory / "tc_head.pth")
    torch.save(tc_adapter.state_dict() if tc_adapter is not None else {},
               directory / "tc_global_adapter.pth")
    binding = getattr(model, "tc_binding", None)
    if binding is not None:
        (directory / "tc_binding.json").write_text(json.dumps(binding, indent=2) + "\n",
                                                     encoding="utf-8")
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
                     restore_rng: bool = False, adapter=None, require_tc: bool = False) -> int:
    model.unet.load_state_dict(torch.load(directory / "unet.pth", map_location="cpu"))
    model.style_encoder.load_state_dict(torch.load(directory / "style_encoder.pth", map_location="cpu"))
    model.content_encoder.load_state_dict(torch.load(directory / "content_encoder.pth", map_location="cpu"))
    adapter_path = directory / "support_adapter.pth"
    if adapter is not None and adapter_path.is_file():
        payload = torch.load(adapter_path, map_location="cpu")
        if payload:
            adapter.load_state_dict(payload)
    proj = getattr(model, "local_style_proj", None)
    proj_path = directory / "local_style_proj.pth"
    if proj is not None and proj_path.is_file():
        payload = torch.load(proj_path, map_location="cpu")
        if payload:
            proj.load_state_dict(payload, strict=True)
    if require_tc and proj is not None:
        if not proj_path.is_file():
            raise RuntimeError(f"TC resume checkpoint missing local projection: {proj_path}")
        if not isinstance(payload, dict) or not payload:
            raise RuntimeError(f"TC resume local projection is empty: {proj_path}")
    tc_head = getattr(model, "tc_head", None)
    tc_head_path = directory / "tc_head.pth"
    if tc_head is not None and not tc_head_path.is_file() and require_tc:
        raise RuntimeError(f"TC resume checkpoint missing {tc_head_path}")
    tc_head_payload = None
    if tc_head is not None and tc_head_path.is_file():
        tc_head_payload = torch.load(tc_head_path, map_location="cpu")
        if not isinstance(tc_head_payload, dict) or not tc_head_payload:
            if require_tc:
                raise RuntimeError(f"TC checkpoint is empty: {tc_head_path}")
        else:
            tc_head.load_state_dict(tc_head_payload, strict=True)
    tc_adapter = getattr(model, "tc_global_adapter", None)
    tc_adapter_path = directory / "tc_global_adapter.pth"
    if tc_adapter is not None and not tc_adapter_path.is_file() and require_tc:
        raise RuntimeError(f"TC resume checkpoint missing {tc_adapter_path}")
    tc_adapter_payload = None
    if tc_adapter is not None and tc_adapter_path.is_file():
        tc_adapter_payload = torch.load(tc_adapter_path, map_location="cpu")
        if not isinstance(tc_adapter_payload, dict) or not tc_adapter_payload:
            if require_tc:
                raise RuntimeError(f"TC checkpoint is empty: {tc_adapter_path}")
        else:
            tc_adapter.load_state_dict(tc_adapter_payload, strict=True)
    head_present = isinstance(tc_head_payload, dict) and bool(tc_head_payload)
    adapter_present = isinstance(tc_adapter_payload, dict) and bool(tc_adapter_payload)
    if head_present != adapter_present:
        raise RuntimeError("TC checkpoint must contain both non-empty H and W_out states")
    # A baseline checkpoint may legitimately contain the empty placeholders
    # written by _save_checkpoint.  Any actual TC weights, including a
    # weight-only warm-start, must carry the same cache/Ec binding as this run.
    if require_tc or head_present:
        binding_path = directory / "tc_binding.json"
        expected = getattr(model, "tc_binding", None)
        if not binding_path.is_file() or expected is None:
            raise RuntimeError("TC checkpoint missing cache binding")
        actual = json.loads(binding_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError("TC checkpoint cache/Ec binding mismatch")
    state_path = directory / "trainer_state.pt"
    if not state_path.is_file():
        if require_tc:
            raise RuntimeError("TC resume checkpoint is missing trainer_state.pt")
        return 0
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if require_tc:
        if optimizer is not None and "optimizer" not in state:
            raise RuntimeError("TC resume trainer state is missing optimizer state")
        if scheduler is not None and "scheduler" not in state:
            raise RuntimeError("TC resume trainer state is missing scheduler state")
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    if restore_rng:
        _set_rng(state.get("rng") or {})
    return int(state.get("step", state.get("global_step", 0)))


def _load_weights_only(model, directory: Path, adapter=None) -> None:
    """Warm-start weights without optimizer/scheduler/RNG/step state."""
    _load_checkpoint(model, directory, optimizer=None, scheduler=None, scaler=None,
                     restore_rng=False, adapter=adapter)


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
                   content_feats, train: bool, support_tokens=None,
                   style_seq_tokens=None, style_seq_mask=None, tc_global_residual=None,
                   tc_loss=None):
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
        style_seq_tokens=style_seq_tokens, style_seq_mask=style_seq_mask,
        tc_global_residual=tc_global_residual,
        content_encoder_downsample_size=args.content_encoder_downsample_size)
    diffusion = F.mse_loss(noise_pred.float(), noise.float())
    if not train:
        return diffusion, offset_sum.detach()
    x0_norm = x0_from_epsilon(noise_scheduler, noise_pred, noisy, timesteps)
    x0 = reNormalize_img(x0_norm)
    perceptual = perceptual_loss.calculate_loss(
        normalize_mean_std(x0), normalize_mean_std(nonorm_target), target.device)
    loss = diffusion + args.perceptual_coefficient * perceptual + args.offset_coefficient * (offset_sum / 2)
    if tc_loss is not None:
        loss = loss + args.tc_loss_coefficient * tc_loss
    return loss, diffusion.detach()


def _style_bundle(es_cache, samples, device, args, local_cache=None):
    return _style_conditions(
        es_cache, samples, device,
        style_pattn=args.arm in STYLE_PATTN_ARMS,
        local_cache=local_cache if args.arm in LOCAL_L128_ARMS else None,
    )


@torch.no_grad()
def _run_val(raw, es_cache, ec_cache, library, val_loader, noise_scheduler, args, device,
             bank=None, local_cache=None, tc_cache=None):
    raw.eval()
    losses = []
    for samples in val_loader:
        target = samples["target_image"].to(device)
        samples = {**samples, "target_image": target}
        style, queries, style_seq, style_mask, local_raw, local_mask = _style_bundle(
            es_cache, samples, device, args, local_cache)
        # Validation never drops: source, CFG and support are all fully on.
        source_draw = torch.zeros(target.shape[0], dtype=torch.bool, device=device)
        cfg_mask = torch.zeros_like(source_draw)
        structure = _structure_features(es_cache, ec_cache, library, samples, queries, args,
                                        source_draw, device)
        content_feats = _content_features(ec_cache, samples, cfg_mask, device)
        tc_residual = None
        if tc_cache is not None:
            tc_residual, _, _ = _tc_conditions(tc_cache, ec_cache, samples,
                                               raw.tc_head, raw.tc_global_adapter,
                                               cfg_mask, device, include_target=False)
        support = _support_tokens(ec_cache, getattr(raw, "support_adapter", None), bank or {},
                                  samples, source_draw, args, device, train=False)
        dummy = samples
        dummy["target_image"] = target
        dummy["content_image"] = samples["content_image"].to(device)
        dummy["nonorm_target_image"] = samples["nonorm_target_image"].to(device)
        seq, mask = _pack_up_style(raw, style, style_seq, style_mask, local_raw, local_mask, cfg_mask)
        loss, _ = _forward_batch(raw, noise_scheduler, None, args, dummy, style, structure,
                                 content_feats, train=False, support_tokens=support,
                                 style_seq_tokens=seq, style_seq_mask=mask,
                                 tc_global_residual=tc_residual)
        losses.append(float(loss))
    raw.train()
    raw.style_encoder.eval()
    raw.content_encoder.eval()
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def _parity_gate(raw, es_cache, ec_cache, library, val_loader, noise_scheduler, args, device,
                 bank, output_dir: Path, local_cache=None):
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
    style, queries, style_seq, style_mask, local_raw, local_mask = _style_bundle(
        es_cache, samples, device, args, local_cache)
    draw = torch.zeros(samples["target_image"].shape[0], dtype=torch.bool, device=device)
    structure = _structure_features(es_cache, ec_cache, library, samples, queries, args, draw, device)
    content_feats = _content_features(ec_cache, samples, draw, device)
    support = _support_tokens(ec_cache, getattr(raw, "support_adapter", None), bank or {},
                              samples, draw, args, device, train=False)
    seq, mask = _pack_up_style(raw, style, style_seq, style_mask, local_raw, local_mask, draw)
    noise = torch.randn_like(samples["target_image"])
    steps = torch.zeros(samples["target_image"].shape[0], device=device).long()
    noisy = noise_scheduler.add_noise(samples["target_image"], noise, steps)

    def run():
        pred, _ = raw(x_t=noisy, timesteps=steps, content_images=samples["content_image"],
                      style_features=style, structure_features=structure,
                      content_features=content_feats, support_tokens=support,
                      style_seq_tokens=seq, style_seq_mask=mask,
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
    if args.tc_enabled and args.arm not in {"F2", "F2RL"}:
        raise RuntimeError("TC-v2 injection requires the nine global-token F2/F2RL arms")
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

    es_sha, ec_sha = _verify_caches(args, args.resume_from or args.warm_start_from)
    if accelerator.is_main_process:
        _write_heartbeat(Path(args.output_dir), status="loading_caches", step=0)
    es_cache = EsCache(Path(args.es_cache_path))
    ec_cache = EcCache(Path(args.ec_cache_path))
    tc_cache = TCV2Cache(Path(args.tc_cache_path)) if args.tc_enabled else None
    tc_binding = None
    if tc_cache is not None:
        tc_binding = {"tc_cache_manifest_sha256": cache_fingerprint(tc_cache.manifest),
                      "ec_checkpoint_sha256": ec_cache.manifest.get("ec_checkpoint_sha256"),
                      "split_manifest_sha256": tc_cache.manifest.get("split_manifest_sha256")}
    local_cache = EsLocalCache(Path(args.es_local_cache_path)) if args.arm in LOCAL_L128_ARMS else None
    split = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    train_fonts = sorted(split.get("stems", split)["train"])
    if getattr(args, "v0913_clean_map", None):
        from scripts.v0913_clean_lib import extra_exclude_indices, load_cp_group, load_donors
        donors = load_donors(args.v0913_clean_map)
        train_fonts = donors["all"]
        cp_group = load_cp_group(args.v0913_clean_map)
        args._v0913_donor_exclude = lambda cp: extra_exclude_indices(train_fonts, cp, donors, cp_group)
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
    if args.support and args.arm in ("F3b", "F3bP") and accelerator.is_main_process:
        _write_heartbeat(Path(args.output_dir), status="prewarm_support_pool", step=0)
        n_warm = _prewarm_support_pool(ec_cache, bank, train_fonts)
        print(f"{args.arm} support pool prewarmed entries=+{n_warm} total={len(_SUPPORT_POOL_CACHE)}", flush=True)

    model = FontDiffuserModel(unet=build_unet(args),
                              style_encoder=build_style_encoder(args),
                              content_encoder=build_content_encoder(args))
    if args.support:
        # in_dim = concatenated per-scale means of the 5 Ec scales.
        ec_dim = sum(EC_SCALE_CHANNELS)
        # F3b/F3bP: standard init (PI). Legacy F3 keeps zero-init final Linear.
        model.support_adapter = SupportAdapter(
            ec_dim, args.style_start_channel * 16, zero_init=(args.arm not in ("F3b", "F3bP"))
        )
    if args.arm in LOCAL_L128_ARMS:
        _attach_local_proj(model)
    if args.tc_enabled:
        model.tc_head = TCV2Head()
        model.tc_global_adapter = TCV2Global9Adapter()
        model.tc_binding = tc_binding
        if args.tc_head_ckpt:
            head_path = Path(args.tc_head_ckpt)
            head_manifest = None
            if not head_path.is_file():
                head_path = REPO / args.tc_head_ckpt
            if head_path.is_dir():
                head_manifest = head_path / "manifest.json"
                head_path = head_path / "tc_head.pth"
            else:
                head_manifest = head_path.parent / "manifest.json"
            if not head_path.is_file():
                raise RuntimeError(f"TC head checkpoint not found: {args.tc_head_ckpt}")
            if head_manifest is None or not head_manifest.is_file():
                raise RuntimeError("TC head checkpoint requires its pretraining manifest")
            if head_manifest.is_file():
                head_meta = json.loads(head_manifest.read_text(encoding="utf-8"))
                if head_meta.get("tc_cache_manifest_sha256") != cache_fingerprint(tc_cache.manifest):
                    raise RuntimeError("TC head/cache manifest binding mismatch")
                if head_meta.get("ec_checkpoint_sha256") != ec_cache.manifest.get("ec_checkpoint_sha256"):
                    raise RuntimeError("TC head/Ec cache checkpoint binding mismatch")
            model.tc_head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
    if args.phase_1_ckpt_dir:
        _load_parent(model, Path(args.phase_1_ckpt_dir), Path(args.output_dir))
    if args.freeze_encoders:
        model.style_encoder.requires_grad_(False).eval()
        model.content_encoder.requires_grad_(False).eval()
    if args.encoder_runtime == "cache_only":
        _ban_encoder_forward(model)

    tc_trainable = []
    if args.tc_enabled:
        tc_trainable = list(model.tc_head.parameters()) + list(model.tc_global_adapter.parameters())
    tc_ids = {id(p) for p in tc_trainable}
    local_module = getattr(model, "local_style_proj", None)
    local_trainable = list(local_module.parameters()) if local_module is not None else []
    # Preserve the legacy F2RL optimizer layout when no explicit local LR is
    # supplied.  This keeps old trainer_state optimizer groups loadable.
    separate_local = False
    if local_trainable and args.local_learning_rate is not None:
        separate_local = True
    local_ids = {id(p) for p in local_trainable} if separate_local else set()
    trainable = [p for p in model.parameters()
                 if p.requires_grad and id(p) not in tc_ids and id(p) not in local_ids]
    param_groups = [{"params": trainable, "lr": args.learning_rate}]
    if separate_local:
        param_groups.append({"params": local_trainable,
                             "lr": (args.local_learning_rate
                                    if args.local_learning_rate is not None
                                    else args.learning_rate)})
    if tc_trainable:
        param_groups.append({"params": tc_trainable, "lr": args.tc_learning_rate})
    optimizer = torch.optim.AdamW(param_groups,
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
    sampler = None
    if getattr(dataset, "sample_weights", None):
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(dataset.sample_weights, dtype=torch.double),
            num_samples=len(dataset.sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed or 3407),
        )
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=sampler is None,
        sampler=sampler,
        batch_size=args.train_batch_size,
        collate_fn=CollateFN(),
        generator=None if sampler else torch.Generator().manual_seed(args.seed or 3407),
    )
    val_set = FontDataset(args, "val", [normalized, normalized, normalized], scr=False)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False,
                                             batch_size=args.train_batch_size,
                                             collate_fn=CollateFN())

    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    raw = accelerator.unwrap_model(model)
    adapter = getattr(raw, "support_adapter", None)
    if args.resume_from and args.warm_start_from:
        raise RuntimeError("--resume_from and --warm_start_from are mutually exclusive")
    if args.warm_start_from:
        _load_weights_only(raw, Path(args.warm_start_from), adapter=adapter)
    global_step = _load_checkpoint(raw, Path(args.resume_from), optimizer, scheduler,
                                   restore_rng=True, adapter=adapter,
                                   require_tc=args.tc_enabled) if args.resume_from else 0
    if args.parity_check and not args.resume_from and not args.warm_start_from:
        _parity_gate(raw, es_cache, ec_cache, library, val_loader, noise_scheduler, args,
                     next(raw.parameters()).device, bank, Path(args.output_dir),
                     local_cache=local_cache)
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
                    style, queries, style_seq, style_mask, local_raw, local_mask = _style_bundle(
                        es_cache, samples, samples["target_image"].device, args, local_cache)
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
                tc_residual, tc_loss, _ = (None, None, None)
                if tc_cache is not None:
                    # H and W_out must stay outside this no_grad region: both are
                    # trainable even though all cache encoders are frozen.
                    tc_residual, tc_loss, _ = _tc_conditions(
                        tc_cache, ec_cache, samples, raw.tc_head, raw.tc_global_adapter,
                        cfg_mask, style.device, include_target=True)
                # Cache features are frozen, but SupportAdapter / R-L128 Linear must
                # build an autograd graph. Calling them inside no_grad freezes output.
                support = _support_tokens(ec_cache, adapter, bank, samples, support_draw,
                                          args, style.device, train=True)
                seq, mask = _pack_up_style(
                    raw, style, style_seq, style_mask, local_raw, local_mask, cfg_mask)
                loss, _ = _forward_batch(model, noise_scheduler, perceptual_loss, args, samples,
                                         style, structure, content_feats, train=True,
                                         support_tokens=support,
                                         style_seq_tokens=seq, style_seq_mask=mask,
                                         tc_global_residual=tc_residual, tc_loss=tc_loss)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    clip_params = (trainable + tc_trainable +
                                   (local_trainable if separate_local else []))
                    accelerator.clip_grad_norm_(clip_params, args.max_grad_norm)
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
                                        noise_scheduler, args, style.device, bank,
                                        local_cache=local_cache, tc_cache=tc_cache)
                    eligible = global_step >= args.best_min_step
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
