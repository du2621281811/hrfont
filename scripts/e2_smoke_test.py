#!/usr/bin/env python3
"""Smoke tests for Stage-A cache-only 9-token / top-10 protocol."""
from __future__ import annotations

import random
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
VARIANT = ROOT / "code/variants/cn2west_stage_a/FontDiffuser"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from hrfont_delta_v2 import DeltaConfig, compute_alpha
from hrfont_feature_cache import EC_SHAPES, EcCache, MemmapTable, key_ec, mix_cached_delta


def sample_refs(seed: int, pool: list[str], count: int = 12):
    rng = random.Random(seed)
    return [rng.sample(pool, rng.randint(1, 8)) for _ in range(count)]


def sampler_test():
    pool = [f"u{i:04X}" for i in range(338)]
    first = sample_refs(3407, pool)
    second = sample_refs(3407, pool)
    assert first == second
    assert all(1 <= len(refs) <= 8 and len(set(refs)) == len(refs) for refs in first)
    print("n-shot sampler: PASS")


def topk_alpha_test():
    torch.manual_seed(3407)
    n, dim, library = 8, 16, 40
    query = F.normalize(torch.randn(n, dim), dim=1)
    proto = F.normalize(torch.randn(library, n, dim), dim=2)
    cfg = DeltaConfig(mode="topk", k_top=10, tau=0.07, eps_alpha=1e-6)
    idx, weights, meta = compute_alpha(query, proto, 3, cfg)
    assert len(idx) == 10 and 3 not in idx
    assert torch.isfinite(weights).all()
    assert abs(float(weights.sum()) - 1.0) < 1e-5
    assert meta["n_active"] == 10
    print("top-10 alpha: PASS")


def mix_delta_test():
    weights = torch.tensor([0.7, 0.3])
    neighbors = [[torch.ones(1, 4, 2, 2), torch.ones(1, 2, 1, 1)],
                 [torch.zeros(1, 4, 2, 2), torch.zeros(1, 2, 1, 1)]]
    neutral = [torch.zeros(1, 4, 2, 2), torch.zeros(1, 2, 1, 1)]
    delta = mix_cached_delta(neighbors, weights, neutral)
    assert torch.allclose(delta[0], torch.full_like(delta[0], 0.7))
    print("mix cached delta: PASS")


def cosine_fp16_topk_parity_test():
    """512 deterministic samples: old compute_alpha vs offline-fp16 score path."""
    generator = torch.Generator().manual_seed(3407)
    library, chars, dim, checks = 228, 338, 64, 512
    # Es pooled.dat is fp16; both the old path and the table builder normalize
    # those decoded values in fp32 before taking cosine similarities.
    pooled = F.normalize(torch.randn(library, chars, dim, generator=generator).half().float(), dim=2)
    cfg = DeltaConfig(mode="topk", k_top=10, k_max=10, tau=0.07, eps_alpha=1e-6)
    exact, flips = 0, []
    for sample in range(checks):
        font = sample % library
        nref = 1 + sample % 8
        refs = torch.randperm(chars, generator=generator)[:nref]
        query = pooled[font, refs]
        prototypes = pooled[:, refs]
        old_idx, _, _ = compute_alpha(query, prototypes, font, cfg)
        fp32_scores = torch.einsum("nd,fnd->f", query.float(), prototypes.float()).div(nref)
        cached_scores = fp32_scores.half().float()
        cached_scores[font] = -torch.inf
        new_idx = torch.topk(cached_scores, k=10, sorted=True).indices.tolist()
        if set(old_idx) == set(new_idx):
            exact += 1
        elif len(flips) < 5:
            old_only = sorted(set(old_idx) - set(new_idx))
            new_only = sorted(set(new_idx) - set(old_idx))
            boundary = torch.topk(fp32_scores.masked_fill(
                torch.arange(library) == font, -torch.inf), k=11).values
            flips.append({"sample": sample, "font": font, "old_only": old_only,
                          "new_only": new_only, "margin_10_11": float(boundary[9] - boundary[10])})
    rate = exact / checks
    print(f"fp16 cosine top-10 exact-set agreement: {exact}/{checks} ({rate:.6%})")
    if flips:
        print(f"boundary flips (up to 5): {flips}")
    assert rate >= 0.99, f"top-10 agreement below 99%: {rate:.6%}"


def batched_ec_delta_parity_test():
    """Old per-key reads and new deduplicated features_many must mix identically."""
    with tempfile.TemporaryDirectory(prefix="hrfont-ec-parity-") as tmp:
        directory = Path(tmp)
        table = MemmapTable(directory)
        keys = [key_ec("content", "", "u4E00"),
                key_ec("target", "font-a", "u4E00"),
                key_ec("target", "font-b", "u4E00")]
        table.set_keys(keys)
        rng = np.random.default_rng(3407)
        for scale_i, shape in enumerate(EC_SHAPES):
            array = table.open_array(f"s{scale_i}", (len(keys), *shape), "w+")
            array[:] = rng.standard_normal(array.shape).astype(np.float16)
        table.save_manifest({"kind": "ec", "ec_checkpoint_sha256": "synthetic"})
        table.flush()
        cache = EcCache(directory)
        weights = torch.tensor([0.6, 0.4])
        old = mix_cached_delta([
            cache.features("target", "font-a", "u4E00"),
            cache.features("target", "font-b", "u4E00")], weights,
            cache.features("content", "", "u4E00"))
        fetched = cache.features_many("target", [
            ("font-a", "u4E00"), ("font-b", "u4E00"), ("font-a", "u4E00")])
        neutral = cache.features_many("content", [("", "u4E00"), ("", "u4E00")])[("", "u4E00")]
        new = mix_cached_delta([fetched[("font-a", "u4E00")],
                                fetched[("font-b", "u4E00")]], weights, neutral)
        assert all(torch.allclose(a, b, rtol=1e-6, atol=1e-7) for a, b in zip(old, new))
        assert len(fetched) == 2
    print("batched/deduplicated Ec delta parity: PASS")


def _install_stubs(torch_mod):
    try:
        __import__("diffusers")
        return False
    except ImportError:
        pass
    class ModelMixin(torch_mod.nn.Module):
        @property
        def dtype(self):
            parameter = next(self.parameters(), None)
            return parameter.dtype if parameter is not None else torch_mod.float32
    class ConfigMixin:
        pass
    diffusers = types.ModuleType("diffusers")
    diffusers.ModelMixin = ModelMixin
    config = types.ModuleType("diffusers.configuration_utils")
    config.ConfigMixin = ConfigMixin
    config.register_to_config = lambda fn: fn
    utils = types.ModuleType("diffusers.utils")
    utils.BaseOutput = object
    utils.logging = types.SimpleNamespace(get_logger=lambda name: __import__("logging").getLogger(name))
    schedulers = types.ModuleType("diffusers.schedulers")
    scheduling_ddpm = types.ModuleType("diffusers.schedulers.scheduling_ddpm")
    scheduling_ddpm.DDPMScheduler = type("DDPMScheduler", (), {})
    sys.modules.update({
        "diffusers": diffusers,
        "diffusers.configuration_utils": config,
        "diffusers.utils": utils,
        "diffusers.schedulers": schedulers,
        "diffusers.schedulers.scheduling_ddpm": scheduling_ddpm,
    })
    if "info_nce" not in sys.modules:
        info_nce = types.ModuleType("info_nce")
        info_nce.InfoNCE = type("InfoNCE", (torch_mod.nn.Module,), {
            "__init__": lambda self, *a, **k: torch_mod.nn.Module.__init__(self)})
        sys.modules["info_nce"] = info_nce
    try:
        __import__("kornia.augmentation")
    except ImportError:
        kornia = types.ModuleType("kornia")
        augmentation = types.ModuleType("kornia.augmentation")
        augmentation.RandomResizedCrop = type("RandomResizedCrop", (), {})
        kornia.augmentation = augmentation
        sys.modules["kornia"] = kornia
        sys.modules["kornia.augmentation"] = augmentation
    return True


def nine_token_forward_test():
    _install_stubs(torch)
    sys.path.insert(0, str(VARIANT))
    from src import FontDiffuserModel, build_content_encoder, build_style_encoder, build_unet
    from types import SimpleNamespace
    args = SimpleNamespace(
        resolution=96, unet_channels=(32, 64, 128, 256),
        style_image_size=(96, 96), content_image_size=(96, 96),
        content_encoder_downsample_size=3, channel_attn=True,
        content_start_channel=64, style_start_channel=64,
    )
    torch.manual_seed(3407)
    model = FontDiffuserModel(unet=build_unet(args),
                              style_encoder=build_style_encoder(args),
                              content_encoder=build_content_encoder(args)).cpu()
    calls = {"es": 0, "ec": 0}
    model.style_encoder.register_forward_pre_hook(lambda *a, **k: calls.__setitem__("es", calls["es"] + 1))
    model.content_encoder.register_forward_pre_hook(lambda *a, **k: calls.__setitem__("ec", calls["ec"] + 1))
    bsz = 2
    style = torch.randn(bsz, 1024, 3, 3)
    content = torch.randn(bsz, 3, 96, 96)
    noisy = torch.randn_like(content)
    timesteps = torch.tensor([10, 900])
    residuals = [
        torch.randn(bsz, 3, 96, 96), torch.randn(bsz, 64, 48, 48),
        torch.randn(bsz, 128, 24, 24), torch.randn(bsz, 256, 12, 12),
        torch.randn(bsz, 256, 12, 12),
    ]
    with torch.no_grad():
        out, offset = model(noisy, timesteps, content_images=content, style_features=style,
                            structure_features=residuals, content_features=residuals,
                            content_encoder_downsample_size=3)
    assert out.shape == noisy.shape and torch.isfinite(out).all()
    assert calls["es"] == 0 and calls["ec"] == 0
    tokens = style.permute(0, 2, 3, 1).reshape(bsz, 9, 1024)
    assert tokens.shape[-2] == 9
    print("9-token cache-only forward: PASS")


def source_import_checks():
    paths = [
        VARIANT / "configs/fontdiffuser.py",
        VARIANT / "dataset/font_dataset.py",
        VARIANT / "src/model.py",
        VARIANT / "train.py",
        ROOT / "scripts/hrfont_feature_cache.py",
        ROOT / "scripts/hrfont_build_e1_caches.py",
    ]
    for path in paths:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print("source import checks: PASS")


def main() -> int:
    try:
        sampler_test()
        topk_alpha_test()
        mix_delta_test()
        cosine_fp16_topk_parity_test()
        batched_ec_delta_parity_test()
        source_import_checks()
        try:
            nine_token_forward_test()
        except (ImportError, ModuleNotFoundError) as exc:
            print(f"model smoke: SKIP ({exc})")
        print("E2 smoke: PASS")
        return 0
    except Exception as exc:
        print(f"E2 smoke: FAIL ({type(exc).__name__}: {exc})")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
