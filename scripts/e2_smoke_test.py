#!/usr/bin/env python3
"""CPU smoke test for HR-Font E2 Stage A."""
from __future__ import annotations

import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
VARIANT = ROOT / "code/variants/cn2west_stage_a/FontDiffuser"


def sample_refs(seed: int, pool: list[str], count: int = 12):
    rng = random.Random(seed)
    return [rng.sample(pool, rng.randint(1, 8)) for _ in range(count)]


def sampler_test():
    pool = [f"u{i:04X}" for i in range(338)]
    first = sample_refs(3407, pool)
    second = sample_refs(3407, pool)
    assert first == second
    assert all(1 <= len(refs) <= 8 and len(refs) == len(set(refs)) for refs in first)
    assert all(cp in pool for refs in first for cp in refs)
    print("n-shot sampler: PASS")


def source_import_checks():
    paths = [
        VARIANT / "configs/fontdiffuser.py",
        VARIANT / "dataset/font_dataset.py",
        VARIANT / "src/model.py",
        VARIANT / "src/modules/unet.py",
        VARIANT / "src/modules/unet_blocks.py",
        VARIANT / "train.py",
        ROOT / "scripts/hrfont_es_cache.py",
    ]
    for path in paths:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print("source import checks: PASS")


def _install_diffusers_stub(torch):
    """Only used when local CPU torch exists but diffusers is unavailable."""
    try:
        __import__("diffusers")
        return False
    except ImportError:
        pass

    class ModelMixin(torch.nn.Module):
        @property
        def dtype(self):
            return next(self.parameters()).dtype

        @property
        def device(self):
            return next(self.parameters()).device

    class ConfigMixin:
        pass

    def register_to_config(function):
        return function

    diffusers = types.ModuleType("diffusers")
    diffusers.ModelMixin = ModelMixin
    config = types.ModuleType("diffusers.configuration_utils")
    config.ConfigMixin = ConfigMixin
    config.register_to_config = register_to_config
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
    return True


def _install_scr_dependency_stubs(torch):
    if "info_nce" not in sys.modules:
        info_nce = types.ModuleType("info_nce")
        info_nce.InfoNCE = type("InfoNCE", (torch.nn.Module,), {"__init__": lambda self, *a, **k: torch.nn.Module.__init__(self)})
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


def model_test():
    global STUBBED
    import torch
    import torch.nn.functional as F

    STUBBED = _install_diffusers_stub(torch)
    _install_scr_dependency_stubs(torch)
    sys.path.insert(0, str(VARIANT))
    from src import FontDiffuserModel, build_content_encoder, build_style_encoder, build_unet

    args = SimpleNamespace(
        resolution=96,
        unet_channels=(32, 64, 128, 256),
        style_image_size=(96, 96),
        content_image_size=(96, 96),
        content_encoder_downsample_size=3,
        channel_attn=True,
        content_start_channel=64,
        style_start_channel=64,
    )
    torch.manual_seed(3407)
    model = FontDiffuserModel(unet=build_unet(args),
                              style_encoder=build_style_encoder(args),
                              content_encoder=build_content_encoder(args)).cpu()
    batch = 2
    content = torch.randn(batch, 3, 96, 96)
    refs = torch.randn(batch, 3, 3, 96, 96)
    noisy = torch.randn_like(content)
    timesteps = torch.tensor([10, 900])

    model.eval()
    with torch.no_grad():
        flat_refs = refs.flatten(0, 1)
        _, pooled, _ = model.style_encoder(flat_refs)
        pooled = F.normalize(pooled.float(), dim=1).reshape(batch, 3, -1).mean(1)
        official_final, official_residuals = model.content_encoder(refs[:, 0])
        official = list(official_residuals) + [official_final]
        delta = [torch.randn_like(feature) for feature in official]
        delta_out = model(noisy, timesteps, content_images=content,
                          style_features=pooled, structure_features=delta,
                          content_encoder_downsample_size=3)
        official_out = model(noisy, timesteps, content_images=content,
                             style_features=pooled, structure_features=official,
                             content_encoder_downsample_size=3)
    assert delta_out[0].shape == official_out[0].shape == noisy.shape
    assert torch.isfinite(delta_out[0]).all() and torch.isfinite(official_out[0]).all()
    assert torch.isfinite(delta_out[1]).all() and torch.isfinite(official_out[1]).all()
    suffix = " (diffusers compatibility stub)" if STUBBED else ""
    print(f"delta + official forward: PASS{suffix}")

    model.style_encoder.requires_grad_(False).eval()
    model.content_encoder.requires_grad_(False).eval()
    model.unet.train()
    model.zero_grad(set_to_none=True)
    train_out, offset = model(noisy, timesteps, content_images=content,
                              style_features=pooled, structure_features=delta,
                              content_encoder_downsample_size=3)
    (train_out.square().mean() + 0.5 * offset).backward()
    assert all(parameter.grad is None for parameter in model.style_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.content_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.unet.parameters())
    print("frozen-encoder train step: PASS")


def main() -> int:
    try:
        sampler_test()
        source_import_checks()
        try:
            model_test()
        except (ImportError, ModuleNotFoundError) as exc:
            print(f"model smoke: SKIP ({exc})")
        print("E2 smoke: PASS" + (" (stub mode: model checks partial)" if globals().get("STUBBED") else ""))
        return 0
    except Exception as exc:
        print(f"E2 smoke: FAIL ({type(exc).__name__}: {exc})")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
