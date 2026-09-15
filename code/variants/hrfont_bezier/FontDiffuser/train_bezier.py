#!/usr/bin/env python3
"""Stage-V1 trainer for the editable HR-Font Bezier head.

The pretrained raster backbone is frozen.  In the formal ``f123_cache`` mode,
the head consumes the exact cached style/content and retrieval mean-Delta
conditions used by the F1/F2/F3 variants.  ``online`` is a lightweight fallback
without Delta, intended for smoke tests only.  This stage intentionally does
not claim topology creation.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from dataset.bezier_dataset import BezierCollateFN, BezierFontDataset
from dataset.font_dataset import FontDataset
from src import (FontDiffuserModel, SoftBezierRasterizer, bezier_training_loss,
                 build_bezier_model, build_content_encoder, build_style_encoder,
                 build_unet)
from train import (_LibraryEs, _content_features, _structure_features,
                   _style_chars_from_cache, _style_conditions, _verify_caches,
                   EcCache, EsCache)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--vector-root", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--raster-ckpt", required=True,
                   help="directory containing unet/content_encoder/style_encoder .pth files")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--resolution", type=int, default=96)
    p.add_argument("--render-resolution", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--decoder-layers", type=int, default=4)
    p.add_argument("--max-offset", type=float, default=0.18)
    p.add_argument("--nshot-min", type=int, default=1)
    p.add_argument("--nshot-max", type=int, default=8)
    p.add_argument("--condition-mode", choices=("f123_cache", "online"),
                   default="f123_cache")
    p.add_argument("--es-cache-path", default="artifacts/f0/es_spatial_f0")
    p.add_argument("--ec-cache-path", default="artifacts/f0/ec_multiscale_f0")
    p.add_argument("--rsi-source", choices=("delta", "official"), default="delta")
    p.add_argument("--delta-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--delta-tau", type=float, default=0.07)
    p.add_argument("--delta-eps-alpha", type=float, default=0.01)
    p.add_argument("--delta-k-max", type=int, default=10)
    p.add_argument("--delta-k-top", type=int, default=10)
    p.add_argument("--delta-mode", choices=("soft", "topk", "threshold"), default="topk")
    p.add_argument("--source-drop", type=float, default=0.0,
                   help="probability of dropping the active structure source")
    return p.parse_args()


def model_args(args):
    return SimpleNamespace(
        resolution=args.resolution,
        unet_channels=(64, 128, 256, 512),
        style_image_size=(args.resolution, args.resolution),
        content_image_size=(args.resolution, args.resolution),
        content_encoder_downsample_size=3,
        channel_attn=True,
        content_start_channel=64,
        style_start_channel=64,
        bezier_point_feature_dim=6,
        bezier_d_model=args.d_model,
        bezier_decoder_layers=args.decoder_layers,
        bezier_max_offset=args.max_offset,
    )


def dataset_args(args):
    return SimpleNamespace(
        data_root=args.data_root,
        nshot_min=args.nshot_min,
        nshot_max=args.nshot_max,
        eval_refs=list("永和书风骨韵天地"),
        split_manifest=args.split_manifest,
        excluded=["FZXianZTJW"],
        resolution=args.resolution,
        num_neg=0,
    )


def native_transform(resolution):
    def native(image):
        if image.size != (resolution, resolution):
            raise ValueError(f"expected native {resolution}x{resolution}, got {image.size}")
        return image
    return transforms.Compose([
        native, transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
    ])


def load_raster_model(args, device):
    cfg = model_args(args)
    raster = FontDiffuserModel(
        unet=build_unet(cfg),
        style_encoder=build_style_encoder(cfg),
        content_encoder=build_content_encoder(cfg),
    )
    root = Path(args.raster_ckpt)
    for name, module in (
        ("unet.pth", raster.unet),
        ("style_encoder.pth", raster.style_encoder),
        ("content_encoder.pth", raster.content_encoder),
    ):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        module.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    raster.to(device).requires_grad_(False).eval()
    return raster, cfg


@torch.no_grad()
def encode_conditions(model, samples, image_transform, device):
    content = samples["content_image"].to(device)
    content_last, residual = model.raster_model.content_encoder(content)
    content_features = list(residual) + [content_last]

    per_sample = []
    for paths in samples["ref_image_paths"]:
        refs = torch.stack([
            image_transform(Image.open(path).convert("RGB")) for path in paths
        ]).to(device)
        style_map, _, _ = model.raster_model.style_encoder(refs)
        per_sample.append(style_map.mean(0))
    style = torch.stack(per_sample)
    structure = [torch.zeros_like(feature) for feature in content_features]
    return style, content_features, structure


@torch.no_grad()
def encode_cached_conditions(samples, es_cache, ec_cache, library, args, device):
    """Reproduce the current F1/F2 structure-conditioning path exactly."""
    style, queries, _, _ = _style_conditions(es_cache, samples, device, style_pattn=False)
    batch = len(samples["char_cp"])
    cfg_mask = torch.zeros(batch, dtype=torch.bool)
    source_draw = torch.rand(batch) < args.source_drop
    content = _content_features(ec_cache, samples, cfg_mask, device)
    structure = _structure_features(
        es_cache, ec_cache, library, samples, queries, args, source_draw, device
    )
    if structure is None:
        structure = [torch.zeros_like(feature) for feature in content]
    return style, content, structure


def open_condition_caches(args):
    """Open verified encoder caches and materialize the retrieval library."""
    # Reuse the strict F1/F2/F3 cache lineage check.  It names the parent
    # checkpoint phase_1_ckpt_dir, while this trainer calls it raster_ckpt.
    args.phase_1_ckpt_dir = args.raster_ckpt
    _verify_caches(args)
    es_cache = EsCache(Path(args.es_cache_path))
    ec_cache = EcCache(Path(args.ec_cache_path))
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    train_fonts = sorted(manifest.get("stems", manifest)["train"])
    library = _LibraryEs(es_cache, train_fonts, _style_chars_from_cache(es_cache))
    return es_cache, ec_cache, library


def save_head(model, output, step, optimizer, args):
    directory = output / f"global_step_{step}"
    directory.mkdir(parents=True, exist_ok=False)
    torch.save(model.condition_bridge.state_dict(), directory / "condition_bridge.pth")
    torch.save(model.vector_decoder.state_dict(), directory / "vector_decoder.pth")
    torch.save({
        "content": model.point_content_proj.state_dict(),
        "structure": model.point_structure_proj.state_dict(),
        "norm": model.point_condition_norm.state_dict(),
    }, directory / "point_condition_sampler.pth")
    torch.save({"step": step, "optimizer": optimizer.state_dict()}, directory / "trainer_state.pt")
    (directory / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    raster, cfg = load_raster_model(args, device)
    model = build_bezier_model(cfg, raster).freeze_raster_backbone().to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    rasterizer = SoftBezierRasterizer(args.render_resolution, samples_per_curve=8).to(device)
    if args.condition_mode == "f123_cache":
        es_cache, ec_cache, library = open_condition_caches(args)
    else:
        es_cache = ec_cache = library = None

    transform = native_transform(args.resolution)
    raster_data = FontDataset(dataset_args(args), "train", [transform, transform, transform], scr=False)
    data = BezierFontDataset(raster_data, args.vector_root)
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=BezierCollateFN(),
        generator=torch.Generator().manual_seed(args.seed),
    )

    step = 0
    progress = tqdm(total=args.steps)
    while step < args.steps:
        for samples in loader:
            if step >= args.steps:
                break
            model.train()
            if args.condition_mode == "f123_cache":
                style, content, structure = encode_cached_conditions(
                    samples, es_cache, ec_cache, library, args, device
                )
            else:
                style, content, structure = encode_conditions(
                    model, samples, transform, device
                )
            vector_keys = (
                "point_features", "point_mask", "segment_indices", "segment_mask",
                "adjacent_segments", "adjacency_mask", "target_metrics", "target_raster_01",
                "target_point_features", "target_point_mask",
                "target_segment_indices", "target_segment_mask",
            )
            vector_batch = {key: samples[key].to(device) for key in vector_keys}
            prediction = model.forward_bezier(
                vector_batch["point_features"], vector_batch["point_mask"],
                style, content, structure,
            )
            loss, parts = bezier_training_loss(prediction, vector_batch, rasterizer)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            step += 1
            progress.update(1)
            if step % 50 == 0:
                progress.set_postfix(loss=f"{loss.item():.4f}", raster=f"{parts['raster'].item():.4f}")
            if step % args.save_every == 0:
                save_head(model, output, step, optimizer, args)
    progress.close()
    if step % args.save_every:
        save_head(model, output, step, optimizer, args)


if __name__ == "__main__":
    main()
