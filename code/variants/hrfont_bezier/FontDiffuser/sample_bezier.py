#!/usr/bin/env python3
"""Generate one editable SVG/GLIF glyph with a trained Bezier head."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset.bezier_dataset import load_outline
from src import build_bezier_model
from src.bezier_io import export_glif, export_svg
from train import _content_features, _structure_features
from train_bezier import load_raster_model, native_transform, open_condition_caches


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raster-ckpt", required=True)
    p.add_argument("--vector-ckpt", required=True,
                   help="global_step_* directory with condition_bridge/vector_decoder weights")
    p.add_argument("--content-image", required=True)
    p.add_argument("--content-outline", required=True)
    p.add_argument("--style-images", nargs="+", required=True)
    p.add_argument("--style-chars", nargs="+",
                   help="characters/codepoints corresponding one-to-one with --style-images")
    p.add_argument("--character", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resolution", type=int, default=96)
    p.add_argument("--render-resolution", type=int, default=64)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--decoder-layers", type=int, default=4)
    p.add_argument("--max-offset", type=float, default=0.18)
    p.add_argument("--device", default="cuda")
    p.add_argument("--condition-mode", choices=("f123_cache", "online"),
                   default="f123_cache")
    p.add_argument("--es-cache-path", default="artifacts/f0/es_spatial_f0")
    p.add_argument("--ec-cache-path", default="artifacts/f0/ec_multiscale_f0")
    p.add_argument("--split-manifest", default="manifests/split_v3_228_16_16.json")
    p.add_argument("--rsi-source", choices=("delta", "official"), default="delta")
    p.add_argument("--delta-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--delta-tau", type=float, default=0.07)
    p.add_argument("--delta-eps-alpha", type=float, default=0.01)
    p.add_argument("--delta-k-max", type=int, default=10)
    p.add_argument("--delta-k-top", type=int, default=10)
    p.add_argument("--delta-mode", choices=("soft", "topk", "threshold"), default="topk")
    p.set_defaults(source_drop=0.0, seed=3407)
    # Kept for compatibility with train_bezier.model_args.
    p.set_defaults(nshot_min=1, nshot_max=8)
    return p.parse_args()


def as_codepoint(value):
    if len(value) == 1:
        return f"u{ord(value):04X}"
    if value.lower().startswith("u"):
        int(value[1:], 16)
        return "u" + value[1:].upper()
    raise ValueError(f"style character must be one Unicode character or uXXXX: {value}")


@torch.no_grad()
def main():
    args = parse_args()
    if len(args.character) != 1:
        raise ValueError("--character must contain exactly one Unicode character")
    device = torch.device(args.device)
    raster, cfg = load_raster_model(args, device)
    model = build_bezier_model(cfg, raster).freeze_raster_backbone().to(device).eval()
    vector_ckpt = Path(args.vector_ckpt)
    model.condition_bridge.load_state_dict(
        torch.load(vector_ckpt / "condition_bridge.pth", map_location="cpu"), strict=True
    )
    model.vector_decoder.load_state_dict(
        torch.load(vector_ckpt / "vector_decoder.pth", map_location="cpu"), strict=True
    )
    sampler = torch.load(vector_ckpt / "point_condition_sampler.pth", map_location="cpu")
    model.point_content_proj.load_state_dict(sampler["content"], strict=True)
    model.point_structure_proj.load_state_dict(sampler["structure"], strict=True)
    model.point_condition_norm.load_state_dict(sampler["norm"], strict=True)

    transform = native_transform(args.resolution)
    content_image = transform(Image.open(args.content_image).convert("RGB")).unsqueeze(0).to(device)
    content_last, residual = model.raster_model.content_encoder(content_image)
    content_features = list(residual) + [content_last]
    refs = torch.stack([
        transform(Image.open(path).convert("RGB")) for path in args.style_images
    ]).to(device)
    style_map, pooled, _ = model.raster_model.style_encoder(refs)
    style = style_map.mean(0, keepdim=True)
    if args.condition_mode == "f123_cache":
        if not args.style_chars or len(args.style_chars) != len(args.style_images):
            raise ValueError("f123_cache requires one --style-chars value per style image")
        es_cache, ec_cache, library = open_condition_caches(args)
        target_cp = f"u{ord(args.character):04X}"
        ref_cps = [as_codepoint(value) for value in args.style_chars]
        sample = {
            "font_stem": ["__external_style__"],
            "char_cp": [target_cp],
            "ref_chars": [ref_cps],
        }
        content_features = _content_features(
            ec_cache, sample, torch.zeros(1, dtype=torch.bool), device
        )
        queries = [F.normalize(pooled, dim=1)]
        structure = _structure_features(
            es_cache, ec_cache, library, sample, queries, args,
            torch.zeros(1, dtype=torch.bool), device,
        )
        if structure is None:
            structure = [torch.zeros_like(feature) for feature in content_features]
    else:
        structure = [torch.zeros_like(feature) for feature in content_features]

    outline = load_outline(args.content_outline)
    point_features = outline["point_features"].unsqueeze(0).to(device)
    point_mask = torch.ones(point_features.shape[:2], dtype=torch.bool, device=device)
    prediction = model.forward_bezier(
        point_features, point_mask, style, content_features, structure
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    points = prediction.points[0].cpu().numpy()
    segments = outline["segments"].numpy()
    segment_contours = outline["segment_contours"].numpy()
    metrics = prediction.metrics[0].cpu().numpy()
    stem = f"u{ord(args.character):04X}"
    export_svg(output / f"{stem}.svg", points, segments, segment_contours)
    export_glif(
        output / f"{stem}.glif", stem, ord(args.character), points, segments,
        segment_contours, advance_width=max(0.0, float(metrics[0]) * 1000),
    )
    np.savez_compressed(
        output / f"{stem}.npz", points=points, segments=segments,
        segment_contours=segment_contours, metrics=metrics,
    )
    print(f"wrote editable glyph to {output}")


if __name__ == "__main__":
    main()
