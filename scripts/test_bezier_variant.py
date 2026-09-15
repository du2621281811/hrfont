#!/usr/bin/env python3
"""CPU self-tests for the editable Bezier HR-Font variant."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
VARIANT = ROOT / "code" / "variants" / "hrfont_bezier" / "FontDiffuser" / "src"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bezier = load_module("hrfont_bezier_core", VARIANT / "modules" / "bezier.py")
bezier_io = load_module("hrfont_bezier_io", VARIANT / "bezier_io.py")


def square_outline(shift_x=0.0):
    endpoints = [(0.2 + shift_x, 0.2), (0.8 + shift_x, 0.2),
                 (0.8 + shift_x, 0.8), (0.2 + shift_x, 0.8)]
    points = list(endpoints)
    segments = []
    for start, end in zip(range(4), (1, 2, 3, 0)):
        p0 = np.asarray(endpoints[start])
        p1 = np.asarray(endpoints[end])
        c1 = len(points)
        points.append(tuple(p0 + (p1 - p0) / 3))
        c2 = len(points)
        points.append(tuple(p0 + 2 * (p1 - p0) / 3))
        segments.append((start, c1, c2, end))
    return torch.tensor(points, dtype=torch.float32), torch.tensor(segments, dtype=torch.long)


def test_identity_and_gradient():
    torch.manual_seed(7)
    points, segments = square_outline()
    point_features = torch.cat([points, torch.zeros(len(points), 4)], dim=1).unsqueeze(0)
    point_mask = torch.ones(1, len(points), dtype=torch.bool)
    memory = torch.randn(1, 3, 32)
    decoder = bezier.BezierOutlineDecoder(
        point_feature_dim=6, d_model=32, nhead=4, layers=1, max_offset=0.2
    )
    prediction = decoder(point_features, point_mask, memory)
    assert torch.equal(prediction.points, points.unsqueeze(0)), "zero-init must preserve template exactly"

    rasterizer = bezier.SoftBezierRasterizer(resolution=24, samples_per_curve=6)
    segment_batch = segments.unsqueeze(0)
    segment_mask = torch.ones(1, len(segments), dtype=torch.bool)
    rendered = rasterizer(prediction.points, segment_batch, segment_mask)
    assert rendered[0, 0, 12, 12] > 0.8
    assert rendered[0, 0, 1, 1] < 0.2

    shifted_points, _ = square_outline(0.08)
    target = rasterizer(shifted_points.unsqueeze(0), segment_batch, segment_mask).detach()
    batch = {
        "segment_indices": segment_batch,
        "segment_mask": segment_mask,
        "point_mask": point_mask,
        "target_raster_01": target,
        "target_metrics": torch.zeros(1, 3),
        "adjacent_segments": torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 0]]]),
        "adjacency_mask": torch.ones(1, 4, dtype=torch.bool),
        "target_point_features": torch.cat([
            shifted_points, torch.zeros(len(shifted_points), 4)
        ], dim=1).unsqueeze(0),
        "target_segment_indices": segment_batch,
        "target_segment_mask": segment_mask,
    }
    loss, parts = bezier.bezier_training_loss(prediction, batch, rasterizer)
    assert torch.isfinite(parts["boundary"]) and parts["boundary"] > 0
    loss.backward()
    grad = decoder.delta_head.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_condition_bridge_masks():
    bridge = bezier.BezierConditionBridge(
        style_dim=16, content_channels=(3, 4), d_model=32
    )
    style = torch.randn(2, 5, 16)
    style_valid = torch.tensor([[True, True, False, False, False], [True] * 5])
    content = [torch.randn(2, 3, 8, 8), torch.randn(2, 4, 4, 4)]
    structure = [torch.zeros_like(x) for x in content]
    memory, padding = bridge(style, content, structure, style_valid)
    assert memory.shape == (2, 9, 32)
    assert padding.shape == (2, 9)
    assert padding[0, :5].tolist() == [False, False, True, True, True]
    assert not padding[:, 5:].any()


def test_export():
    points, segments = square_outline()
    contour_ids = np.zeros(len(segments), dtype=np.int64)
    with tempfile.TemporaryDirectory() as directory:
        svg = Path(directory) / "A.svg"
        glif = Path(directory) / "A.glif"
        bezier_io.export_svg(svg, points.numpy(), segments.numpy(), contour_ids)
        bezier_io.export_glif(glif, "A", ord("A"), points.numpy(), segments.numpy(), contour_ids)
        assert "<path" in svg.read_text(encoding="utf-8")
        glif_text = glif.read_text(encoding="utf-8")
        assert 'type="curve"' in glif_text and 'type="line"' in glif_text and "<outline>" in glif_text


if __name__ == "__main__":
    test_identity_and_gradient()
    test_condition_bridge_masks()
    test_export()
    print("Bezier variant self-tests: PASS")
