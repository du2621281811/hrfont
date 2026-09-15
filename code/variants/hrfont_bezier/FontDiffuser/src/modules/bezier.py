"""Editable cubic-Bezier head for HR-Font.

The raster diffusion model remains the visual backbone.  This module deforms a
neutral-font outline and renders it differentiably, so the output remains a
small, editable contour graph instead of a traced bitmap.

MVP scope: fixed topology.  Adding/removing contours is deliberately left to a
future topology-edit head; ``topology_compatible`` is therefore explicit in the
batch contract.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BezierPrediction:
    points: torch.Tensor
    point_delta: torch.Tensor
    metrics: torch.Tensor


class BezierConditionBridge(nn.Module):
    """Turn HR-Font style/content/Delta features into vector-decoder memory."""

    def __init__(
        self,
        style_dim: int = 1024,
        content_channels=(3, 64, 128, 256, 256),
        d_model: int = 256,
    ):
        super().__init__()
        self.style_proj = nn.Linear(style_dim, d_model)
        self.content_proj = nn.ModuleList(nn.Linear(c, d_model) for c in content_channels)
        self.structure_proj = nn.ModuleList(nn.Linear(c, d_model) for c in content_channels)
        self.type_embedding = nn.Embedding(3, d_model)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _pool(feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim == 4:
            return feature.float().mean(dim=(-2, -1))
        if feature.ndim == 3:
            return feature.float().mean(dim=1)
        if feature.ndim == 2:
            return feature.float()
        raise ValueError(f"condition feature must be rank 2/3/4, got {tuple(feature.shape)}")

    def forward(self, style_tokens, content_features, structure_features=None, style_mask=None):
        if style_tokens.ndim == 4:
            style_tokens = style_tokens.flatten(2).transpose(1, 2)
        if style_tokens.ndim != 3:
            raise ValueError("style_tokens must be [B,L,C] or [B,C,H,W]")

        memory = [self.style_proj(style_tokens.float()) + self.type_embedding.weight[0]]
        masks = []
        if style_mask is None:
            masks.append(torch.zeros(style_tokens.shape[:2], dtype=torch.bool, device=style_tokens.device))
        else:
            # Transformer uses True for padding; HR-Font masks use True for valid.
            masks.append(~style_mask.bool())

        if len(content_features) != len(self.content_proj):
            raise ValueError("content feature count does not match configured channel list")
        for feature, projection in zip(content_features, self.content_proj):
            token = projection(self._pool(feature)).unsqueeze(1)
            memory.append(token + self.type_embedding.weight[1])
            masks.append(torch.zeros(token.shape[:2], dtype=torch.bool, device=token.device))

        if structure_features is not None:
            if len(structure_features) != len(self.structure_proj):
                raise ValueError("structure feature count does not match configured channel list")
            for feature, projection in zip(structure_features, self.structure_proj):
                token = projection(self._pool(feature)).unsqueeze(1)
                memory.append(token + self.type_embedding.weight[2])
                masks.append(torch.zeros(token.shape[:2], dtype=torch.bool, device=token.device))

        return self.norm(torch.cat(memory, dim=1)), torch.cat(masks, dim=1)


class BezierOutlineDecoder(nn.Module):
    """Predict bounded edits to an existing editable contour graph.

    ``point_features`` begins with normalized x/y coordinates and may contain
    flags such as control-point type and normalized contour id.  Points are not
    generated as an unordered cloud: their original contour order is retained.
    """

    def __init__(
        self,
        point_feature_dim: int = 6,
        d_model: int = 256,
        nhead: int = 8,
        layers: int = 4,
        max_offset: float = 0.18,
        metrics_dim: int = 3,
    ):
        super().__init__()
        self.max_offset = float(max_offset)
        self.point_embed = nn.Sequential(
            nn.Linear(point_feature_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.point_norm = nn.LayerNorm(d_model)
        self.delta_head = nn.Linear(d_model, 2)
        self.metrics_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, metrics_dim)
        )

        # Exact template identity at initialization, analogous to identity-safe RSI.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, point_features, point_mask, memory, memory_padding_mask=None, point_context=None):
        if point_features.ndim != 3 or point_features.shape[-1] < 2:
            raise ValueError("point_features must be [B,N,F] with x/y in the first two channels")
        if point_mask.shape != point_features.shape[:2]:
            raise ValueError("point_mask must be [B,N]")
        if not torch.all(point_mask.any(dim=1)):
            raise ValueError("every outline must contain at least one valid point")

        query = self.point_embed(point_features.float())
        if point_context is not None:
            if point_context.shape != query.shape:
                raise ValueError("point_context must match embedded point shape [B,N,D]")
            query = query + point_context.float()
        hidden = self.decoder(
            query,
            memory.float(),
            tgt_key_padding_mask=~point_mask.bool(),
            memory_key_padding_mask=memory_padding_mask,
        )
        hidden = self.point_norm(hidden)
        delta = torch.tanh(self.delta_head(hidden)) * self.max_offset
        delta = delta * point_mask.unsqueeze(-1)
        points = point_features[..., :2].float() + delta

        weights = point_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
        # Normalized advance width, left side bearing, right side bearing.
        metrics = self.metrics_head(pooled)
        return BezierPrediction(points=points, point_delta=delta, metrics=metrics)


def gather_cubic_segments(points: torch.Tensor, segment_indices: torch.Tensor) -> torch.Tensor:
    """Gather [start, control1, control2, end] for every cubic segment."""
    if segment_indices.ndim != 3 or segment_indices.shape[-1] != 4:
        raise ValueError("segment_indices must be [B,S,4]")
    b, s, _ = segment_indices.shape
    idx = segment_indices.clamp_min(0).reshape(b, s * 4, 1).expand(-1, -1, 2)
    return torch.gather(points, 1, idx).reshape(b, s, 4, 2)


def sample_cubic_segments(segments: torch.Tensor, samples_per_curve: int = 8) -> torch.Tensor:
    """Sample cubic curves as ordered polylines, including both endpoints."""
    t = torch.linspace(0, 1, samples_per_curve + 1, device=segments.device, dtype=segments.dtype)
    t = t.view(1, 1, -1, 1)
    omt = 1 - t
    p0, p1, p2, p3 = (segments[:, :, i:i + 1] for i in range(4))
    return omt.pow(3) * p0 + 3 * omt.pow(2) * t * p1 + 3 * omt * t.pow(2) * p2 + t.pow(3) * p3


class SoftBezierRasterizer(nn.Module):
    """Differentiable filled-outline renderer based on a soft winding number.

    Coordinates are normalized to [0,1], with y increasing upward.  Correct
    outer/hole winding is part of the vector dataset contract.
    """

    def __init__(self, resolution: int = 64, samples_per_curve: int = 8, temperature: float = 0.08):
        super().__init__()
        self.resolution = int(resolution)
        self.samples_per_curve = int(samples_per_curve)
        self.temperature = float(temperature)

    def forward(self, points, segment_indices, segment_mask):
        segments = gather_cubic_segments(points, segment_indices)
        curves = sample_cubic_segments(segments, self.samples_per_curve)
        edge_a = curves[:, :, :-1].reshape(points.shape[0], -1, 2)
        edge_b = curves[:, :, 1:].reshape(points.shape[0], -1, 2)
        edge_mask = segment_mask.unsqueeze(-1).expand(-1, -1, self.samples_per_curve).reshape(points.shape[0], -1)

        axis = (torch.arange(self.resolution, device=points.device, dtype=points.dtype) + 0.5) / self.resolution
        yy, xx = torch.meshgrid(axis.flip(0), axis, indexing="ij")
        pixels = torch.stack([xx, yy], dim=-1).reshape(1, -1, 1, 2)
        va = edge_a[:, None] - pixels
        vb = edge_b[:, None] - pixels
        cross = va[..., 0] * vb[..., 1] - va[..., 1] * vb[..., 0]
        dot = (va * vb).sum(-1)
        angles = torch.atan2(cross, dot.clamp(min=-1e6, max=1e6))
        angles = angles * edge_mask[:, None].to(angles.dtype)
        winding = angles.sum(-1).abs() / (2 * torch.pi)
        ink = torch.sigmoid((winding - 0.5) / self.temperature)
        return ink.reshape(points.shape[0], 1, self.resolution, self.resolution)


def bezier_training_loss(
    prediction: BezierPrediction,
    batch: dict,
    rasterizer: SoftBezierRasterizer,
    raster_weight: float = 1.0,
    metrics_weight: float = 0.1,
    offset_weight: float = 0.01,
    continuity_weight: float = 0.1,
    boundary_weight: float = 0.2,
):
    """MVP loss without assuming neutral/target point correspondence."""
    rendered = rasterizer(prediction.points, batch["segment_indices"], batch["segment_mask"])
    target = batch["target_raster_01"].float()
    if target.shape[-2:] != rendered.shape[-2:]:
        target = F.interpolate(target, size=rendered.shape[-2:], mode="area")
    raster_loss = F.l1_loss(rendered, target)

    metric_loss = F.smooth_l1_loss(prediction.metrics, batch["target_metrics"].float())
    valid = batch["point_mask"].unsqueeze(-1).to(prediction.point_delta.dtype)
    offset_loss = (prediction.point_delta.abs() * valid).sum() / valid.sum().clamp_min(1)

    segments = gather_cubic_segments(prediction.points, batch["segment_indices"])
    # Sidecar stores pairs of adjacent segment indices; invalid pairs are masked.
    adjacency = batch.get("adjacent_segments")
    adjacency_mask = batch.get("adjacency_mask")
    if adjacency is not None and adjacency_mask is not None and adjacency_mask.any():
        b, a, _ = adjacency.shape
        left_idx = adjacency[..., 0].clamp_min(0).view(b, a, 1, 1).expand(-1, -1, 4, 2)
        right_idx = adjacency[..., 1].clamp_min(0).view(b, a, 1, 1).expand(-1, -1, 4, 2)
        left = torch.gather(segments, 1, left_idx)[..., 3, :]
        right = torch.gather(segments, 1, right_idx)[..., 0, :]
        error = (left - right).abs().sum(-1)
        continuity_loss = (error * adjacency_mask).sum() / adjacency_mask.sum().clamp_min(1)
    else:
        continuity_loss = prediction.points.new_zeros(())

    # Order-free boundary supervision: unlike pointwise regression, Chamfer does
    # not assume that neutral and target fonts use matching control points.
    if "target_segment_indices" in batch:
        predicted_boundary = sample_cubic_segments(segments, samples_per_curve=6).reshape(points_shape := prediction.points.shape[0], -1, 2)
        target_segments = gather_cubic_segments(
            batch["target_point_features"][..., :2].float(), batch["target_segment_indices"]
        )
        target_boundary = sample_cubic_segments(target_segments, samples_per_curve=6).reshape(points_shape, -1, 2)
        pred_valid = batch["segment_mask"].unsqueeze(-1).expand(-1, -1, 7).reshape(points_shape, -1)
        target_valid = batch["target_segment_mask"].unsqueeze(-1).expand(-1, -1, 7).reshape(points_shape, -1)
        distance = torch.cdist(predicted_boundary, target_boundary, p=2)
        large = torch.finfo(distance.dtype).max / 100
        p_to_t = distance.masked_fill(~target_valid[:, None], large).min(-1).values
        t_to_p = distance.masked_fill(~pred_valid[:, :, None], large).min(-2).values
        boundary_loss = (
            (p_to_t * pred_valid).sum() / pred_valid.sum().clamp_min(1)
            + (t_to_p * target_valid).sum() / target_valid.sum().clamp_min(1)
        )
    else:
        boundary_loss = prediction.points.new_zeros(())

    total = (
        raster_weight * raster_loss
        + metrics_weight * metric_loss
        + offset_weight * offset_loss
        + continuity_weight * continuity_loss
        + boundary_weight * boundary_loss
    )
    return total, {
        "raster": raster_loss.detach(),
        "metrics": metric_loss.detach(),
        "offset": offset_loss.detach(),
        "continuity": continuity_loss.detach(),
        "boundary": boundary_loss.detach(),
        "rendered": rendered.detach(),
    }
