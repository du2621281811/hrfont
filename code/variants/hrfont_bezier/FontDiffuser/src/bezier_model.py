"""Hybrid raster/vector HR-Font model."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.bezier import BezierConditionBridge, BezierOutlineDecoder


class BezierFontDiffuserModel(nn.Module):
    """Wrap an existing FontDiffuser model with an editable-outline head.

    Raster callers keep using ``forward`` unchanged.  Vector training calls
    ``forward_bezier`` with frozen/cached HR-Font conditioning features.
    """

    def __init__(
        self,
        raster_model,
        style_dim=1024,
        content_channels=(3, 64, 128, 256, 256),
        point_feature_dim=6,
        d_model=256,
        decoder_layers=4,
        max_offset=0.18,
    ):
        super().__init__()
        self.raster_model = raster_model
        self.condition_bridge = BezierConditionBridge(style_dim, content_channels, d_model)
        self.point_content_proj = nn.ModuleList(nn.Conv2d(c, d_model, 1) for c in content_channels)
        self.point_structure_proj = nn.ModuleList(nn.Conv2d(c, d_model, 1) for c in content_channels)
        self.point_condition_norm = nn.LayerNorm(d_model)
        self.vector_decoder = BezierOutlineDecoder(
            point_feature_dim=point_feature_dim,
            d_model=d_model,
            layers=decoder_layers,
            max_offset=max_offset,
        )

    def forward(self, *args, **kwargs):
        return self.raster_model(*args, **kwargs)

    def forward_bezier(
        self,
        point_features,
        point_mask,
        style_tokens,
        content_features,
        structure_features=None,
        style_mask=None,
    ):
        memory, memory_padding_mask = self.condition_bridge(
            style_tokens, content_features, structure_features, style_mask
        )
        point_context = self._sample_point_conditions(
            point_features[..., :2], content_features, structure_features
        )
        return self.vector_decoder(
            point_features, point_mask, memory, memory_padding_mask, point_context
        )

    def _sample_point_conditions(self, points, content_features, structure_features=None):
        """Sample raster/Delta fields at each editable control point."""
        if len(content_features) != len(self.point_content_proj):
            raise ValueError("content feature count does not match point sampler")
        # Vector coordinates are y-up; image feature maps are y-down.
        grid = torch.stack([points[..., 0] * 2 - 1, 1 - points[..., 1] * 2], dim=-1).unsqueeze(2)
        sampled = None
        for feature, projection in zip(content_features, self.point_content_proj):
            value = F.grid_sample(
                projection(feature.float()), grid, mode="bilinear",
                padding_mode="border", align_corners=False,
            ).squeeze(-1).transpose(1, 2)
            sampled = value if sampled is None else sampled + value
        if structure_features is not None:
            if len(structure_features) != len(self.point_structure_proj):
                raise ValueError("structure feature count does not match point sampler")
            for feature, projection in zip(structure_features, self.point_structure_proj):
                value = F.grid_sample(
                    projection(feature.float()), grid, mode="bilinear",
                    padding_mode="border", align_corners=False,
                ).squeeze(-1).transpose(1, 2)
                sampled = sampled + value
        return self.point_condition_norm(sampled)

    def freeze_raster_backbone(self):
        self.raster_model.requires_grad_(False)
        self.raster_model.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        # A frozen raster teacher must not re-enter train mode through the wrapper.
        if not any(p.requires_grad for p in self.raster_model.parameters()):
            self.raster_model.eval()
        return self
