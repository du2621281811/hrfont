"""SupportAdapter: project frozen Ec(support) tokens into UNet cross-attention context."""
from __future__ import annotations

import torch
import torch.nn as nn


class SupportAdapter(nn.Module):
    """Maps pooled Ec features (D) -> cross-attention tokens (context_dim)."""

    def __init__(self, in_dim: int, context_dim: int, hidden: int | None = None):
        super().__init__()
        h = hidden or max(in_dim, context_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Linear(h, context_dim),
            nn.LayerNorm(context_dim),
        )
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, support_feats: torch.Tensor) -> torch.Tensor:
        """support_feats: (B, N, in_dim) -> (B, N, context_dim)"""
        if support_feats.numel() == 0:
            return support_feats.new_zeros(support_feats.shape[0], 0, self.net[-2].out_features)
        return self.net(support_feats)
