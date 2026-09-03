"""Stroke-embedding drop-in for FontDiffuser SCR supervision.

Frozen get_stroke_embedding embedder; InfoNCE (SCR-isomorphic) or cosine aux.
Expects images in [-1, 1] (FD pred_original_sample_norm / target_images).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

STROKE_ROOT = Path("/root/projects/get_stroke_embedding")
sys.path.insert(0, str(STROKE_ROOT))
from models import StrokeFeatureExtractor, StrokeStyleTokenAdapter  # noqa: E402


class StrokeSCR(nn.Module):
    def __init__(
        self,
        weights_path: str | Path | None = None,
        temperature: float = 0.07,
        mode: str = "infonce",
        device: str = "cuda",
    ):
        super().__init__()
        if mode not in ("infonce", "cos", "style_cos"):
            raise ValueError(f"mode must be infonce|cos|style_cos, got {mode}")
        self.mode = mode
        self.temperature = float(temperature)
        weights_path = Path(
            weights_path or (STROKE_ROOT / "weights/stroke_embedding.pt")
        )
        self.fe = StrokeFeatureExtractor()
        self.ad = StrokeStyleTokenAdapter()
        try:
            payload = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(weights_path, map_location="cpu")
        self.fe.load_state_dict(payload["stroke_feature_extractor"], strict=True)
        self.ad.load_state_dict(payload["stroke_style_adapter"], strict=True)
        self.fe.requires_grad_(False)
        self.ad.requires_grad_(False)
        self.to(device)
        self.eval()

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """x: BCHW in [-1, 1] → B×768 L2-normalized."""
        feats = self.fe(x.float())
        tok = self.ad(feats).mean(dim=1)
        return F.normalize(tok, dim=1, eps=1e-6)

    def forward(
        self,
        sample_imgs: torch.Tensor,
        pos_imgs: torch.Tensor,
        neg_imgs: torch.Tensor | None = None,
        nce_layers: str = "0",  # unused; kept for SCR call-site compat
    ):
        # q keeps grad; pos/neg detached at loss
        q = self.embed(sample_imgs)
        with torch.no_grad():
            k_pos = self.embed(pos_imgs)
            k_neg = None
            if neg_imgs is not None and self.mode == "infonce":
                # neg_imgs: B, N, C, H, W
                b, n, c, h, w = neg_imgs.shape
                flat = neg_imgs.reshape(b * n, c, h, w)
                k_neg = self.embed(flat).view(b, n, -1)
        return q, k_pos, k_neg

    def calculate_nce_loss(
        self,
        sample_s: torch.Tensor,
        pos_s: torch.Tensor,
        neg_s: torch.Tensor | None,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """InfoNCE, or per-sample cosine (cos / style_cos). Optional sample_weight [B]."""
        if self.mode in ("cos", "style_cos") or neg_s is None:
            per = 1.0 - (sample_s * pos_s.detach()).sum(dim=1)
            if sample_weight is not None:
                w = sample_weight.float().reshape(-1)
                denom = w.sum().clamp_min(1e-6)
                return (per * w).sum() / denom
            return per.mean()

        # sample_s: B×D, pos_s: B×D, neg_s: B×N×D
        q = sample_s
        k_pos = pos_s.detach()
        k_neg = neg_s.detach()
        tau = max(self.temperature, 1e-6)
        pos_logit = (q * k_pos).sum(dim=1, keepdim=True) / tau  # B×1
        neg_logit = torch.einsum("bd,bnd->bn", q, k_neg) / tau  # B×N
        logits = torch.cat([pos_logit, neg_logit], dim=1)  # B×(1+N)
        # target class 0 = positive
        labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
        return F.cross_entropy(logits, labels)
