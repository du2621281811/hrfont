"""Fixed-cardinality vector head + batched soft occupancy (F2-VEC).

See reports/F2_VEC_MULTITASK_DESIGN_20260913.md. No Python pixel loops.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


N_PRIM = 16
PTS_PER = 3
RES = 96
BETA = 8.0
STYLE_DIM = 1024
EC_DIM = 3 + 64 + 128 + 256 + 256  # 707
COND_DIM = 256


def pool_ec_batch(feats: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([x.float().mean(dim=(2, 3)) for x in feats], dim=1)


def pool_style_batch(style: torch.Tensor) -> torch.Tensor:
    if style.ndim == 4:
        return style.float().mean(dim=(2, 3))
    return style.float()


def ink_from_rgb(image: torch.Tensor) -> torch.Tensor:
    """White-bg RGB in [0, 1] -> ink occupancy [B,1,H,W]."""
    return (1.0 - image.float().mean(dim=1, keepdim=True)).clamp(0.0, 1.0)


def script_bucket(cp: str) -> int:
    token = str(cp)
    if token.startswith("u") or token.startswith("U"):
        try:
            code = int(token[1:], 16)
        except ValueError:
            return 2
    else:
        code = ord(token[0]) if token else 0
    if 0x4E00 <= code <= 0x9FFF:
        return 0
    if 0x3040 <= code <= 0x30FF:
        return 1
    return 2


def script_mean(per_sample: torch.Tensor, cps) -> torch.Tensor:
    grouped = {0: [], 1: [], 2: []}
    for i, cp in enumerate(cps):
        grouped[script_bucket(cp)].append(per_sample[i])
    parts = [torch.stack(v).mean() for v in grouped.values() if v]
    return torch.stack(parts).mean()


class VecHead(nn.Module):
    def __init__(self, style_dim: int = STYLE_DIM, ec_dim: int = EC_DIM, d: int = COND_DIM):
        super().__init__()
        self.style_proj = nn.Linear(style_dim, d)
        self.struct_proj = nn.Linear(ec_dim, d)
        self.content_proj = nn.Linear(ec_dim, d)
        self.x0_proj = nn.Linear(3, d)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, N_PRIM * 8),
        )
        self.log_sigma_img = nn.Parameter(torch.zeros(1))
        self.log_sigma_vec = nn.Parameter(torch.zeros(1))
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.3)
                nn.init.zeros_(layer.bias)

    def condition(self, style_vec, struct_vec, content_vec, x0_hint=None, x0_scale: float = 0.0):
        cond = self.style_proj(style_vec) + self.struct_proj(struct_vec) + self.content_proj(content_vec)
        if x0_hint is not None and x0_scale != 0.0:
            pooled = x0_hint.float().mean(dim=(2, 3))
            cond = cond + x0_scale * self.x0_proj(pooled)
        return self.norm(cond)

    def decode(self, cond: torch.Tensor):
        raw = self.mlp(cond).view(-1, N_PRIM, 8)
        ctrl = torch.sigmoid(raw[..., :6]).view(-1, N_PRIM, PTS_PER, 2)
        sign = torch.tanh(raw[..., 6])
        exist = torch.sigmoid(raw[..., 7])
        return ctrl, sign, exist

    def forward(self, style_vec, struct_vec, content_vec, x0_hint=None, x0_scale: float = 0.0):
        cond = self.condition(style_vec, struct_vec, content_vec, x0_hint, x0_scale)
        return self.decode(cond)


def render_occupancy(ctrl: torch.Tensor, sign: torch.Tensor, exist: torch.Tensor,
                     res: int = RES, beta: float = BETA) -> torch.Tensor:
    """Batched signed-ellipse occupancy. ctrl [B,16,3,2] in (0,1). Returns [B,1,H,W]."""
    bsz = ctrl.shape[0]
    device = ctrl.device
    dtype = ctrl.dtype
    axis = torch.linspace(0.5 / res, 1.0 - 0.5 / res, res, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack((gx, gy), dim=-1)
    center = ctrl.mean(dim=2)
    radius = (ctrl - center.unsqueeze(2)).norm(dim=-1).amax(dim=-1).clamp_min(1.0 / res)
    delta = grid.view(1, 1, res, res, 2) - center.view(bsz, N_PRIM, 1, 1, 2)
    dist = delta.norm(dim=-1)
    occ = torch.sigmoid(-beta * (dist - radius.view(bsz, N_PRIM, 1, 1)))
    signed = sign.view(bsz, N_PRIM, 1, 1) * exist.view(bsz, N_PRIM, 1, 1) * (2.0 * occ - 1.0)
    field = signed.sum(dim=1)
    return torch.sigmoid(beta * field).unsqueeze(1)


def close_loss(ctrl: torch.Tensor) -> torch.Tensor:
    p0 = ctrl[:, :, 0, :]
    p2 = ctrl[:, :, 2, :]
    loss = ctrl.new_zeros(())
    for group in range(4):
        sl = slice(4 * group, 4 * group + 4)
        a, b = p2[:, sl], p0[:, sl]
        loss = loss + (a[:, :-1] - b[:, 1:]).pow(2).mean()
        loss = loss + (a[:, -1] - b[:, 0]).pow(2).mean()
    return loss / 4.0


def thin_loss(ctrl: torch.Tensor, exist: torch.Tensor) -> torch.Tensor:
    width = (ctrl.amax(dim=2) - ctrl.amin(dim=2)).norm(dim=-1)
    return (F.relu(exist - 0.05) * F.relu(0.04 - width)).mean()


def soft_iou_per_sample(alpha: torch.Tensor, ink: torch.Tensor) -> torch.Tensor:
    dims = (1, 2, 3)
    inter = (alpha * ink).sum(dim=dims)
    union = (alpha + ink - alpha * ink).sum(dim=dims).clamp_min(1e-6)
    return inter / union


def vector_losses(alpha, ink, ctrl, exist, consist_ink=None, cps=None):
    render = (alpha - ink).abs().mean(dim=(1, 2, 3))
    iou = 1.0 - soft_iou_per_sample(alpha, ink)
    if cps is None:
        l_render = render.mean()
        l_iou = iou.mean()
    else:
        l_render = script_mean(render, cps)
        l_iou = script_mean(iou, cps)
    l_close = close_loss(ctrl)
    l_thin = thin_loss(ctrl, exist)
    l_vec = l_render + 0.5 * l_iou + 0.2 * l_close + 0.05 * l_thin
    l_consist = alpha.new_zeros(())
    if consist_ink is not None:
        l_consist = (alpha - consist_ink).abs().mean()
    return {
        "l_vec": l_vec,
        "l_render": l_render.detach(),
        "l_iou": l_iou.detach(),
        "l_close": l_close.detach(),
        "l_thin": l_thin.detach(),
        "l_consist": l_consist,
        "iou": (1.0 - iou).mean().detach(),
    }
