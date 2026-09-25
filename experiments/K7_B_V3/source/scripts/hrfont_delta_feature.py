#!/usr/bin/env python3
"""Shared one-reference alpha and feature-delta utilities for Stage A MVP."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

STYLE_CHAR = "永"
TOP_M = 3
TAU = 0.07


def image_path(root: Path, split: str, role: str, font: str, ch: str) -> Path:
    key = f"u{ord(ch):04X}"
    if role == "ContentImage":
        return root / split / role / f"{key}.jpg"
    return root / split / role / font / f"{font}+{key}.jpg"


@torch.no_grad()
def build_style_prototypes(
    style_encoder: torch.nn.Module,
    data_root: Path,
    split: str,
    fonts: Iterable[str],
    load_image: Callable[[Path], torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Encode the same single reference glyph for every font."""
    prototypes: dict[str, torch.Tensor] = {}
    for font in fonts:
        path = image_path(data_root, split, "StyleImage", font, STYLE_CHAR)
        if not path.exists():
            continue
        feat, _, _ = style_encoder(load_image(path).unsqueeze(0).to(device))
        prototypes[font] = feat.flatten(1).mean(0).float().cpu()
    return prototypes


def alpha_topk(
    query: torch.Tensor,
    prototypes: dict[str, torch.Tensor],
    candidates: Iterable[str],
    *,
    top_m: int = TOP_M,
    tau: float = TAU,
) -> list[tuple[str, float]]:
    scores: list[tuple[str, float]] = []
    q = query.float().flatten()
    for font in candidates:
        value = prototypes.get(font)
        if value is None:
            continue
        score = F.cosine_similarity(q[None], value.float().flatten()[None]).item()
        scores.append((font, float(score)))
    scores.sort(key=lambda item: (-item[1], item[0]))
    top = scores[:top_m]
    if not top:
        raise RuntimeError("no alpha candidates")
    weights = torch.softmax(torch.tensor([score / tau for _, score in top]), dim=0)
    return [(font, float(weights[i])) for i, (font, _) in enumerate(top)]


@torch.no_grad()
def feature_delta(
    content_encoder: torch.nn.Module,
    content: torch.Tensor,
    alpha: list[tuple[str, float]],
    *,
    data_root: Path,
    split: str,
    ch: str,
    load_image: Callable[[Path], torch.Tensor],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return content multi-scale features and sum(alpha*Ec(bank c))-Ec(content c)."""
    content_feat, content_res = content_encoder(content)
    content_all = list(content_res) + [content_feat]
    mixed: list[torch.Tensor] | None = None
    for font, weight in alpha:
        path = image_path(data_root, split, "TargetImage", font, ch)
        if not path.exists():
            raise FileNotFoundError(path)
        bank = load_image(path).unsqueeze(0).to(device)
        bank_feat, bank_res = content_encoder(bank)
        bank_all = list(bank_res) + [bank_feat]
        if len(bank_all) != len(content_all):
            raise RuntimeError("content encoder scale count mismatch")
        if mixed is None:
            mixed = [value.float() * weight for value in bank_all]
        else:
            mixed = [acc + value.float() * weight for acc, value in zip(mixed, bank_all)]
    if mixed is None:
        raise RuntimeError("empty alpha mixture")
    delta = [mix.to(base.dtype) - base for mix, base in zip(mixed, content_all)]
    for index, (base, value) in enumerate(zip(content_all, delta)):
        if base.shape != value.shape:
            raise RuntimeError(f"delta shape mismatch at scale {index}: {base.shape} != {value.shape}")
    return content_all, delta
