#!/usr/bin/env python3
"""E2 feature-space font delta utilities.

The real encoders are deliberately injected.  FontDiffuser's contracts are
``Es(x) -> (style_map, pooled_style, residuals)`` and
``Ec(x) -> (final_content, residuals)``.  This module represents Ec features
in the RSI order ``list(residuals) + [final_content]``.

Training style images remain sampled from the full 338-character style pool;
only the deterministic ref8 set is used to construct retrieval prototypes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

REF8_DEFAULT = "永和书风骨韵天地"


@dataclass(frozen=True)
class DeltaConfig:
    tau: float = 0.07
    mode: str = "soft"
    eps_alpha: float = 0.01
    k_max: int = 10
    k_top: int = 3
    rng_seed: int = 3407

    def __post_init__(self) -> None:
        if self.mode not in {"soft", "topk", "threshold"}:
            raise ValueError(f"unknown alpha mode: {self.mode}")
        if self.tau <= 0 or self.eps_alpha < 0 or self.k_max < 0 or self.k_top < 0:
            raise ValueError("invalid DeltaConfig")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "DeltaConfig":
        return cls(**value)

    def summary(self) -> str:
        return (f"alpha mode={self.mode} tau={self.tau:g} eps_alpha={self.eps_alpha:g} "
                f"K_max={self.k_max} K_top={self.k_top} seed={self.rng_seed}")


def cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarities, shape ``[len(a), len(b)]``."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"cosine inputs must be [M,D]/[N,D], got {a.shape}/{b.shape}")
    return F.normalize(a.float(), dim=1) @ F.normalize(b.float(), dim=1).T


def truncate_renorm(weights: torch.Tensor, eps: float, k_max: int) -> tuple[list[int], torch.Tensor]:
    """Keep entries strictly above eps, cap by weight, and renormalize."""
    if weights.ndim != 1:
        raise ValueError("weights must be 1-D")
    keep = torch.nonzero(weights > eps, as_tuple=False).flatten()
    if k_max == 0 or keep.numel() == 0:
        return [], weights.new_empty((0,))
    order = sorted(keep.tolist(), key=lambda i: (-float(weights[i].detach()), i))[:k_max]
    chosen = weights[order]
    total = chosen.sum()
    if not torch.isfinite(total) or total <= 0:
        return [], weights.new_empty((0,))
    return order, chosen / total


@torch.no_grad()
def compute_alpha(
    ref8_feats: torch.Tensor,
    proto_feats: torch.Tensor,
    exclude_idx: int | None,
    cfg: DeltaConfig,
) -> tuple[list[int], torch.Tensor, dict]:
    """Compute a sparse, normalized neighborhood over candidate prototypes.

    ``ref8_feats`` are the target font's n normalized Es vectors ([n, D], one
    per ref char); ``proto_feats`` are the per-char normalized Es vectors of
    each candidate font ([N, n, D], same chars).  Aggregation = per-char
    cosine averaged over the n chars (collaborator-agreed; matches the design
    doc formula), then soft/topk/threshold neighborhood selection.  Exclusion
    is applied before every weighting mode.
    """
    if ref8_feats.ndim != 2 or proto_feats.ndim != 3:
        raise ValueError("ref8_feats must be [n,D] and proto_feats [N,n,D]")
    if proto_feats.shape[1] != ref8_feats.shape[0]:
        raise ValueError("char count mismatch between refs and prototypes")
    q = F.normalize(ref8_feats.float(), dim=1)
    p = F.normalize(proto_feats.float(), dim=2)
    cos = torch.einsum("nd,fnd->fn", q, p).mean(dim=1)
    valid = torch.ones(len(cos), dtype=torch.bool, device=cos.device)
    if exclude_idx is not None:
        if not 0 <= exclude_idx < len(cos):
            raise IndexError(exclude_idx)
        valid[exclude_idx] = False
    valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
    valid_cos = cos[valid]
    if valid_cos.numel():
        pre_soft = torch.softmax(valid_cos / cfg.tau, dim=0)
        entropy = float((-(pre_soft * pre_soft.clamp_min(1e-12).log()).sum()).detach())
        max_cos = float(valid_cos.max().detach())
    else:
        pre_soft = valid_cos
        entropy, max_cos = 0.0, float("nan")

    if cfg.mode == "soft":
        local, selected = truncate_renorm(pre_soft, cfg.eps_alpha, cfg.k_max)
    elif cfg.mode == "topk":
        k = min(cfg.k_top, len(valid_cos))
        if k:
            order = sorted(range(len(valid_cos)), key=lambda i: (-float(valid_cos[i]), i))[:k]
            selected = torch.softmax(valid_cos[order] / cfg.tau, dim=0)
            local = order
        else:
            local, selected = [], valid_cos.new_empty((0,))
    else:
        raw = (valid_cos - cfg.eps_alpha).clamp_min(0)
        local, selected = truncate_renorm(raw, 0.0, cfg.k_max)

    indices = [int(valid_idx[i]) for i in local]
    meta = {
        "mode": cfg.mode, "tau": cfg.tau, "eps_alpha": cfg.eps_alpha,
        "k_max": cfg.k_max, "n_active": len(indices),
        "entropy": entropy, "max_cosine": max_cos,
    }
    return indices, selected, meta


def _ec_features(output) -> list[torch.Tensor]:
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise TypeError("Ec must return (final_feature, residual_feature_list)")
    final, residuals = output[0], output[1]
    if not isinstance(residuals, (tuple, list)):
        raise TypeError("Ec residual features must be a list")
    return list(residuals) + [final]


def load_rgb_tensor(path: str | Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.size != (96, 96):
        raise ValueError(f"expected native 96x96 image: {path} ({image.size})")
    x = torch.from_numpy(__import__("numpy").array(image, copy=True)).permute(2, 0, 1).float() / 127.5 - 1
    return x.unsqueeze(0)


@torch.no_grad()
def feature_delta(
    ec: nn.Module,
    images_neighbor: Sequence[torch.Tensor | str | Path],
    weights: torch.Tensor,
    image_neutral: torch.Tensor,
    ec_kwargs: dict | None = None,
) -> list[torch.Tensor] | None:
    """Return multi-scale Δ in FontDiffuser RSI order: ``residuals + [h]``.

    Each neighbor is encoded separately before its feature is weighted. Paths
    are loaded as native RGB 96x96 tensors normalized to [-1, 1].  Empty
    weights return ``None`` and mean that the caller must disable Δ.
    """
    if weights.numel() == 0:
        return None
    if len(images_neighbor) != len(weights):
        raise ValueError("neighbor/weight count mismatch")
    kwargs = ec_kwargs or {}
    device = next(ec.parameters(), image_neutral).device
    neutral = image_neutral.to(device)
    base = _ec_features(ec(neutral, **kwargs))
    mixed = [torch.zeros_like(x, dtype=torch.float32) for x in base]
    for image, weight in zip(images_neighbor, weights):
        x = load_rgb_tensor(image) if isinstance(image, (str, Path)) else image
        feats = _ec_features(ec(x.to(device), **kwargs))
        if len(feats) != len(base):
            raise ValueError("Ec scale count mismatch")
        for scale, value in enumerate(feats):
            if value.shape != base[scale].shape:
                raise ValueError(f"Ec scale {scale} shape mismatch")
            mixed[scale].add_(value.float(), alpha=float(weight))
    return [(value - ref.float()).to(ref.dtype) for value, ref in zip(mixed, base)]


def _style_vector(output) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[1] if len(output) >= 2 else output[0]
    return output.flatten(1)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_style_image(root: Path, stem: str, ch: str) -> Path:
    cp = f"u{ord(ch):04X}"
    candidates = sorted(root.glob(f"**/StyleImage/{stem}/{stem}+{cp}.png"))
    if not candidates:
        candidates = sorted(root.glob(f"**/{stem}/{stem}+{cp}.png"))
    if not candidates:
        raise FileNotFoundError(f"style render missing: {stem} {ch} ({cp}) under {root}")
    return candidates[0]


@torch.no_grad()
def build_style_prototypes(
    es: nn.Module,
    font_stems: Iterable[str],
    render_fn_or_dir: Callable[[str, str], torch.Tensor | str | Path] | str | Path,
    ref8: str,
    device: str | torch.device,
    workers: int = 0,
    es_ckpt_sha: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Build deterministic ref8 Es prototypes; ``workers`` is reserved/ignored."""
    del workers
    es = es.to(device).eval()
    prototypes, records = {}, []
    for stem in sorted(font_stems):
        vectors, hashes = [], []
        for ch in ref8:
            source = (render_fn_or_dir(stem, ch) if callable(render_fn_or_dir)
                      else find_style_image(Path(render_fn_or_dir), stem, ch))
            if isinstance(source, (str, Path)):
                hashes.append({"char": ch, "sha256": sha256_file(source)})
                image = load_rgb_tensor(source)
            else:
                image = source if source.ndim == 4 else source.unsqueeze(0)
                hashes.append({"char": ch, "sha256": None})
            vector = _style_vector(es(image.to(device))).float()
            vectors.append(F.normalize(vector, dim=1).squeeze(0).cpu())
        proto = torch.stack(vectors)  # [n, D] per-char, L2-normalized
        prototypes[stem] = proto
        records.append({"stem": stem, "prototype_per_char": proto.tolist(), "n_chars": len(ref8), "ref8_pngs": hashes})
    manifest = {"font_stems": sorted(prototypes), "ref8": ref8,
                "es_ckpt_sha256": es_ckpt_sha, "fonts": records}
    return prototypes, manifest


class DummyEs(nn.Module):
    def __init__(self, dim: int = 12):
        super().__init__()
        g = torch.Generator().manual_seed(3407)
        self.conv = nn.Conv2d(3, dim, 5, stride=2, padding=2, bias=False)
        self.conv.weight.data.copy_(torch.randn(self.conv.weight.shape, generator=g) * .08)

    def forward(self, x):
        style = torch.tanh(self.conv(x))
        pooled = F.adaptive_avg_pool2d(style, 1).flatten(1)
        return style, pooled, [x, style]


class DummyEc(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(3408)
        self.c1 = nn.Conv2d(3, 5, 3, padding=1)
        self.c2 = nn.Conv2d(5, 7, 3, stride=2, padding=1)

    def forward(self, x):
        r = torch.tanh(self.c1(x))
        h = torch.tanh(self.c2(r))
        return h, [x, r]


def _font_paths() -> list[Path]:
    roots = [Path("/System/Library/Fonts"), Path("/Library/Fonts")]
    names = ("PingFang", "STHeiti", "Songti", "Hiragino")
    found = []
    for root in roots:
        if root.exists():
            found.extend(p for p in root.rglob("*") if p.suffix.lower() in {".ttf", ".ttc", ".otf"}
                         and any(n.lower() in p.name.lower() for n in names))
    return sorted(set(found))


def render_glyph(ch: str, font_path: Path | None, shade: int = 0) -> torch.Tensor:
    image = Image.new("RGB", (96, 96), "white")
    font = ImageFont.truetype(str(font_path), 68) if font_path else ImageFont.load_default()
    draw = ImageDraw.Draw(image)
    box = draw.textbbox((0, 0), ch, font=font)
    xy = ((96 - (box[2] - box[0])) / 2 - box[0], (96 - (box[3] - box[1])) / 2 - box[1])
    draw.text(xy, ch, font=font, fill=(shade, shade, shade))
    import numpy as np
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1


def run_smoke() -> None:
    torch.manual_seed(3407)
    es, ec = DummyEs().eval(), DummyEc().eval()
    fonts = _font_paths()
    chosen = (fonts[:4] if len(fonts) >= 4 else fonts + [None] * (4 - len(fonts)))
    ref = REF8_DEFAULT
    feats = []
    for font in chosen:
        rows = [F.normalize(_style_vector(es(render_glyph(ch, font))), dim=1).squeeze(0) for ch in ref]
        feats.append(torch.stack(rows))
    protos = torch.stack(feats)  # [4, n, D] per-char
    # Six candidates are required by the smoke contract.
    protos = torch.cat([protos, F.normalize(torch.randn(2, protos.shape[1], protos.shape[2]), dim=2)])
    idx, weights, _ = compute_alpha(feats[0], protos, 0, DeltaConfig(eps_alpha=0, k_max=10))
    assert 0 not in idx and len(idx) == 5 and torch.allclose(weights.sum(), torch.tensor(1.0))
    isolated = torch.tensor([.97, .005, .005, .005, .005, .01])
    keep, w = truncate_renorm(isolated, .01, 10)
    assert keep == [0] and torch.allclose(w, torch.ones(1))
    flat = torch.ones(6) / 6
    keep, _ = truncate_renorm(flat, .01, 3)
    assert keep == [0, 1, 2]
    images = [render_glyph("永", chosen[i], i * 25) for i in range(3)]
    w3 = torch.tensor([.2, .3, .5])
    delta = feature_delta(ec, images, w3, render_glyph("永", chosen[0]))
    assert delta is not None and [tuple(x.shape) for x in delta] == [(1, 3, 96, 96), (1, 5, 96, 96), (1, 7, 48, 48)]
    pixel_mix = sum(float(w) * x for w, x in zip(w3, images))
    pixel_delta = [a - b for a, b in zip(_ec_features(ec(pixel_mix)), _ec_features(ec(images[0])))]
    assert any(not torch.allclose(a, b, atol=1e-6) for a, b in zip(delta, pixel_delta))
    assert feature_delta(ec, [], torch.empty(0), images[0]) is None
    print("hrfont_delta_v2 smoke: PASS")


def _load_cli_es(mode: str, path: str | None) -> nn.Module:
    if mode == "dummy":
        return DummyEs()
    if mode == "torchscript":
        if not path:
            raise SystemExit("--encoder-path is required for torchscript")
        return torch.jit.load(path, map_location="cpu")
    raise SystemExit("variant CLI loading is intentionally not implicit; import this module and pass the built Es to build_style_prototypes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")
    proto = sub.add_parser("proto")
    proto.add_argument("--data-root", required=True)
    proto.add_argument("--split-manifest", required=True)
    proto.add_argument("--output", required=True)
    proto.add_argument("--ref8", default=REF8_DEFAULT)
    proto.add_argument("--encoder", choices=("dummy", "torchscript", "variant"), default="dummy")
    proto.add_argument("--encoder-path")
    proto.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "smoke":
        run_smoke()
        return
    split = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    stems = split.get("stems", split)["train"]
    es = _load_cli_es(args.encoder, args.encoder_path)
    ckpt_sha = sha256_file(args.encoder_path) if args.encoder_path else None
    _, manifest = build_style_prototypes(es, stems, args.data_root, args.ref8, args.device,
                                         es_ckpt_sha=ckpt_sha)
    Path(args.output).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(DeltaConfig().summary())
    print(f"wrote {args.output}: {len(stems)} prototypes")


if __name__ == "__main__":
    main()
