"""Target-character appearance completion (TC-v2).

The module deliberately has no dependency on Es, alpha, donor features or Delta.
It consumes cached VGG appearance statistics for the reference glyphs and the
707-dimensional pooled Ec content descriptor, and predicts a standardized 896-d
target appearance descriptor.  A separate zero-initialized projection maps that
descriptor to the existing nine global style tokens.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


TC_APPEARANCE_DIM = 2 * (64 + 128 + 256)
TC_EC_DIM = 3 + 64 + 128 + 256 + 256
TC_GLOBAL_TOKENS = 9
TC_GLOBAL_DIM = 1024


def appearance_stats(features: Iterable[torch.Tensor]) -> torch.Tensor:
    """Return FP32 concat(mean, std) for VGG feature maps.

    ``correction=0`` is intentional: cache extraction and online inference use
    the population standard deviation, including for a one-pixel feature map.
    """
    values = []
    for feature in features:
        feature = feature.float()
        if feature.ndim != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(feature.shape)}")
        values.extend((feature.mean(dim=(2, 3)), feature.std(dim=(2, 3), correction=0)))
    out = torch.cat(values, dim=1)
    if out.shape[1] != TC_APPEARANCE_DIM:
        raise ValueError(f"expected {TC_APPEARANCE_DIM} appearance dims, got {out.shape[1]}")
    return out


def pooled_ec_features(features: Iterable[torch.Tensor]) -> torch.Tensor:
    """Pool the five Ec scales into the fixed 707-d query descriptor."""
    values = []
    for feature in features:
        feature = feature.float()
        if feature.ndim != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(feature.shape)}")
        values.append(feature.mean(dim=(2, 3)))
    out = torch.cat(values, dim=1)
    if out.shape[1] != TC_EC_DIM:
        raise ValueError(f"expected {TC_EC_DIM} Ec dims, got {out.shape[1]}")
    return out


class TCV2Cache:
    """Read-only float16 cache of raw/standardized VGG appearance descriptors.

    Keys are ``tc|role|split|font|cp``.  The cache intentionally stores both
    roles in one table so a loader cannot accidentally use a target row as a
    reference row without an explicit role.  ``stats`` is the standardized
    train-fitted descriptor consumed by H.
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("kind") != "tc_v2_appearance_cache":
            raise RuntimeError("invalid TC-v2 cache manifest kind")
        if int(self.manifest.get("appearance_dim", -1)) != TC_APPEARANCE_DIM:
            raise RuntimeError("TC-v2 cache appearance dimension mismatch")
        if self.manifest.get("dtype") != "float16":
            raise RuntimeError("TC-v2 cache dtype must be float16")
        self.keys = [line for line in (self.directory / "keys.txt").read_text(
            encoding="utf-8").splitlines() if line]
        if len(self.keys) != len(set(self.keys)):
            raise RuntimeError("TC-v2 cache contains duplicate keys")
        for key in self.keys:
            parts = key.split("|", 4)
            if len(parts) != 5 or parts[0] != "tc" or parts[1] not in {"ref", "target"} \
                    or parts[2] not in {"train", "val", "test"} or not parts[3] or not parts[4]:
                raise RuntimeError(f"invalid TC-v2 cache key: {key}")
            if parts[1] == "target" and parts[2] == "test":
                raise RuntimeError("TC-v2 cache must not contain test target teachers")
        if int(self.manifest.get("entries", -1)) != len(self.keys):
            raise RuntimeError("TC-v2 cache manifest/key count mismatch")
        expected_bytes = len(self.keys) * TC_APPEARANCE_DIM * np.dtype(np.float16).itemsize
        payload = self.directory / "appearance.dat"
        if not payload.is_file() or payload.stat().st_size != expected_bytes:
            raise RuntimeError("TC-v2 appearance payload size mismatch")
        progress = json.loads((self.directory / "progress.json").read_text(encoding="utf-8"))
        if progress.get("done") != progress.get("total") or progress.get("total") != len(self.keys):
            raise RuntimeError("TC-v2 cache is incomplete")
        stats = self.manifest.get("train_target_stats")
        if not isinstance(stats, dict) or len(stats.get("mean", [])) != TC_APPEARANCE_DIM \
                or len(stats.get("scale", [])) != TC_APPEARANCE_DIM:
            raise RuntimeError("TC-v2 cache has invalid train-only statistics")
        if not np.isfinite(np.asarray(stats["mean"], dtype=np.float32)).all() \
                or not np.isfinite(np.asarray(stats["scale"], dtype=np.float32)).all():
            raise RuntimeError("TC-v2 train statistics contain non-finite values")
        self.index = {key: i for i, key in enumerate(self.keys)}
        self.appearance = np.memmap(payload, dtype=np.float16,
                                    mode="r", shape=(len(self.keys), TC_APPEARANCE_DIM))
        self._stats = None

    @staticmethod
    def key(role: str, split: str, font: str, cp: str) -> str:
        if role not in {"ref", "target"}:
            raise ValueError(f"invalid TC role: {role}")
        return f"tc|{role}|{split}|{font}|{cp}"

    def raw(self, role: str, split: str, font: str, cp: str) -> torch.Tensor:
        key = self.key(role, split, font, cp)
        try:
            row = self.appearance[self.index[key]]
        except KeyError as exc:
            raise KeyError(key) from exc
        value = np.array(row, copy=True)
        if not np.isfinite(value).all():
            raise RuntimeError(f"TC-v2 cache row is non-finite: {key}")
        return torch.from_numpy(value).float()

    def stats(self, role: str, split: str, font: str, cp: str) -> torch.Tensor:
        return self.standardizer.transform(self.raw(role, split, font, cp))

    @property
    def standardizer(self) -> "AppearanceStandardizer":
        if self._stats is None:
            stats = self.manifest.get("train_target_stats")
            if not isinstance(stats, dict):
                raise RuntimeError("TC cache is missing train_target_stats")
            self._stats = AppearanceStandardizer(stats.get("mean"), stats.get("scale"))
        return self._stats

    def validate_split(self, split: str) -> None:
        allowed = {"train", "val", "test"}
        if split not in allowed:
            raise ValueError(f"unknown split {split}; expected one of {sorted(allowed)}")


class AppearanceStandardizer(nn.Module):
    """Train-only per-dimension standardization persisted with the TC cache."""

    def __init__(self, mean: torch.Tensor | None = None, scale: torch.Tensor | None = None,
                 min_scale: float = 1e-4):
        super().__init__()
        if mean is None or scale is None:
            raise ValueError("TC-v2 statistics require both mean and scale")
        mean = torch.as_tensor(mean).float()
        scale = torch.as_tensor(scale).float()
        if mean.numel() != TC_APPEARANCE_DIM or scale.numel() != TC_APPEARANCE_DIM:
            raise ValueError("appearance statistics must have 896 dimensions")
        self.register_buffer("mean", mean.reshape(TC_APPEARANCE_DIM))
        self.register_buffer("scale", scale.reshape(TC_APPEARANCE_DIM).clamp_min(min_scale))

    @classmethod
    def fit(cls, rows: torch.Tensor, min_scale: float = 1e-4) -> "AppearanceStandardizer":
        rows = torch.as_tensor(rows).float()
        if rows.ndim != 2 or rows.shape[1] != TC_APPEARANCE_DIM:
            raise ValueError("fit expects [N,896]")
        return cls(rows.mean(0), rows.std(0, correction=0), min_scale=min_scale)

    def transform(self, rows: torch.Tensor) -> torch.Tensor:
        return (rows.float() - self.mean) / self.scale

    def inverse(self, rows: torch.Tensor) -> torch.Tensor:
        return rows.float() * self.scale + self.mean

    def state_dict_json(self) -> dict:
        return {"mean": self.mean.detach().cpu().tolist(),
                "scale": self.scale.detach().cpu().tolist()}


class TCV2Head(nn.Module):
    """Ref-statistics + Ec query -> target appearance predictor."""

    def __init__(self, appearance_dim: int = TC_APPEARANCE_DIM,
                 ec_dim: int = TC_EC_DIM, hidden_dim: int = 256,
                 out_dim: int = TC_APPEARANCE_DIM, num_heads: int = 8):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.ref_proj = nn.Linear(appearance_dim, hidden_dim)
        self.content_proj = nn.Linear(ec_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads,
                                                 batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, ref_stats: torch.Tensor, content_ec: torch.Tensor,
                ref_mask: torch.Tensor | None = None) -> torch.Tensor:
        if ref_stats.ndim != 3 or ref_stats.shape[-1] != TC_APPEARANCE_DIM:
            raise ValueError("ref_stats must be [B,N,896]")
        if content_ec.ndim != 2 or content_ec.shape[-1] != TC_EC_DIM:
            raise ValueError("content_ec must be [B,707]")
        keys = self.ref_proj(ref_stats)
        query = self.content_proj(content_ec).unsqueeze(1)
        if ref_mask is not None:
            ref_mask = ref_mask.bool()
            if ref_mask.ndim != 2 or ref_mask.shape[:2] != ref_stats.shape[:2]:
                raise ValueError("ref_mask must be [B,N]")
            if not bool(ref_mask.any(dim=1).all()):
                raise ValueError("TC-v2 requires at least one valid reference per sample")
        key_padding = None if ref_mask is None else ~ref_mask
        attended, _ = self.cross_attn(query, keys, keys, key_padding_mask=key_padding)
        attended = self.norm(attended + query).squeeze(1)
        return self.mlp(torch.cat([attended, query.squeeze(1)], dim=1))


class TCV2Global9Adapter(nn.Module):
    """Zero-init residual projection for the existing nine global tokens."""

    def __init__(self, appearance_dim: int = TC_APPEARANCE_DIM,
                 token_dim: int = TC_GLOBAL_DIM, n_tokens: int = TC_GLOBAL_TOKENS):
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.proj = nn.Linear(appearance_dim, token_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, appearance: torch.Tensor) -> torch.Tensor:
        if appearance.ndim != 2 or appearance.shape[-1] != TC_APPEARANCE_DIM:
            raise ValueError("appearance must be [B,896]")
        return self.proj(appearance).unsqueeze(1).expand(-1, self.n_tokens, -1)


def cache_fingerprint(manifest: dict) -> str:
    """Stable manifest fingerprint used to bind train/val/test cache consumers."""
    payload = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_standardizer(path: Path, standardizer: AppearanceStandardizer, manifest: dict) -> None:
    payload = {"kind": "tc_v2_stats", "stats": standardizer.state_dict_json(),
               "cache_manifest_sha256": cache_fingerprint(manifest)}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def load_standardizer(path: Path, manifest: dict | None = None) -> AppearanceStandardizer:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") != "tc_v2_stats":
        raise RuntimeError(f"not a TC-v2 statistics file: {path}")
    if manifest is not None and payload.get("cache_manifest_sha256") != cache_fingerprint(manifest):
        raise RuntimeError("TC-v2 statistics/cache manifest binding mismatch")
    stats = payload.get("stats") or {}
    return AppearanceStandardizer(stats.get("mean"), stats.get("scale"))
