#!/usr/bin/env python3
"""Memmap Es/Ec feature caches for cache-only Stage-A training."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

EC_SHAPES = (
    (3, 96, 96),
    (64, 48, 48),
    (128, 24, 24),
    (256, 12, 12),
    (256, 12, 12),
)
ES_SPATIAL = (1024, 3, 3)
ES_POOLED = (1024,)


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_es(split: str, font: str, cp: str) -> str:
    return f"es|{split}|{font}|{cp}"


def key_ec(role: str, font: str, cp: str) -> str:
    return f"ec|{role}|{font}|{cp}"


class MemmapTable:
    def __init__(self, directory: Path, create: bool = False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keys_path = self.directory / "keys.txt"
        self.manifest_path = self.directory / "manifest.json"
        self.progress_path = self.directory / "progress.json"
        self.keys: list[str] = []
        self.index: dict[str, int] = {}
        self.arrays: dict[str, np.memmap] = {}

    def set_keys(self, keys: list[str]) -> None:
        self.keys = list(keys)
        self.index = {key: i for i, key in enumerate(self.keys)}
        self.keys_path.write_text("\n".join(self.keys) + "\n", encoding="utf-8")

    def load_keys(self) -> list[str]:
        if not self.keys_path.exists():
            raise FileNotFoundError(self.keys_path)
        self.keys = [line for line in self.keys_path.read_text(encoding="utf-8").splitlines() if line]
        self.index = {key: i for i, key in enumerate(self.keys)}
        return self.keys

    def open_array(self, name: str, shape: tuple[int, ...], mode: str) -> np.memmap:
        path = self.directory / f"{name}.dat"
        array = np.memmap(path, dtype=np.float16, mode=mode, shape=shape)
        self.arrays[name] = array
        return array

    def save_manifest(self, payload: dict) -> None:
        self.manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def load_manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def save_progress(self, done: int, total: int) -> None:
        self.progress_path.write_text(json.dumps({"done": done, "total": total}) + "\n", encoding="utf-8")

    def load_progress(self) -> int:
        if not self.progress_path.exists():
            return 0
        return int(json.loads(self.progress_path.read_text(encoding="utf-8")).get("done", 0))

    def flush(self) -> None:
        for array in self.arrays.values():
            array.flush()


class EsCache:
    def __init__(self, directory: Path):
        self.table = MemmapTable(directory)
        self.table.load_keys()
        n = len(self.table.keys)
        self.spatial = self.table.open_array("spatial", (n, *ES_SPATIAL), "r")
        self.pooled = self.table.open_array("pooled", (n, *ES_POOLED), "r")
        self.manifest = self.table.load_manifest()

    def _row(self, split: str, font: str, cp: str) -> int:
        key = key_es(split, font, cp)
        if key not in self.table.index:
            raise KeyError(key)
        return self.table.index[key]

    def spatial_tensor(self, split: str, font: str, cp: str) -> torch.Tensor:
        return torch.from_numpy(np.array(self.spatial[self._row(split, font, cp)], copy=True)).float()

    def pooled_tensor(self, split: str, font: str, cp: str) -> torch.Tensor:
        return torch.from_numpy(np.array(self.pooled[self._row(split, font, cp)], copy=True)).float()


class EcCache:
    def __init__(self, directory: Path):
        self.table = MemmapTable(directory)
        self.table.load_keys()
        n = len(self.table.keys)
        self.scales = [
            self.table.open_array(f"s{i}", (n, *shape), "r")
            for i, shape in enumerate(EC_SHAPES)
        ]
        self.manifest = self.table.load_manifest()

    def features(self, role: str, font: str, cp: str) -> list[torch.Tensor]:
        key = key_ec(role, font, cp)
        if key not in self.table.index:
            raise KeyError(key)
        row = self.table.index[key]
        return [torch.from_numpy(np.array(scale[row], copy=True)).float().unsqueeze(0) for scale in self.scales]


def mix_cached_delta(neighbor_feats: list[list[torch.Tensor]], weights: torch.Tensor,
                     neutral_feats: list[torch.Tensor]) -> list[torch.Tensor]:
    if not neighbor_feats or weights.numel() == 0:
        raise RuntimeError("top-K alpha produced an empty neighborhood")
    mixed = []
    for scale, base in enumerate(neutral_feats):
        acc = torch.zeros_like(base, dtype=torch.float32)
        for feats, weight in zip(neighbor_feats, weights):
            acc.add_(feats[scale].float(), alpha=float(weight))
        mixed.append((acc - base.float()).to(base.dtype))
    return mixed
