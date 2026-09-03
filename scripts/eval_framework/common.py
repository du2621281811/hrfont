#!/usr/bin/env python3
"""配置、复现性与轻量日志工具。仅依赖 PyYAML/NumPy/PyTorch。"""
from __future__ import annotations

import hashlib, json, os, random, sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
import yaml


class FrozenConfig(Mapping):
    def __init__(self, values: dict[str, Any]):
        object.__setattr__(self, "_data", MappingProxyType({k: freeze(v) for k, v in values.items()}))
    def __getitem__(self, key): return self._data[key]
    def __iter__(self): return iter(self._data)
    def __len__(self): return len(self._data)
    def __getattr__(self, key):
        try: return self._data[key]
        except KeyError as exc: raise AttributeError(key) from exc
    def __setattr__(self, *_): raise TypeError("FrozenConfig is immutable")


def freeze(value: Any) -> Any:
    if isinstance(value, dict): return FrozenConfig(value)
    if isinstance(value, list): return tuple(freeze(x) for x in value)
    return value


def thaw(value: Any) -> Any:
    if isinstance(value, FrozenConfig): return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, tuple): return [thaw(x) for x in value]
    return value


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        out[key] = deep_merge(out[key], value) if key in out and isinstance(out[key], dict) and isinstance(value, dict) else value
    return out


def _set_dot(cfg: dict, item: str) -> None:
    if "=" not in item: raise ValueError(f"--set requires key=value: {item}")
    dotted, raw = item.split("=", 1); parts = dotted.split("."); node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict): raise KeyError(f"unknown config key: {dotted}")
        node = node[part]
    if parts[-1] not in node: raise KeyError(f"unknown config key: {dotted}")
    value = yaml.safe_load(raw)
    old = node[parts[-1]]
    if old is not None and not isinstance(value, type(old)) and not (isinstance(old, float) and isinstance(value, (int, float))):
        raise TypeError(f"type mismatch for {dotted}: expected {type(old).__name__}")
    node[parts[-1]] = value


def canonical_json(cfg: Any) -> str:
    return json.dumps(thaw(cfg), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_sha256(cfg: Any) -> str:
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()


def load_config(base: str | Path, override: str | Path | None = None, sets: list[str] | None = None) -> FrozenConfig:
    def read(path):
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict): raise ValueError(f"config must be mapping: {path}")
        return value
    cfg = read(base)
    if override: cfg = deep_merge(cfg, read(override))
    for item in sets or []: _set_dot(cfg, item)
    if any(x is None or (isinstance(x, str) and x.upper() == "TBD") for x in _walk(cfg)):
        raise ValueError("config contains null/TBD")
    return freeze(cfg)


def _walk(x):
    if isinstance(x, dict):
        for v in x.values(): yield from _walk(v)
    elif isinstance(x, list):
        for v in x: yield from _walk(v)
    else: yield x


def parse_cli(argv=None) -> dict[str, Any]:
    """无 argparse 冲突的小型解析器：--config/--override/--set/--key value。"""
    args = list(sys.argv[1:] if argv is None else argv); out = {"set": []}; i = 0
    while i < len(args):
        token = args[i]
        if not token.startswith("--"): raise ValueError(f"unexpected argument: {token}")
        key = token[2:].replace("-", "_")
        if key in {"help", "build_cache", "smoke"}: out[key] = True; i += 1; continue
        if i + 1 >= len(args): raise ValueError(f"missing value: {token}")
        if key == "set": out["set"].append(args[i + 1])
        else: out[key] = args[i + 1]
        i += 2
    return out


def seed_all(seed: int, deterministic: bool = True) -> torch.Generator:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    return make_generator(seed)


def make_generator(seed: int, epoch: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed) + 1000 * int(epoch))


def worker_init_fn(seed: int, epoch: int = 0):
    def init(worker_id: int):
        value = int(seed) + 1000 * int(epoch) + worker_id
        random.seed(value); np.random.seed(value % (2**32)); torch.manual_seed(value)
    return init


class JSONLLogger:
    def __init__(self, path): self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
    def log(self, **record):
        with self.path.open("a", encoding="utf-8") as f: f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def save_resolved_config(cfg, out_dir: str | Path, input_path: str | Path | None = None) -> str:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); sha = config_sha256(cfg)
    if input_path: (out / "config.input.yaml").write_text(Path(input_path).read_text(encoding="utf-8"), encoding="utf-8")
    (out / "config.resolved.yaml").write_text(yaml.safe_dump(thaw(cfg), allow_unicode=True, sort_keys=True), encoding="utf-8")
    (out / "config.canonical.json").write_text(canonical_json(cfg) + "\n", encoding="utf-8")
    (out / "config.sha256").write_text(sha + "\n", encoding="utf-8")
    return sha


def device_from_config(cfg):
    wanted = str(getattr(cfg.train, "device", "auto"))
    if wanted != "auto": return torch.device(wanted)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
