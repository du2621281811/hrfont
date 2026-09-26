#!/usr/bin/env python3
"""Contract and provenance guard for v0921 train-only, non-bank data."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from scripts.hrfont_feature_cache import EcCache, EsCache, key_ec, key_es

SCHEMA_VERSION = 1
ROLE = "train_only_nonbank"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


@dataclass(frozen=True)
class V0921Spec:
    spec_path: Path
    data_root: Path
    pairs_path: Path
    style_pool_path: Path
    es_cache_path: Path
    ec_cache_path: Path
    fonts: tuple[str, ...]
    split: str
    pairs: tuple[dict[str, str], ...]
    style_pool: dict[str, tuple[str, ...]]
    payload_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "V0921Spec":
        spec_path = Path(path).resolve()
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"v0921 schema_version must be {SCHEMA_VERSION}: {spec_path}")
        if payload.get("dataset_id") != "v0921":
            raise RuntimeError("v0921 spec dataset_id must be 'v0921'")
        if payload.get("role") != ROLE or payload.get("bank_inclusion") is not False:
            raise RuntimeError("v0921 must declare role=train_only_nonbank and bank_inclusion=false")
        if payload.get("split") != "train":
            raise RuntimeError("v0921 is training-only; split must be 'train'")
        base = spec_path.parent
        data_root = _resolve(base, payload["data_root"])
        pairs_path = _resolve(base, payload["pairs"])
        style_pool_path = _resolve(base, payload["style_pool"])
        es_cache_path = _resolve(base, payload["es_cache"])
        ec_cache_path = _resolve(base, payload["ec_cache"])
        paths = {"data_root": data_root, "pairs": pairs_path, "style_pool": style_pool_path,
                 "es_cache": es_cache_path, "ec_cache": ec_cache_path}
        missing = [f"{name}={path}" for name, path in paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("v0921 spec is not ready: " + "; ".join(missing))
        fonts = tuple(sorted(set(payload.get("fonts", []))))
        if not fonts:
            raise RuntimeError("v0921 spec fonts must be non-empty")
        if len(fonts) != len(payload.get("fonts", [])):
            raise RuntimeError("v0921 spec fonts contain duplicates")
        if any("|" in font or not font.strip() for font in fonts):
            raise RuntimeError("v0921 font stems contain an invalid separator/empty name")
        pairs = tuple(_load_tsv(pairs_path))
        required = {"split", "font", "cp"}
        if not pairs or not required.issubset(pairs[0]):
            raise RuntimeError(f"v0921 pairs must contain {sorted(required)}: {pairs_path}")
        if any(row["split"] != "train" for row in pairs):
            raise RuntimeError("v0921 pairs may contain only split=train rows")
        if set(row["font"] for row in pairs) - set(fonts):
            raise RuntimeError("v0921 pairs contain fonts absent from spec")
        pool_payload = json.loads(style_pool_path.read_text(encoding="utf-8"))
        style_pool = {font: tuple(sorted(pool_payload.get(font, []))) for font in fonts}
        if any(len(style_pool[font]) < 8 for font in fonts):
            raise RuntimeError("v0921 style_pool must contain at least 8 refs for every font")
        if set(pool_payload) != set(fonts):
            raise RuntimeError("v0921 style_pool keys must exactly match spec fonts")
        if any(row["cp"] in style_pool[row["font"]] for row in pairs):
            raise RuntimeError("v0921 target cp cannot also be a reference cp")
        return cls(spec_path, data_root, pairs_path, style_pool_path, es_cache_path,
                   ec_cache_path, fonts, "train", pairs, style_pool, sha256_file(spec_path))

    def validate_runtime(self, bank_fonts: set[str], base_es_manifest: dict | None = None,
                         base_ec_manifest: dict | None = None, base_data_root: Path | None = None) -> None:
        overlap = set(self.fonts) & set(bank_fonts)
        if overlap:
            raise RuntimeError(f"v0921 bank contamination: extra fonts already in donor bank: {sorted(overlap)}")
        for font in self.fonts:
            for cp in self.style_pool[font]:
                path = self.data_root / "train" / "StyleImage" / font / f"{font}+{cp}.png"
                if not path.is_file():
                    raise FileNotFoundError(path)
        for row in self.pairs:
            font, cp = row["font"], row["cp"]
            path = self.data_root / "train" / "TargetImage" / font / f"{font}+{cp}.png"
            content = self.data_root / "train" / "ContentImage" / f"{cp}.png"
            if not path.is_file() or not content.is_file():
                raise FileNotFoundError(path if not path.is_file() else content)
            if base_data_root is not None:
                base_content = Path(base_data_root) / "train" / "ContentImage" / f"{cp}.png"
                if not base_content.is_file():
                    raise FileNotFoundError(base_content)
                if sha256_file(content) != sha256_file(base_content):
                    raise RuntimeError(f"v0921 content image differs from frozen V2 content: {cp}")
        es = EsCache(self.es_cache_path)
        ec = EcCache(self.ec_cache_path)
        for font in self.fonts:
            for cp in self.style_pool[font]:
                if key_es("train", font, cp) not in es.table.index:
                    raise RuntimeError(f"v0921 Es cache missing style key: {font} {cp}")
        for row in self.pairs:
            font, cp = row["font"], row["cp"]
            if key_ec("target", font, cp) not in ec.table.index:
                raise RuntimeError(f"v0921 Ec cache missing target key: {font} {cp}")
            if key_ec("content", "", cp) not in ec.table.index:
                raise RuntimeError(f"v0921 Ec cache missing content key: {cp}")
        for cache_name, cache in (("Es", es), ("Ec", ec)):
            if cache.manifest.get("bank_inclusion") is not False:
                raise RuntimeError(f"v0921 {cache_name} cache must declare bank_inclusion=false")
        for label, base, extra in (("Es", base_es_manifest, es.manifest),
                                   ("Ec", base_ec_manifest, ec.manifest)):
            if base is None:
                continue
            for field in ("encoder_sha256", "es_checkpoint_sha256", "ec_checkpoint_sha256"):
                if field in base and field in extra and base[field] != extra[field]:
                    raise RuntimeError(f"v0921 {label} cache {field} does not match active frozen encoder")

    def assert_target_chars(self, allowed: set[str]) -> None:
        missing = sorted({row["cp"] for row in self.pairs} - set(allowed))
        if missing:
            raise RuntimeError("v0921 target codepoints are absent from the frozen detail manifest: " + ", ".join(missing))

    def provenance(self) -> dict:
        return {"dataset_id": "v0921", "role": ROLE, "bank_inclusion": False,
                "spec_path": str(self.spec_path), "spec_sha256": self.payload_sha256,
                "data_root": str(self.data_root), "pairs_sha256": sha256_file(self.pairs_path),
                "style_pool_sha256": sha256_file(self.style_pool_path),
                "es_cache": str(self.es_cache_path), "ec_cache": str(self.ec_cache_path),
                "fonts": list(self.fonts), "pairs": len(self.pairs)}


class NonBankFamilyPolicy:
    def __init__(self, current, extra_fonts: set[str]):
        self.current = current
        self.extra_fonts = set(extra_fonts)

    def __getattr__(self, name):
        return getattr(self.current, name)

    def exclude(self, font, valid):
        if font in self.extra_fonts:
            return valid.clone()
        return self.current.exclude(font, valid)

    def verify(self, font, selected):
        if font in self.extra_fonts:
            return
        return self.current.verify(font, selected)

    def snapshot(self):
        result = self.current.snapshot()
        result["nonbank_extra_fonts"] = sorted(self.extra_fonts)
        return result
