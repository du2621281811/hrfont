#!/usr/bin/env python3
"""Build the small, train-only v0921 Es/Ec caches used by K7-B.

The frozen V2 bank cache is never opened here.  Only v0921 style references,
v0921 targets, and canonical V2 content rows are encoded.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from scripts.hrfont_feature_cache import EC_SHAPES, ES_POOLED, ES_SPATIAL, MemmapTable, key_ec, key_es, sha256_file
from scripts.hrfont_build_e1_caches import encode_ec_batch, encode_es_batch, load_encoder


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def jobs_for(kind: str, data_root: Path, spec: dict):
    fonts = spec["fonts"]
    pools = load_json(Path(spec["style_pool"]))
    pairs = [line.split("\t") for line in Path(spec["pairs"]).read_text(encoding="utf-8").splitlines()[1:] if line.strip()]
    if kind == "es":
        return [(key_es("train", f, cp), data_root / "train" / "StyleImage" / f / f"{f}+{cp}.png")
                for f in fonts for cp in pools[f]]
    out = []
    cps = sorted({cp for _, _, cp in pairs})
    for cp in cps:
        out.append((key_ec("content", "", cp), Path(spec["data_root"]) / "train" / "ContentImage" / f"{cp}.png"))
    for _, f, cp in pairs:
        out.append((key_ec("target", f, cp), data_root / "train" / "TargetImage" / f / f"{f}+{cp}.png"))
    return out


def build(kind: str, jobs, out: Path, ckpt: Path, device: str, batch: int, spec: dict):
    out.mkdir(parents=True, exist_ok=True)
    keys = [k for k, _ in jobs]
    table = MemmapTable(out)
    if (out / "keys.txt").exists():
        if table.load_keys() != keys:
            shutil.rmtree(out)
            out.mkdir(parents=True, exist_ok=True)
            table = MemmapTable(out)
            table.set_keys(keys)
    else:
        table.set_keys(keys)
    n = len(jobs)
    if kind == "es":
        arrays = [table.open_array("spatial", (n, *ES_SPATIAL), "w+"), table.open_array("pooled", (n, *ES_POOLED), "w+")]
    else:
        arrays = [table.open_array(f"s{i}", (n, *shape), "w+") for i, shape in enumerate(EC_SHAPES)]
    model, encoder_sha = load_encoder(kind, ckpt, device)
    for start in range(0, n, batch):
        chunk = jobs[start:start + batch]
        paths = [p for _, p in chunk]
        if kind == "es":
            spatial, pooled = encode_es_batch(model, paths, device)
            arrays[0][start:start + len(chunk)] = spatial
            arrays[1][start:start + len(chunk)] = pooled
        else:
            values = encode_ec_batch(model, paths, device)
            for array, value in zip(arrays, values):
                array[start:start + len(chunk)] = value
        if (start + len(chunk)) % 512 == 0 or start + len(chunk) == n:
            for array in arrays: array.flush()
            print(f"{kind} {start + len(chunk)}/{n}", flush=True)
    for array in arrays: array.flush()
    manifest = {
        "kind": kind, "entries": n, "dtype": "float16", "encoder_sha256": encoder_sha,
        "es_checkpoint_sha256": encoder_sha if kind == "es" else None,
        "ec_checkpoint_sha256": encoder_sha if kind == "ec" else None,
        "bank_inclusion": False, "dataset_id": "v0921", "role": "train_only_nonbank",
        "source_spec_sha256": sha256_file(Path(spec["spec_path"])),
        "keys_sha256": sha256_file(out / "keys.txt"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "COMPLETE.json").write_text(json.dumps({"status": "COMPLETE", "identity": manifest}, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--es-out", type=Path, required=True)
    ap.add_argument("--ec-out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gpu", type=int)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    spec = load_json(args.spec); spec["spec_path"] = str(args.spec.resolve())
    root = Path(spec["data_root"])
    build("es", jobs_for("es", root, spec), args.es_out, args.ckpt, args.device, args.batch, spec)
    build("ec", jobs_for("ec", root, spec), args.ec_out, args.ckpt, args.device, args.batch, spec)


if __name__ == "__main__":
    main()
