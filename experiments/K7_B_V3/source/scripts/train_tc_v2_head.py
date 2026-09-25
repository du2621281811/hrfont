#!/usr/bin/env python3
"""Pretrain only the TC-v2 H head from frozen descriptor caches.

This is deliberately separate from diffusion ``--resume_from``.  Its output
contains ``tc_head.pth`` and a small manifest, and can be supplied to the joint
trainer with ``--tc_head_ckpt``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
VARIANT = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(VARIANT))
from hrfont_feature_cache import EcCache  # noqa: E402
from src.tc_v2 import (TCV2Cache, TCV2Head, cache_fingerprint,
                       pooled_ec_features)  # noqa: E402


def clean_map_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("pairs_train.tsv", "pairs_val.tsv", "pairs_test.tsv", "sample_weights.json"):
        file = path / name
        if not file.is_file():
            raise RuntimeError(f"missing clean-map file: {file}")
        digest.update(name.encode("utf-8"))
        digest.update(file.read_bytes())
    return digest.hexdigest()


class TCDataset(Dataset):
    def __init__(self, tc_cache: TCV2Cache, ec_cache: EcCache, split: str,
                 nshot_min: int, nshot_max: int, seed: int,
                 clean_map: Path | None = None):
        self.tc = tc_cache
        self.ec = ec_cache
        self.split = split
        self.nshot_min = nshot_min
        self.nshot_max = nshot_max
        self.rng = random.Random(seed)
        self.rows = []
        self.refs = {}
        for key in tc_cache.keys:
            _, role, part, font, cp = key.split("|", 4)
            if part != split:
                continue
            if role == "target":
                self.rows.append((font, cp))
            elif role == "ref":
                self.refs.setdefault(font, []).append(cp)
        for font in self.refs:
            self.refs[font].sort()
        self.weights = [1.0] * len(self.rows)
        clean = clean_map or (Path(self.tc.manifest["clean_map"])
                              if self.tc.manifest.get("clean_map") else None)
        if split == "train" and clean:
            clean_dir = Path(clean)
            pair_file = clean_dir / "pairs_train.tsv"
            weight_file = clean_dir / "sample_weights.json"
            if pair_file.is_file() and weight_file.is_file():
                pair_group = {(row["font"], row["cp"]): row["script_group"]
                              for row in csv.DictReader(pair_file.open(encoding="utf-8"), delimiter="\t")}
                groups = json.loads(weight_file.read_text(encoding="utf-8"))["pair_weight"]
                self.weights = [float(groups[pair_group[row]]) for row in self.rows]
            else:
                raise RuntimeError(f"clean-map files missing under {clean}")
        if not self.rows:
            raise RuntimeError(f"TC cache has no {split} target rows")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        font, cp = self.rows[index]
        choices = self.refs.get(font, [])
        if not choices:
            raise RuntimeError(f"TC cache has no refs for {self.split}/{font}")
        if self.split == "val":
            n = min(self.nshot_max, len(choices))
            refs = choices[:n]
        else:
            n = self.rng.randint(self.nshot_min, min(self.nshot_max, len(choices)))
            refs = self.rng.sample(choices, n)
        ref_desc = torch.stack([self.tc.stats("ref", self.split, font, ref)
                                for ref in refs])
        ec = pooled_ec_features(self.ec.features("content", "", cp))[0]
        target = self.tc.stats("target", self.split, font, cp)
        return ref_desc, ec, target


def collate(batch):
    refs, ec, target = zip(*batch)
    n_max = max(x.shape[0] for x in refs)
    padded, masks = [], []
    for row in refs:
        padded.append(torch.cat([row, torch.zeros(n_max - row.shape[0], row.shape[1])]))
        masks.append(torch.cat([torch.ones(row.shape[0], dtype=torch.bool),
                                torch.zeros(n_max - row.shape[0], dtype=torch.bool)]))
    return torch.stack(padded), torch.stack(masks), torch.stack(ec), torch.stack(target)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tc-cache", type=Path, required=True)
    ap.add_argument("--ec-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--nshot-min", type=int, default=1)
    ap.add_argument("--nshot-max", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--clean-map", type=Path, default=None,
                    help="Override cache clean-map path (must match cache manifest hash).")
    args = ap.parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.warmup_steps < 0 or args.eval_interval <= 0:
        raise ValueError("steps/batch-size/eval-interval must be positive; warmup must be non-negative")
    if args.nshot_min <= 0 or args.nshot_max < args.nshot_min:
        raise ValueError("invalid nshot range")
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    tc = TCV2Cache(args.tc_cache)
    ec = EcCache(args.ec_cache)
    manifest_clean = Path(tc.manifest["clean_map"]) if tc.manifest.get("clean_map") else None
    clean_dir = args.clean_map or manifest_clean
    if tc.manifest.get("clean_map_sha256"):
        if clean_dir is None or clean_map_digest(clean_dir) != tc.manifest["clean_map_sha256"]:
            raise RuntimeError("clean-map path/hash does not match TC cache manifest")
    ds = TCDataset(tc, ec, "train", args.nshot_min, args.nshot_max, args.seed, clean_dir)
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(ds.weights, dtype=torch.double),
                                                      num_samples=len(ds), replacement=True)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        collate_fn=collate, num_workers=0)
    val_ds = TCDataset(tc, ec, "val", args.nshot_min, args.nshot_max, args.seed + 1, clean_dir)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)
    it = iter(loader)
    model = TCV2Head().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    losses = []
    for step in range(1, args.steps + 1):
        try:
            refs, mask, ec_vec, target = next(it)
        except StopIteration:
            it = iter(loader)
            refs, mask, ec_vec, target = next(it)
        refs, mask = refs.to(args.device), mask.to(args.device)
        ec_vec, target = ec_vec.to(args.device), target.to(args.device)
        scale = min(1.0, step / max(1, args.warmup_steps))
        for group in opt.param_groups:
            group["lr"] = args.lr * scale
        pred = model(refs, ec_vec, mask)
        loss = F.smooth_l1_loss(pred.float(), target.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if step == 1 or step % 100 == 0:
            print(f"step={step} loss={losses[-1]:.6f}", flush=True)
        if val_loader is not None and (step == args.steps or step % args.eval_interval == 0):
            model.eval()
            val_total, val_count = 0.0, 0
            with torch.no_grad():
                for refs, mask, ec_vec, target in val_loader:
                    pred = model(refs.to(args.device), ec_vec.to(args.device), mask.to(args.device))
                    b = target.shape[0]
                    val_total += float(F.smooth_l1_loss(pred.float(), target.to(args.device).float())) * b
                    val_count += b
            print(f"step={step} val_loss={val_total / max(1, val_count):.6f}", flush=True)
            model.train()
    torch.save(model.state_dict(), args.out / "tc_head.pth")
    (args.out / "manifest.json").write_text(json.dumps({
        "kind": "tc_v2_head_pretrain",
        "tc_cache_manifest_sha256": cache_fingerprint(tc.manifest),
        "ec_checkpoint_sha256": ec.manifest.get("ec_checkpoint_sha256"),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "weighted_sampler": True,
        "seed": args.seed,
        "last_loss": losses[-1],
    }, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
