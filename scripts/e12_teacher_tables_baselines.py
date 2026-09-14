#!/usr/bin/env python3
"""Teacher Tables 1–3: Ours + CLIP / DINOv2 / LPIPS-Alex feature baselines.

Same protocol as e12_eval_retrieval_cosine / e12_validate_auc_cosine:
  - test families only (70/15/15 by group, seed 3407)
  - zh ref8 mean-pool proto vs latin(+digit) queries
  - cosine similarity (LPIPS also reported as 1 - distance for verification)

Table 3 separation (Ours Style Score S01) on GT queries:
  matched / hard (same coarse weight class) / random / wrong-family

    HF_ENDPOINT=https://hf-mirror.com \\
    /root/miniforge3/envs/boogu/bin/python scripts/e12_teacher_tables_baselines.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path("/root/projects/hrfont")
sys.path.insert(0, str(ROOT / "scripts" / "eval_framework"))
from data import (  # noqa: E402
    GlyphDataset,
    as_chars,
    group_table,
    load_manifest,
    pick_negative,
    split_families,
)
from models import load_phi_checkpoint  # noqa: E402
from train_utils import binary_auc  # noqa: E402

CACHE = ROOT / "artifacts/e12/cache_v0913_b"
PHI = ROOT / "runs/e12_phi_s2_b_s3407/best.pt"
OUT = ROOT / "reports/e12_paper"
REF8 = list("永和书风骨韵天地")
QUERY_LATIN = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
QUERY_LD = QUERY_LATIN + list("0123456789")
SPLIT_RATIOS = (0.7, 0.15, 0.15)
SPLIT_SEED = 3407

# Coarse weight buckets from Founder-style stems
_LIGHT = {"l", "el", "ultralight", "extralight", "thin", "light", "demilight"}
_REG = {"r", "m", "regular", "normal", "medium", "book", "roman"}
_BOLD = {"b", "h", "db", "eb", "ub", "bold", "semibold", "demibold", "extrabold",
         "ultrabold", "black", "heavy", "cu", "da", "te"}
_ITAL = {"italic", "oblique"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def coarse_weight(family: str) -> str:
    """Parse Founder-style stems into light/regular/bold/italic/unknown."""
    parts = re.split(r"[-_]", family)
    tags: set[str] = set()
    for p in parts[1:]:
        pl = p.lower()
        tags.add(pl)
        # e.g. 512B / 510M / 507R / 515H
        m = re.match(r"^\d+([a-z]+)$", pl)
        if m:
            tags.add(m.group(1))
        # e.g. jw-h already split; lone letter tokens
        if len(pl) == 1:
            tags.add(pl)
    if tags & _ITAL:
        return "italic"
    if tags & _BOLD:
        return "bold"
    if tags & _LIGHT:
        return "light"
    if tags & _REG:
        return "regular"
    return "unknown"


def to_pil_rgb(tensor_chw: torch.Tensor) -> Image.Image:
    """GlyphDataset returns float [0,1] CxHxW (often grayscale repeated)."""
    x = tensor_chw.detach().cpu()
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    arr = (x.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


class FeatureBackbone:
    name: str

    def embed_batch(self, tensors: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        raise NotImplementedError


class OursPhi(FeatureBackbone):
    name = "Ours"

    def __init__(self, path: Path, device: torch.device):
        self.model = load_phi_checkpoint(path, 3, 512, str(device)).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed_batch(self, tensors: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        x = torch.stack(tensors).to(device)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        z = self.model(x)
        return F.normalize(z, dim=1)


class CLIPBackbone(FeatureBackbone):
    name = "CLIP"

    def __init__(self, device: torch.device):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device

    @torch.no_grad()
    def embed_batch(self, tensors: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        images = [to_pil_rgb(t) for t in tensors]
        inputs = self.processor(images=images, return_tensors="pt")
        pixel = inputs["pixel_values"].to(device)
        # Some transformers builds return a ModelOutput from get_image_features;
        # project pooled vision features explicitly.
        vision = self.model.vision_model(pixel_values=pixel)
        pooled = vision.pooler_output
        z = self.model.visual_projection(pooled)
        return F.normalize(z, dim=1)


class DINOv2Backbone(FeatureBackbone):
    name = "DINOv2"

    def __init__(self, device: torch.device):
        from transformers import AutoImageProcessor, AutoModel

        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.device = device

    @torch.no_grad()
    def embed_batch(self, tensors: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        images = [to_pil_rgb(t) for t in tensors]
        inputs = self.processor(images=images, return_tensors="pt")
        pixel = inputs["pixel_values"].to(device)
        out = self.model(pixel_values=pixel)
        z = out.last_hidden_state[:, 0]  # CLS
        return F.normalize(z, dim=1)


class LPIPSAlexFeature(FeatureBackbone):
    """AlexNet trunk features (LPIPS backbone), cosine on concatenated layer feats."""

    name = "LPIPS-Alex"

    def __init__(self, device: torch.device):
        import lpips

        self.lpips = lpips.LPIPS(net="alex").to(device).eval()
        self.device = device
        self.tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def embed_batch(self, tensors: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        xs = []
        for t in tensors:
            x = t
            if x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
            xs.append(self.tf(x))
        x = torch.stack(xs).to(device)
        # lpips expects ~[-1,1] for distance; for trunk use ImageNet-norm via net
        feats = self.lpips.net.forward(x)
        # feats is list of layer activations
        flat = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]
        z = torch.cat(flat, dim=1)
        return F.normalize(z, dim=1)

    @torch.no_grad()
    def lpips_distance(self, a: torch.Tensor, b: torch.Tensor, device: torch.device) -> float:
        """Single-pair LPIPS distance (lower = more similar)."""
        def prep(t):
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
            t = F.interpolate(t[None], size=(64, 64), mode="bilinear", align_corners=False)
            return t * 2 - 1  # [0,1] -> [-1,1]

        d = self.lpips(prep(a).to(device), prep(b).to(device))
        return float(d.view(-1)[0])


def embed_all(backbone: FeatureBackbone, ds: GlyphDataset, by: dict, keys: list[tuple[str, str]],
              device: torch.device, batch: int = 32) -> dict[tuple[str, str], torch.Tensor]:
    out: dict[tuple[str, str], torch.Tensor] = {}
    buf_t, buf_k = [], []
    for key in keys:
        if key not in by:
            continue
        buf_t.append(ds[by[key]][0])
        buf_k.append(key)
        if len(buf_t) >= batch:
            z = backbone.embed_batch(buf_t, device).cpu()
            for i, k in enumerate(buf_k):
                out[k] = z[i]
            buf_t, buf_k = [], []
    if buf_t:
        z = backbone.embed_batch(buf_t, device).cpu()
        for i, k in enumerate(buf_k):
            out[k] = z[i]
    return out


def proto_of(emb: dict, family: str) -> torch.Tensor | None:
    zs = [emb[(family, c)] for c in REF8 if (family, c) in emb]
    if len(zs) < len(REF8):
        return None
    return F.normalize(torch.stack(zs).mean(0), dim=0)


def run_retrieval(name: str, emb: dict, families: list[str]) -> dict:
    protos = {}
    usable = []
    for f in families:
        p = proto_of(emb, f)
        if p is None:
            continue
        protos[f] = p
        usable.append(f)
    proto_mat = torch.stack([protos[f] for f in usable])
    rows = []
    for f in usable:
        for ch in QUERY_LATIN:
            if (f, ch) not in emb:
                continue
            z = F.normalize(emb[(f, ch)], dim=0)
            sims = proto_mat @ z
            order = torch.argsort(sims, descending=True)
            rank = int((order == usable.index(f)).nonzero(as_tuple=True)[0]) + 1
            rows.append({"family": f, "char": ch, "rank": rank, "sc": float(sims[usable.index(f)])})
    n = len(rows)
    return {
        "method": name,
        "n_queries": n,
        "n_gallery": len(usable),
        "R@1": float(np.mean([r["rank"] == 1 for r in rows])) if n else None,
        "R@5": float(np.mean([r["rank"] <= 5 for r in rows])) if n else None,
        "MRR": float(np.mean([1.0 / r["rank"] for r in rows])) if n else None,
    }


def run_verification(name: str, emb: dict, test_families: list[str], all_families: list[str],
                     groups: dict, max_per_family: int = 16) -> dict:
    pos, neg, order_ok = [], [], []
    for f in test_families:
        qs = [c for c in QUERY_LD if (f, c) in emb and proto_of(emb, f) is not None]
        if not qs:
            continue
        wrong = pick_negative(f, all_families, groups, cross_group=True)
        if wrong is None or proto_of(emb, wrong) is None:
            wrong = pick_negative(
                f, [x for x in all_families if proto_of(emb, x) is not None], groups, cross_group=True
            )
        if wrong is None:
            continue
        proto = proto_of(emb, f)
        step = max(1, len(qs) // max_per_family)
        use_q = qs[::step][:max_per_family]
        for ch in use_q:
            if (f, ch) not in emb or (wrong, ch) not in emb:
                continue
            cos_s = float((proto * F.normalize(emb[(f, ch)], dim=0)).sum())
            cos_c = float((proto * F.normalize(emb[(wrong, ch)], dim=0)).sum())
            pos.append(cos_s)
            neg.append(cos_c)
            order_ok.append(cos_s > cos_c)
    labels = [1] * len(pos) + [0] * len(neg)
    scores = pos + neg
    return {
        "method": name,
        "n_pairs": len(pos),
        "ROC-AUC": float(binary_auc(labels, scores)) if pos and neg else None,
        "pairwise_order_acc": float(np.mean(order_ok)) if order_ok else None,
        "mean_cos_matched": float(np.mean(pos)) if pos else None,
        "mean_cos_mismatched": float(np.mean(neg)) if neg else None,
    }


def run_separation_ours(emb: dict, test_families: list[str], rng: random.Random) -> dict:
    """Style Score S01 separation for Ours on GT latin/digit queries."""
    buckets = defaultdict(list)
    for f in test_families:
        if proto_of(emb, f) is None:
            continue
        buckets[coarse_weight(f)].append(f)

    matched, hard, rand_neg = [], [], []
    for f in test_families:
        proto = proto_of(emb, f)
        if proto is None:
            continue
        qs = [c for c in QUERY_LD if (f, c) in emb]
        if not qs:
            continue
        w = coarse_weight(f)
        # Hard: same known weight class (exclude unknown–unknown, which is not attribute-hard)
        hard_pool = [
            x
            for x in buckets.get(w, [])
            if x != f and w != "unknown" and proto_of(emb, x) is not None
        ]
        any_pool = [x for x in test_families if x != f and proto_of(emb, x) is not None]
        for ch in qs[:: max(1, len(qs) // 8)][:8]:
            z = F.normalize(emb[(f, ch)], dim=0)
            matched.append((float((proto * z).sum()) + 1) / 2)

            if hard_pool:
                hf = hard_pool[rng.randrange(len(hard_pool))]
                if (hf, ch) in emb:
                    zh = F.normalize(emb[(hf, ch)], dim=0)
                    hard.append((float((proto * zh).sum()) + 1) / 2)

            if any_pool:
                rf = any_pool[rng.randrange(len(any_pool))]
                if (rf, ch) in emb:
                    zr = F.normalize(emb[(rf, ch)], dim=0)
                    rand_neg.append((float((proto * zr).sum()) + 1) / 2)

    def stats(xs):
        if not xs:
            return {"n": 0, "mean": None, "std": None}
        a = np.asarray(xs, dtype=np.float64)
        return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=0))}

    return {
        "metric": "style_score_01 = (cosine+1)/2",
        "matched": stats(matched),
        "hard_negative_same_weight_class": stats(hard),
        "random_negative": stats(rand_neg),
        "same_coarse_category_different_font": stats(hard),
        "weight_class_counts": {k: len(v) for k, v in sorted(buckets.items())},
        "note": (
            "hard = different test font with same parsed weight class "
            "(light/regular/bold/italic); unknown-class fonts excluded from hard pool"
        ),
    }


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--skip-baselines", action="store_true", help="Ours only")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    manifest = load_manifest(CACHE)
    splits = split_families(manifest, SPLIT_RATIOS, SPLIT_SEED, by_group=True)
    groups = group_table(manifest)
    test_families = sorted(splits["test"])
    all_families = sorted({x["family"] for x in manifest["fonts"]})

    need_chars = as_chars(REF8) + as_chars(QUERY_LD)
    ds = GlyphDataset(CACHE, all_families, need_chars, 3)
    by = {(r["family"], r["char"]): i for i, r in enumerate(ds.records)}
    keys = [(f, c) for f in all_families for c in need_chars if (f, c) in by]

    backbones: list[FeatureBackbone] = [OursPhi(PHI, device)]
    if not args.skip_baselines:
        print("[load] CLIP…", flush=True)
        backbones.append(CLIPBackbone(device))
        print("[load] DINOv2…", flush=True)
        backbones.append(DINOv2Backbone(device))
        print("[load] LPIPS-Alex…", flush=True)
        backbones.append(LPIPSAlexFeature(device))

    table1, table2 = [], []
    separation = None
    for bb in backbones:
        print(f"[embed] {bb.name}", flush=True)
        emb = embed_all(bb, ds, by, keys, device, args.batch)
        # free GPU for next model if needed — keep CPU emb
        if hasattr(bb, "model"):
            del bb.model
        if hasattr(bb, "lpips"):
            # keep for nothing
            pass
        torch.cuda.empty_cache()

        t1 = run_retrieval(bb.name, emb, test_families)
        t2 = run_verification(bb.name, emb, test_families, all_families, groups)
        table1.append(t1)
        table2.append(t2)
        print(f"  retrieval {t1}", flush=True)
        print(f"  verify {t2}", flush=True)
        if bb.name == "Ours":
            separation = run_separation_ours(emb, test_families, random.Random(SPLIT_SEED))

    payload = {
        "generated_at": utc_now(),
        "protocol": {
            "cache": str(CACHE),
            "phi": str(PHI),
            "split": "70/15/15 by_group seed 3407; metrics on test families",
            "ref8": REF8,
            "retrieval_query": "A-Z a-z",
            "verification_query": "A-Z a-z 0-9",
            "score": "cosine on mean-pooled zh proto vs en query (except LPIPS distance ablation N/A here)",
        },
        "table1_retrieval": table1,
        "table2_verification": table2,
        "table3_separation_ours": separation,
    }
    out_p = OUT / "teacher_tables_1to3.json"
    out_p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", out_p)


if __name__ == "__main__":
    main()
