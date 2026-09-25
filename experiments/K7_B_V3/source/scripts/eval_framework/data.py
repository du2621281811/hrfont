#!/usr/bin/env python3
"""外部字体渲染缓存与 E12 数据集。字体族拆分以 manifest 的 family/stem 为单位。"""
from __future__ import annotations

import hashlib, json, random, re
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset


# Weight/style tokens that do NOT make a new typeface. NotoSansCJK-Bold and
# NotoSansCJK-Regular are one design, so they must never straddle a split, and one
# must never be the "wrong font" negative for the other.
_WEIGHT_TOKENS = {
    "thin", "extralight", "ultralight", "light", "demilight", "book", "regular",
    "normal", "medium", "semibold", "demibold", "bold", "extrabold", "ultrabold",
    "black", "heavy", "italic", "oblique", "roman", "r", "l", "m", "b", "h", "db", "eb",
}


def typeface_group(family: str) -> str:
    """Collapse weight/style variants onto their shared typeface."""
    parts = re.split(r"[-_]", family)
    while len(parts) > 1 and parts[-1].lower() in _WEIGHT_TOKENS:
        parts.pop()
    return "-".join(parts)


def group_of(row: dict) -> str:
    return row.get("group") or typeface_group(row["family"])


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def scan_fonts(fonts_dir: str | None = None, manifest: str | None = None) -> list[Path]:
    if bool(fonts_dir) == bool(manifest): raise ValueError("provide exactly one of fonts_dir or fonts_manifest")
    if manifest:
        base = Path(manifest).resolve().parent
        paths = [(base / x.strip()).resolve() if not Path(x.strip()).is_absolute() else Path(x.strip()) for x in Path(manifest).read_text().splitlines() if x.strip() and not x.startswith("#")]
    else:
        paths = [p for p in Path(fonts_dir).rglob("*") if p.suffix.lower() in {".ttf", ".otf", ".ttc"}]
    return sorted({p.resolve() for p in paths}, key=lambda p: str(p).casefold())


def _font_size(path: Path, chars: Sequence[str], strategy: str, fixed_size: int, canvas: int, margin: int) -> int:
    if strategy == "fixed_size": return fixed_size
    if strategy != "per_font_height_fit": raise ValueError(f"unknown size strategy: {strategy}")
    inner = canvas - 2 * margin; lo, hi, best = 8, max(16, canvas * 3), 8
    draw = ImageDraw.Draw(Image.new("L", (1, 1)))
    while lo <= hi:
        mid = (lo + hi) // 2; font = ImageFont.truetype(str(path), mid)
        height = max(draw.textbbox((0, 0), ch, font=font)[3] - draw.textbbox((0, 0), ch, font=font)[1] for ch in chars)
        if height <= inner: best, lo = mid, mid + 1
        else: hi = mid - 1
    return best


def render_glyph(path: Path, ch: str, size: int, canvas: int = 96) -> Image.Image:
    font = ImageFont.truetype(str(path), size); image = Image.new("L", (canvas, canvas), 255); draw = ImageDraw.Draw(image)
    box = draw.textbbox((0, 0), ch, font=font); w, h = box[2] - box[0], box[3] - box[1]
    draw.text(((canvas - w) // 2 - box[0], (canvas - h) // 2 - box[1]), ch, font=font, fill=0)
    return image


def build_cache(fonts: Sequence[Path], chars: Sequence[str], cache_dir: str | Path, canvas=96, strategy="fixed_size", fixed_size=80, margin=6) -> dict:
    cache = Path(cache_dir); cache.mkdir(parents=True, exist_ok=True)
    font_rows = [{"path": str(p), "stem": p.stem, "family": p.stem, "group": typeface_group(p.stem), "sha256": sha256_file(p), "id": f"{p.stem}-{sha256_file(p)[:10]}"} for p in fonts]
    spec = {"version": 1, "canvas": canvas, "strategy": strategy, "fixed_size": fixed_size, "margin": margin, "chars": list(chars), "fonts": font_rows}
    spec_sha = hashlib.sha256(json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest_path = cache / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("cache_sha256") != spec_sha: raise RuntimeError("cache manifest mismatch; choose a new cache_dir")
    records = []
    for row in font_rows:
        path = Path(row["path"]); size = _font_size(path, chars, strategy, fixed_size, canvas, margin)
        for ch in chars:
            rel = Path(row["id"]) / f"u{ord(ch):06X}.png"; target = cache / rel; target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists(): render_glyph(path, ch, size, canvas).save(target)
            records.append({"font_id": row["id"], "family": row["family"], "group": row["group"], "stem": row["stem"], "char": ch, "path": str(rel)})
    result = dict(spec, cache_sha256=spec_sha, records=records)
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return result


def as_chars(xs: Sequence) -> list[str]:
    """YAML turns bare `0`-`9` into ints; glyph keys are always strings."""
    return [str(x) for x in xs]


def load_manifest(cache_dir): return json.loads((Path(cache_dir) / "manifest.json").read_text(encoding="utf-8"))


def split_families(manifest: dict, ratios=(0.7, 0.15, 0.15), seed=3407, by_group: bool = True) -> dict[str, list[str]]:
    """Split fonts into train/val/test, by typeface group unless explicitly disabled.

    Splitting on raw family names lets weight variants of one typeface land on both
    sides, which inflates held-out scores. `by_group=False` only exists to reproduce
    pre-2026-09-06 runs.
    """
    rows = manifest["fonts"]
    if by_group:
        units = sorted({group_of(x) for x in rows})
        members = {u: sorted(x["family"] for x in rows if group_of(x) == u) for u in units}
    else:
        units = sorted({x["family"] for x in rows})
        members = {u: [u] for u in units}
    rng = random.Random(seed); rng.shuffle(units)
    n = len(units); n_train = max(1, int(n * ratios[0])); n_val = max(1, int(n * ratios[1])) if n >= 3 else 0
    if n_train + n_val >= n: n_train, n_val = max(1, n - 2), 1 if n >= 3 else 0
    chosen = {"train": units[:n_train], "val": units[n_train:n_train+n_val], "test": units[n_train+n_val:]}
    return {k: sorted(f for u in v for f in members[u]) for k, v in chosen.items()}


def group_table(manifest: dict) -> dict[str, str]:
    return {x["family"]: group_of(x) for x in manifest["fonts"]}


def assert_disjoint_splits(splits, groups: dict[str, str] | None = None):
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            overlap = set(splits[a]) & set(splits[b])
            if overlap: raise RuntimeError(f"font-family leakage {a}/{b}: {sorted(overlap)}")
            if groups:
                g = {groups[f] for f in splits[a]} & {groups[f] for f in splits[b]}
                if g: raise RuntimeError(f"typeface-group leakage {a}/{b}: {sorted(g)}")


def pick_negative(family: str, candidates: Sequence[str], groups: dict[str, str] | None,
                  cross_group: bool = True) -> str | None:
    """Deterministic negative font for `family`.

    With `cross_group`, the negative comes from a different typeface, so the score is
    not dominated by near-identical weight pairs (e.g. Medium vs Regular).
    """
    pool = [f for f in candidates if f != family]
    if cross_group and groups:
        cross = [f for f in pool if groups.get(f) != groups.get(family)]
        pool = cross or []
    if not pool:
        return None
    return pool[int(hashlib.sha256(family.encode()).hexdigest(), 16) % len(pool)]


class GlyphDataset(Dataset):
    def __init__(self, cache_dir, families=None, chars=None, channels=3):
        self.root = Path(cache_dir); manifest = load_manifest(cache_dir); fam = set(families or [x["family"] for x in manifest["fonts"]]); wanted = set(as_chars(chars) if chars else manifest["chars"])
        self.records = [x for x in manifest["records"] if x["family"] in fam and x["char"] in wanted]; self.channels = channels
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        row = self.records[index]; array = np.asarray(Image.open(self.root / row["path"]).convert("L"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0); tensor = tensor.repeat(self.channels, 1, 1) if self.channels > 1 else tensor
        return tensor, row


class CrossScriptPairDataset(Dataset):
    """每个索引固定一个 font/汉字，Latin 字符按 seed+index 均衡选取。batch 内其他字体为负。"""
    def __init__(self, cache_dir, chinese_chars, latin_chars, families=None, channels=3, seed=3407):
        chinese_chars, latin_chars = as_chars(chinese_chars), as_chars(latin_chars)
        self.glyphs = GlyphDataset(cache_dir, families, chinese_chars+latin_chars, channels); self.seed = seed
        self.by = {(r["family"], r["char"]): i for i, r in enumerate(self.glyphs.records)}; self.families = sorted({r["family"] for r in self.glyphs.records})
        self.chinese, self.latin = list(chinese_chars), list(latin_chars); self.keys = [(f, c) for f in self.families for c in self.chinese if (f,c) in self.by]
    def __len__(self): return len(self.keys)
    def __getitem__(self, index):
        family, ch = self.keys[index]; latin = self.latin[index % len(self.latin)]
        a, _ = self.glyphs[self.by[(family,ch)]]; b, _ = self.glyphs[self.by[(family,latin)]]
        return a, b, family


class GlyphClassDataset(Dataset):
    def __init__(self, cache_dir, chars, families=None, channels=3):
        self.chars = list(chars); self.base = GlyphDataset(cache_dir, families, self.chars, channels)
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, row = self.base[i]; return x, self.chars.index(row["char"]), row["family"], row["char"]


class MembershipDataset(Dataset):
    """确定性 episode；偶数同族，奇数异族；负样本 query 字符保持一致（hard negative）。"""
    def __init__(self, cache_dir, query_chars, ref_chars, families=None, channels=3, seed=3407, episodes=1000, visually_near=None):
        self.base = GlyphDataset(cache_dir, families, list(query_chars)+list(ref_chars), channels); self.query_chars=list(query_chars); self.ref_chars=list(ref_chars); self.seed=seed; self.episodes=episodes
        self.by={(r["family"],r["char"]):i for i,r in enumerate(self.base.records)}; self.families=sorted({r["family"] for r in self.base.records}); self.near=visually_near or {}
        if len(self.families)<2: raise ValueError("MembershipDataset needs >=2 families")
    def __len__(self): return self.episodes
    def __getitem__(self, i):
        # A positive/negative pair shares its reference family and query char.
        # Using i % n ties each family to one label whenever n is even.
        pair=i//2
        rng=random.Random(self.seed+pair); ref_family=self.families[pair%len(self.families)]; label=1 if i%2==0 else 0
        candidates=[x for x in self.near.get(ref_family,[]) if x in self.families and x!=ref_family] or [x for x in self.families if x!=ref_family]
        query_family=ref_family if label else candidates[rng.randrange(len(candidates))]; ch=self.query_chars[(pair//len(self.families))%len(self.query_chars)]
        query,_=self.base[self.by[(query_family,ch)]]; refs=torch.stack([self.base[self.by[(ref_family,r)]][0] for r in self.ref_chars])
        return query, refs, torch.tensor(float(label)), query_family, ref_family
