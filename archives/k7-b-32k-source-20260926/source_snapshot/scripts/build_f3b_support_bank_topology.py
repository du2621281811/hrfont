#!/usr/bin/env python3
"""Build F3b topology-selected support bank (PI 2026-09-09, pinned).

Authoritative F3b support-char scheme: B0 skeleton topology-geometry signature
ranking, per target char -> top-16 CN style-pool chars.

Signature (block-wise cosine, weights counts 0.45 / direction 0.25 /
projection 0.25 / geometry 0.05; ties by Unicode ascending; fully deterministic):
  - counts: log1p(holes, 8-conn foreground components, skeleton endpoints,
    junction clusters), normalized by per-dim max over the full B0 table,
    then unit-norm.
  - direction: 8-bin undirected skeleton-edge direction histogram (pixel->
    neighbour angle mod pi), junction-adjacent edges excluded, unit-norm.
  - projection: concat(8-bin row ink profile, 8-bin col ink profile), unit-norm.
  - geometry: [ink density, bbox aspect ratio], unit-norm.

Known limits (documented, P0-guarded): counter/vertical classes rank well
(o/0/Q -> ri/yue/bai/ju...; l -> gong/shi/yi), steep-slant triangle letters
(A) rank boxy compounds rather than ren/da; P0 (relevant vs random vs
anti-relevant, same checkpoint) adjudicates whether the bank beats random.

Outputs (deterministic, no RNG):
  artifacts/f0/support_bank_f3b_topology.json           <- train.py bank
  artifacts/f0/topology_signatures_f3b.json             <- full signature table
  artifacts/f0/support_bank_f3b_topology.manifest.json  <- SHA + params + font SHA
  artifacts/f0/preview_support_bank_f3b_topology.html   <- review panel
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]

BLOCK_W = dict(counts=0.45, direction=0.25, projection=0.25, geometry=0.05)


# ------------------------------------------------------------------ render

def fit_font_size(font_path: str, chars: list[str], canvas: int, inner: int) -> int:
    lo, hi = 1, canvas
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(font_path, mid)
        ok = True
        for ch in chars:
            bbox = font.getbbox(ch)
            if bbox[2] - bbox[0] > inner or bbox[3] - bbox[1] > inner:
                ok = False
                break
        if ok:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def render_glyph(font: ImageFont.FreeTypeFont, ch: str, canvas: int) -> np.ndarray:
    img = Image.new("L", (canvas, canvas), 255)
    d = ImageDraw.Draw(img)
    bbox = font.getbbox(ch)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (canvas - w) // 2 - bbox[0]
    y = (canvas - h) // 2 - bbox[1]
    d.text((x, y), ch, fill=0, font=font)
    return np.asarray(img)


# --------------------------------------------------------------- signature

def skeleton_features(binary: np.ndarray) -> dict:
    skel = skeletonize(binary > 0)
    si = skel.astype(int)
    _, n_comp = ndi.label(binary > 0, structure=np.ones((3, 3), int))
    bg, n_bg = ndi.label(binary == 0, structure=np.ones((3, 3), int))
    border = set(np.concatenate(
        [bg[0].ravel(), bg[-1].ravel(), bg[:, 0].ravel(), bg[:, -1].ravel()]))
    n_holes = sum(1 for i in range(1, n_bg + 1) if i not in border)
    nbr = ndi.convolve(si, np.ones((3, 3), int), mode="constant") - si
    n_ep = int(((nbr == 1) & (si > 0)).sum())
    jm = (nbr >= 3) & (si > 0)
    _, n_junc = ndi.label(jm, structure=np.ones((3, 3), int))
    # direction: skeleton edges, junction-adjacent excluded
    hist = np.zeros(8)
    ys, xs = np.nonzero(si)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            yy = np.clip(ys + dy, 0, skel.shape[0] - 1)
            xx = np.clip(xs + dx, 0, skel.shape[1] - 1)
            m = (si[yy, xx] > 0) & ~jm[ys, xs]
            ang = np.arctan2(dy, dx) % math.pi
            b = np.clip((ang / (math.pi / 8)).astype(int), 0, 7)
            for bi in range(8):
                hist[bi] += m[b == bi].sum()
    rows = binary.sum(axis=1)
    cols = binary.sum(axis=0)
    step = binary.shape[0] // 8
    hproj = np.array([rows[i * step:(i + 1) * step].sum() for i in range(8)], float)
    vproj = np.array([cols[i * step:(i + 1) * step].sum() for i in range(8)], float)
    hproj /= hproj.sum() + 1e-6
    vproj /= vproj.sum() + 1e-6
    ys2, xs2 = np.nonzero(binary)
    geom = np.array([
        binary.mean(),
        (xs2.max() - xs2.min() + 1) / (ys2.max() - ys2.min() + 1),
    ])
    return dict(
        holes=n_holes, components=int(n_comp), endpoints=n_ep, junctions=int(n_junc),
        direction=hist, hproj=hproj, vproj=vproj, geometry=geom,
    )


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def block_cosine(fa: dict, fb: dict, counts_scale: np.ndarray) -> float:
    ca = unit(np.log1p([fa["holes"], fa["components"], fa["endpoints"],
                        fa["junctions"]]) / counts_scale)
    cb = unit(np.log1p([fb["holes"], fb["components"], fb["endpoints"],
                        fb["junctions"]]) / counts_scale)
    s_counts = float(np.dot(ca, cb))
    s_dir = float(np.dot(unit(fa["direction"]), unit(fb["direction"])))
    pa = np.concatenate([fa["hproj"], fa["vproj"]])
    pb = np.concatenate([fb["hproj"], fb["vproj"]])
    s_proj = float(np.dot(unit(pa), unit(pb)))
    s_geom = float(np.dot(unit(fa["geometry"]), unit(fb["geometry"])))
    return (BLOCK_W["counts"] * s_counts + BLOCK_W["direction"] * s_dir
            + BLOCK_W["projection"] * s_proj + BLOCK_W["geometry"] * s_geom)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", default=str(
        ROOT / "artifacts/e12/fonts/noto_sans_sc/NotoSansSC[wght].ttf"))
    ap.add_argument("--charset", default=str(
        ROOT / "manifests/charset_cn2west_v2_planned.json"))
    ap.add_argument("--canvas", type=int, default=128)
    ap.add_argument("--inner", type=int, default=112)
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--out_dir", default=str(ROOT / "artifacts/f0"))
    args = ap.parse_args()

    font_path = Path(args.font)
    if not font_path.is_file():
        raise SystemExit(f"font not found: {font_path}")
    cs = json.loads(Path(args.charset).read_text(encoding="utf-8"))
    han = list(cs["style_han_338"])
    targets: list[str] = []
    for chars in cs["target"].values():
        targets.extend(chars)
    all_chars = sorted(set(han) | set(targets))
    assert len(han) == 338, f"style_han_338 expected 338, got {len(han)}"
    assert len(targets) == 295, f"targets expected 295, got {len(targets)}"

    fs = fit_font_size(str(font_path), all_chars, args.canvas, args.inner)
    font = ImageFont.truetype(str(font_path), fs)
    try:
        font.set_variation_by_axes([400])
    except Exception:
        pass

    feats: dict[str, dict] = {}
    missing: list[str] = []
    for ch in all_chars:
        img = render_glyph(font, ch, args.canvas)
        if img.min() > 128:
            missing.append(ch)
            continue
        feats[ch] = skeleton_features(img > 128)

    counts_scale = np.maximum(np.array([
        np.log1p([feats[c]["holes"], feats[c]["components"],
                  feats[c]["endpoints"], feats[c]["junctions"]])
        for c in han
    ]).max(axis=0), 1e-6)

    han_set = {c: feats[c] for c in han if c in feats}
    bank: dict[str, list[str]] = {}
    ranks: dict[str, list[tuple[str, float]]] = {}
    for ch in targets:
        if ch not in feats:
            bank[f"u{ord(ch):04X}"] = []
            ranks[ch] = []
            continue
        scored = [(block_cosine(feats[ch], hf, counts_scale), hc)
                  for hc, hf in han_set.items()]
        scored.sort(key=lambda t: (-t[0], t[1]))
        top = scored[: args.top]
        bank[f"u{ord(ch):04X}"] = [f"u{ord(hc):04X}" for _, hc in top]
        ranks[ch] = [(hc, s) for s, hc in top]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sig_table = {
        "font_path": str(font_path),
        "font_sha256": sha256_file(font_path),
        "canvas": args.canvas,
        "inner": args.inner,
        "font_size": fs,
        "top": args.top,
        "block_weights": BLOCK_W,
        "counts_scale_log1p_max": counts_scale.tolist(),
        "missing_glyphs": missing,
        "signatures": {
            ch: {
                "holes": feats[ch]["holes"],
                "components": feats[ch]["components"],
                "endpoints": feats[ch]["endpoints"],
                "junctions": feats[ch]["junctions"],
                "direction": feats[ch]["direction"].tolist(),
                "hproj": feats[ch]["hproj"].tolist(),
                "vproj": feats[ch]["vproj"].tolist(),
                "geometry": feats[ch]["geometry"].tolist(),
            }
            for ch in sorted(feats)
        },
    }
    sig_json = json.dumps(sig_table, ensure_ascii=False, indent=2)
    sig_path = out_dir / "topology_signatures_f3b.json"
    sig_path.write_text(sig_json + "\n", encoding="utf-8")

    bank_payload = {
        "note": (
            "F3b PI 2026-09-09 (pinned): topology-geometry selection. B0 skeleton "
            "signature ranking per target char -> top-16 CN style-pool chars; "
            "own-font Ec features consumed by train.py. Deterministic: ties by "
            "Unicode ascending. Train samples k_s~U{4..16} from the ranked list; "
            "inference uses first support_k (top-ranked)."
        ),
        "mode": "topology_top16",
        "k": 8,
        "k_train_min": 4,
        "k_train_max": 16,
        "sample_random_train": True,
        "top": args.top,
        "n_query": len(bank),
        "support": bank,
        "signature_sha256": sha256_text(sig_json),
        "builder": "scripts/build_f3b_support_bank_topology.py",
    }
    bank_path = out_dir / "support_bank_f3b_topology.json"
    bank_json = json.dumps(bank_payload, ensure_ascii=False, indent=2)
    bank_path.write_text(bank_json + "\n", encoding="utf-8")

    manifest = {
        "builder": "scripts/build_f3b_support_bank_topology.py",
        "font_path": str(font_path),
        "font_sha256": sha256_file(font_path),
        "canvas": args.canvas,
        "inner": args.inner,
        "font_size": fs,
        "top": args.top,
        "block_weights": BLOCK_W,
        "counts_scale_log1p_max": counts_scale.tolist(),
        "missing_glyphs": missing,
        "n_target": len(bank),
        "n_han": len(han_set),
        "signature_table_sha256": sha256_file(sig_path),
        "bank_sha256": sha256_file(bank_path),
        "schema": {
            "bank_keys": "uXXXX target cp -> list[uXXXX style cp]",
            "compat": "train.py bank['support'][target_cp]",
        },
    }
    manifest_path = out_dir / "support_bank_f3b_topology.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    preview = build_preview(ranks, font, args.canvas, missing, manifest)
    pv_path = out_dir / "preview_support_bank_f3b_topology.html"
    pv_path.write_text(preview, encoding="utf-8")

    empty_banks = sum(1 for v in bank.values() if not v)
    print(f"bank={bank_path} queries={len(bank)} empty={empty_banks} "
          f"missing_glyphs={len(missing)} fs={fs}")
    print(f"signatures={sig_path}")
    print(f"manifest={manifest_path}")
    print(f"preview={pv_path}")
    print(f"font_sha256={manifest['font_sha256']}")
    print(f"bank_sha256={manifest['bank_sha256']}")


def build_preview(ranks, font, canvas, missing, manifest) -> str:
    def glyph_b64(ch: str, size: int = 44) -> str:
        img = Image.new("L", (size, size), 255)
        d = ImageDraw.Draw(img)
        f2 = font.font_variant(size=size)
        bbox = f2.getbbox(ch)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text(((size - w) // 2 - bbox[0], (size - h) // 2 - bbox[1]),
               ch, fill=0, font=f2)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    script_examples: dict[str, list[str]] = {}
    for ch in ranks:
        name = unicodedata.name(ch, "")
        if "HIRAGANA" in name:
            k = "hiragana"
        elif "KATAKANA" in name:
            k = "katakana"
        elif "BOPOMOFO" in name:
            k = "bopomofo"
        elif ch.isdigit():
            k = "digit"
        elif "LATIN" in name and ch.isupper():
            k = "latin_upper"
        elif "LATIN" in name and ch.islower() and ord(ch) > 127:
            k = "latin_ext"
        elif "LATIN" in name and ch.islower():
            k = "latin_lower"
        else:
            k = "other"
        script_examples.setdefault(k, []).append(ch)

    parts = [
        "<html><head><meta charset='utf-8'>",
        "<style>body{font-family:Menlo,monospace;background:#fff;color:#111;padding:16px}",
        ".row{display:flex;align-items:center;gap:6px;margin:6px 0;flex-wrap:wrap}",
        ".tgt{font-weight:bold;font-size:20px;width:1.6em;text-align:center}",
        ".cn{font-size:13px;width:2.6em;text-align:center;color:#333}",
        ".sim{font-size:10px;color:#888;width:3.2em}",
        ".sec{margin-top:20px;font-weight:bold}",
        "</style></head><body>",
        "<h3>F3b topology support bank — B0 (Noto Sans SC) preview</h3>",
        f"<p>canvas={manifest['canvas']} fs={manifest['font_size']} "
        f"top={manifest['top']} missing={len(missing)}<br>",
        "每个目标字符（粗体）→ top-16 CN 字（相似度）</p>",
    ]
    order = ["digit", "latin_upper", "latin_lower", "latin_ext",
             "hiragana", "katakana", "bopomofo", "other"]
    for k in order:
        if k not in script_examples:
            continue
        parts.append(f"<div class='sec'>{k}</div>")
        for ch in script_examples[k][:3]:
            top = ranks.get(ch, [])
            row = [f"<div class='row'><span class='tgt'>{ch}</span>"]
            for hc, sim in top[:16]:
                row.append(f"<span class='cn'><img "
                           f"src='data:image/png;base64,{glyph_b64(hc)}'><br>{hc}</span>")
                row.append(f"<span class='sim'>{sim:.3f}</span>")
            row.append("</div>")
            parts.append("".join(row))
    parts.append("</body></html>")
    return "".join(parts)


if __name__ == "__main__":
    main()
