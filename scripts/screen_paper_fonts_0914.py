#!/usr/bin/env python3
"""Screen font_files_0914 for paper-showcase candidates.

Filters by distinctive Chinese style names, cmap coverage (CN ref + Latin),
CN↔Latin ink consistency proxy, and renders a review HTML.

Does NOT claim fonts are style-consistent — only ranks candidates for eye review.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/root/projects/hrfont")
FONT_DIR = Path("/root/font_files_0914/founder_fonts_download")
MANIFEST = FONT_DIR / "font_manifest.txt"
OUT = ROOT / "reports/paper_fonts_0914_screen"
KNOWN = set(
    p.name
    for sp in ("train", "val", "test")
    for p in (ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2" / sp / "TargetImage").iterdir()
    if p.is_dir()
)

CANVAS = 96
MARGIN = 6
INNER = CANVAS - 2 * MARGIN
T_INK = 250

REF8 = list("永和书风骨韵天地")
LATIN = list("AaGgRrQqEe08àě")
PROBE_CN = list("永和书风骨")
PROBE_LATIN = list("AaGgRr08e")

# Chinese display-name keywords → paper category (pick diversity)
STYLE_RULES: list[tuple[str, str]] = [
    (r"隶书|隶变|古隶|华隶|黑隶|铁筋隶|祥隶|楷隶|瘦隶|根隶|栋隶", "lishu"),
    (r"魏碑|北魏", "weibei"),
    (r"行楷|行草", "xingkai"),
    (r"草书|章草|大草|黄草", "caoshu"),
    (r"卡通|胖头鱼|胖娃|胖胖|启体", "cartoon"),
    (r"综艺|剪纸|琥珀|彩云|姚体|美黑|流行|超粗黑|康体", "display"),
    (r"篆", "zhuan"),
    (r"手写|字迹", "shouxie"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_manifest() -> list[dict]:
    rows = []
    text = MANIFEST.read_text(encoding="utf-8-sig")
    for line in text.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        fid, disp, fname = parts[0].strip(), parts[1].strip(), parts[2].strip()
        path = FONT_DIR / fname
        if not path.is_file():
            # try case variants
            hits = list(FONT_DIR.glob(fname))
            if not hits:
                continue
            path = hits[0]
        stem = path.stem
        # strip leading numeric id prefixes: 173_FZHTJW / 176_2309_FZSSJW
        m = re.match(r"^(?:\d+_)+(.+)$", stem)
        clean = m.group(1) if m else stem
        rows.append(
            {
                "id": fid,
                "disp": disp,
                "fname": path.name,
                "path": str(path),
                "stem": stem,
                "clean": clean,
            }
        )
    return rows


def categorize(disp: str) -> str | None:
    for pat, cat in STYLE_RULES:
        if re.search(pat, disp):
            return cat
    return None


def family_key(clean: str) -> str:
    s = re.sub(r"(JF|JW|FW)$", "", clean)
    for _ in range(3):
        s = re.sub(
            r"[-_](L|R|B|M|H|EB|SB|DB|EL|Te|Da|Cu|Zhong|Zhun|Xi|Xian|Italic)$",
            "",
            s,
            flags=re.I,
        )
    return s


def cmap_of(path: str) -> set[int] | None:
    try:
        f = TTFont(path, lazy=True, fontNumber=0)
        c = set((f.getBestCmap() or {}).keys())
        f.close()
        return c
    except Exception:
        return None


def coverage(cmap: set[int], chars: list[str]) -> tuple[int, list[str]]:
    miss = [ch for ch in chars if ord(ch) not in cmap]
    return len(chars) - len(miss), miss


def _max_bbox(ttf: str, fs: int, chars: list[str]) -> tuple[int, int]:
    font = ImageFont.truetype(ttf, fs)
    img = Image.new("L", (CANVAS, CANVAS), 255)
    draw = ImageDraw.Draw(img)
    mw = mh = 0
    for ch in chars:
        bbox = draw.textbbox((0, 0), ch, font=font)
        mw = max(mw, bbox[2] - bbox[0])
        mh = max(mh, bbox[3] - bbox[1])
    return mw, mh


def find_size(ttf: str, chars: list[str]) -> int:
    lo, hi, best = 8, 300, 10
    for _ in range(16):
        if lo > hi:
            break
        fs = (lo + hi) // 2
        mw, mh = _max_bbox(ttf, fs, chars)
        if mw <= INNER and mh <= INNER:
            best = fs
            lo = fs + 1
        else:
            hi = fs - 1
    return best


def render_glyph(ttf: str, ch: str, fs: int) -> Image.Image:
    font = ImageFont.truetype(ttf, fs)
    img = Image.new("L", (CANVAS, CANVAS), 255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), ch, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cur = fs
    while (w > INNER or h > INNER) and cur > 10:
        cur -= 1
        font = ImageFont.truetype(ttf, cur)
        bbox = draw.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (CANVAS - w) // 2 - bbox[0]
    y = (CANVAS - h) // 2 - bbox[1]
    draw.text((x, y), ch, fill=0, font=font)
    return img


def ink_stats(img: Image.Image) -> dict:
    a = np.asarray(img)
    ys, xs = np.where(a < T_INK)
    if len(ys) == 0:
        return {"fill": 0.0, "bbox": 0.0, "empty": True}
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    return {
        "fill": float(np.count_nonzero(a < T_INK)) / (CANVAS * CANVAS),
        "bbox": float((y1 - y0 + 1) * (x1 - x0 + 1)) / (CANVAS * CANVAS),
        "empty": False,
        "edge": bool(y0 <= 0 or x0 <= 0 or y1 >= CANVAS - 1 or x1 >= CANVAS - 1),
    }


def strip_row(imgs: list[Image.Image], labels: list[str], title: str) -> Image.Image:
    cell, lab_h, head = 96, 18, 22
    n = len(imgs)
    W = n * cell
    H = head + cell + lab_h
    out = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    d.text((4, 4), title, fill=(30, 30, 30), font=font)
    for i, (im, lab) in enumerate(zip(imgs, labels)):
        out.paste(im.convert("RGB"), (i * cell, head))
        d.text((i * cell + 4, head + cell + 2), lab, fill=(60, 60, 60), font=font)
    return out


def score_candidate(rec: dict) -> dict | None:
    path = rec["path"]
    cmap = cmap_of(path)
    if cmap is None:
        return None
    n_ref, miss_ref = coverage(cmap, REF8)
    n_lat, miss_lat = coverage(cmap, LATIN)
    if n_ref < len(REF8) or n_lat < len(LATIN) - 2:  # allow missing 1–2 accents
        return {
            **rec,
            "ok": False,
            "reason": "cmap",
            "n_ref": n_ref,
            "n_lat": n_lat,
            "miss_ref": miss_ref,
            "miss_lat": miss_lat,
        }

    chars = list(dict.fromkeys(PROBE_CN + PROBE_LATIN))
    try:
        fs = find_size(path, chars)
    except Exception as e:
        return {**rec, "ok": False, "reason": f"render:{e}"}

    cn_imgs, lat_imgs = [], []
    cn_fills, lat_fills = [], []
    empty = 0
    edge = 0
    for ch in PROBE_CN:
        im = render_glyph(path, ch, fs)
        st = ink_stats(im)
        cn_imgs.append(im)
        cn_fills.append(st["fill"])
        empty += int(st["empty"])
        edge += int(st.get("edge", False))
    for ch in PROBE_LATIN:
        im = render_glyph(path, ch, fs)
        st = ink_stats(im)
        lat_imgs.append(im)
        lat_fills.append(st["fill"])
        empty += int(st["empty"])
        edge += int(st.get("edge", False))

    cn_m = float(np.mean(cn_fills)) if cn_fills else 0.0
    lat_m = float(np.mean(lat_fills)) if lat_fills else 0.0
    # relative ink mismatch: high → CN/Latin weight/contrast diverge
    mismatch = abs(cn_m - lat_m) / max(cn_m, lat_m, 1e-6)
    # distinctiveness proxy: higher CN ink variance + farther from mid gray density
    distinct = float(np.std(cn_fills)) + abs(cn_m - 0.12)

    if empty > 0:
        return {**rec, "ok": False, "reason": "empty_glyph", "empty": empty}
    if cn_m < 0.02 or lat_m < 0.015:
        return {**rec, "ok": False, "reason": "too_thin", "cn_fill": cn_m, "lat_fill": lat_m}

    sheet = Image.new("RGB", (max(len(PROBE_CN), len(PROBE_LATIN)) * 96, 22 + 96 + 18 + 22 + 96 + 18), (235, 235, 235))
    r1 = strip_row(cn_imgs, PROBE_CN, f"CN · {rec['disp']}")
    r2 = strip_row(lat_imgs, PROBE_LATIN, f"Latin GT · mismatch={mismatch:.2f}")
    sheet.paste(r1, (0, 0))
    sheet.paste(r2, (0, r1.height))

    return {
        **rec,
        "ok": True,
        "fs": fs,
        "n_ref": n_ref,
        "n_lat": n_lat,
        "miss_lat": miss_lat,
        "cn_fill": cn_m,
        "lat_fill": lat_m,
        "mismatch": mismatch,
        "distinct": distinct,
        "edge_hits": edge,
        "sheet": sheet,
    }


def pick_diverse(ok_rows: list[dict], per_cat: int = 4, total: int = 24) -> list[dict]:
    by = defaultdict(list)
    for r in ok_rows:
        by[r["cat"]].append(r)
    for cat in by:
        # prefer lower mismatch, then higher distinct, one per family
        seen = set()
        uniq = []
        for r in sorted(by[cat], key=lambda x: (x["mismatch"], -x["distinct"])):
            fk = family_key(r["clean"])
            if fk in seen:
                continue
            seen.add(fk)
            uniq.append(r)
        by[cat] = uniq

    # round-robin across categories for diversity
    order = ["lishu", "weibei", "xingkai", "caoshu", "display", "cartoon", "shouxie", "zhuan"]
    picked: list[dict] = []
    idx = {c: 0 for c in order}
    while len(picked) < total:
        progressed = False
        for c in order:
            bucket = by.get(c) or []
            i = idx[c]
            if i >= len(bucket) or i >= per_cat:
                continue
            picked.append(bucket[i])
            idx[c] = i + 1
            progressed = True
            if len(picked) >= total:
                break
        if not progressed:
            break
    return picked


def write_html(picked: list[dict], all_ok: list[dict], stats: dict) -> None:
    cards = []
    for i, r in enumerate(picked, 1):
        rel = f"thumbs/{r['clean']}.png"
        cards.append(
            f"""<div class="card" data-cat="{r['cat']}">
  <div class="meta"><b>{i}. {r['disp']}</b> · <code>{r['clean']}</code> · {r['cat']}</div>
  <div class="nums">mismatch={r['mismatch']:.3f} · cn_fill={r['cn_fill']:.3f} · lat_fill={r['lat_fill']:.3f} · distinct={r['distinct']:.3f}</div>
  <img src="{rel}" alt="{r['clean']}">
</div>"""
        )
    all_ok_json = json.dumps(
        [
            {
                "disp": r["disp"],
                "clean": r["clean"],
                "cat": r["cat"],
                "mismatch": round(r["mismatch"], 3),
                "distinct": round(r["distinct"], 3),
            }
            for r in all_ok
        ],
        ensure_ascii=False,
        indent=2,
    )
    cat_chips = " ".join(f"<span class=chip>{k}:{v}</span>" for k, v in stats["by_cat"].items())
    html = f"""<!doctype html>
<meta charset=utf-8>
<title>Paper font screen · font_files_0914</title>
<style>
body{{font:14px/1.45 ui-sans-serif,system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
a{{color:#9cf}} .note{{color:#aaa;max-width:980px}}
.card{{margin:18px 0;padding:12px;border:1px solid #333;background:#161616}}
.meta{{margin-bottom:4px}} .nums{{color:#9ab;font-size:12px;margin-bottom:8px}}
img{{image-rendering:pixelated;background:#222;max-width:100%}}
.chip{{display:inline-block;margin:2px 6px 2px 0;padding:2px 8px;border:1px solid #456;border-radius:4px;background:#1a2430}}
</style>
<h1>论文展示字体筛选 · font_files_0914</h1>
<p class="note">生成 {stats['generated_at']} · 池 {stats['n_pool']} · 风格关键词命中 {stats['n_keyword']} ·
cmap+渲染通过 {stats['n_ok']} · 已用数据集排除 {stats['n_excluded_known']} ·
<strong>mismatch 低 = 中西墨量更接近（仅代理，非风格一致结论）</strong>。
请眼看：CN 与 Latin 是否同一套设计语言；选风格反差大、笔画清晰、适合出图的。</p>
<p>类别：{cat_chips}</p>
<p class="note">合作者入口：<a href="/reports/paper_fonts_0914_screen/">/reports/paper_fonts_0914_screen/</a>
或 <a href="http://172.19.45.13:19000/reports/paper_fonts_0914_screen/">http://172.19.45.13:19000/reports/paper_fonts_0914_screen/</a></p>
{''.join(cards)}
<details><summary>全部通过候选 ({len(all_ok)})</summary>
<pre>{all_ok_json}</pre>
</details>
"""
    (OUT / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "thumbs").mkdir(exist_ok=True)
    rows = parse_manifest()
    keyword = []
    excluded_known = 0
    for r in rows:
        cat = categorize(r["disp"])
        if not cat:
            continue
        # prefer JW; skip pure JF if JW sibling likely later — still keep all for now
        if r["clean"] in KNOWN or family_key(r["clean"]) in {family_key(k) for k in KNOWN}:
            # soft: exact stem or family already in dataset
            if r["clean"] in KNOWN:
                excluded_known += 1
                continue
        r["cat"] = cat
        keyword.append(r)

    # prefer JW over JF/FW duplicates in same family+cat
    fam_best: dict[tuple[str, str], dict] = {}
    for r in keyword:
        key = (family_key(r["clean"]), r["cat"])
        score = (0 if r["clean"].endswith("JW") else 1, len(r["clean"]))
        prev = fam_best.get(key)
        if prev is None or score < (0 if prev["clean"].endswith("JW") else 1, len(prev["clean"])):
            fam_best[key] = r
    candidates = list(fam_best.values())
    # cap extreme categories that explode (shouxie)
    by_cat = defaultdict(list)
    for r in candidates:
        by_cat[r["cat"]].append(r)
    capped = []
    for cat, items in by_cat.items():
        # keep up to 40 per cat before expensive render
        capped.extend(items[:40] if cat == "shouxie" else items[:30])
    candidates = capped

    print(f"candidates to score: {len(candidates)}", flush=True)
    ok_rows = []
    fail: Counter[str] = Counter()
    for i, rec in enumerate(candidates):
        res = score_candidate(rec)
        if not res:
            fail["null"] += 1
            continue
        if not res.get("ok"):
            fail[res.get("reason", "fail")] += 1
            continue
        sheet = res.pop("sheet")
        thumb = OUT / "thumbs" / f"{res['clean']}.png"
        sheet.save(thumb)
        ok_rows.append(res)
        if (i + 1) % 20 == 0:
            print(f"  scored {i+1}/{len(candidates)} ok={len(ok_rows)}", flush=True)

    ok_rows.sort(key=lambda r: (r["mismatch"], -r["distinct"]))
    picked = pick_diverse(ok_rows, per_cat=4, total=24)
    for r in picked:
        r["picked"] = True

    by_cat_counts = {k: sum(1 for r in ok_rows if r["cat"] == k) for k in sorted({r['cat'] for r in ok_rows})}
    stats = {
        "generated_at": utc_now(),
        "n_pool": len(rows),
        "n_keyword": len(keyword),
        "n_candidates_scored": len(candidates),
        "n_ok": len(ok_rows),
        "n_excluded_known": excluded_known,
        "fail": dict(fail),
        "by_cat": by_cat_counts,
    }
    write_html(picked, ok_rows, stats)
    payload = {
        "stats": stats,
        "shortlist": [
            {
                "rank": i + 1,
                "disp": r["disp"],
                "clean": r["clean"],
                "path": r["path"],
                "cat": r["cat"],
                "mismatch": r["mismatch"],
                "distinct": r["distinct"],
                "cn_fill": r["cn_fill"],
                "lat_fill": r["lat_fill"],
                "miss_lat": r.get("miss_lat"),
            }
            for i, r in enumerate(picked)
        ],
        "all_ok": [
            {
                "disp": r["disp"],
                "clean": r["clean"],
                "path": r["path"],
                "cat": r["cat"],
                "mismatch": r["mismatch"],
                "distinct": r["distinct"],
            }
            for r in ok_rows
        ],
    }
    (OUT / "shortlist.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("shortlist:")
    for i, r in enumerate(picked, 1):
        print(f"  {i:2d}. [{r['cat']}] {r['disp']}  {r['clean']}  mm={r['mismatch']:.3f}")
    print(f"wrote {OUT}/index.html")


if __name__ == "__main__":
    main()
