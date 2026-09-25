#!/usr/bin/env python3
"""Protocol-A dataset for suiti ∪ font_50 keep-649.

Writes ONLY to a new directory. Refuses to touch the F0–F3 A disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/root/projects/hrfont")
OLD_A = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
OUT = ROOT / "data/fontdiffuser-p649-t295-s338-cn2west-v2a-r0"
REVIEW = ROOT / "data/p649_v2a_review"
SUITI = Path("/root/projects/font_crosslingual/data/suiti_fonts_probe/随体字体集")
FONT50 = Path("/root/data/font_50")
CHARSET = ROOT / "manifests/charset_cn2west_v2_planned.json"
SPLIT_DIR = ROOT / "manifests"
NOTO = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

sys_path_note = ROOT / "scripts"
import sys

sys.path.insert(0, str(sys_path_note))
from build_cn2west_v2_proto_abc import (  # noqa: E402
    CANVAS_96,
    INNER_AB,
    cp_name,
    file_name,
    find_size_A,
    ink_yong,
    render_glyph_AB,
    style_chars,
    target_chars,
)

DROP_STEMS = {"FZGoolongWuxiaFontR", "FZXianZTJW"}
ABS_MEAN_BBOX = 0.20
T_INK = 250
PROBES = list("永和书AaGgWwMm01ilㄅあア")


def script_probes() -> list[dict]:
    t = json.loads(CHARSET.read_text(encoding="utf-8"))["target"]
    return [
        {"id": "style_han", "label": "Style 汉字", "n": 338, "sample": list("永和书风骨韵天地")},
        {"id": "ascii_digits", "label": "数字", "n": 10, "sample": list(t["ascii_digits"])},
        {"id": "ascii_letters", "label": "拉丁", "n": 52, "sample": list("AaGgWwMmIiLlQq")},
        {"id": "latin_ext_letters", "label": "扩展拉丁", "n": 27, "sample": list(t["latin_ext_letters"])},
        {"id": "hiragana", "label": "平假名", "n": 83, "sample": list("あいうえおかがきぎぱぽん")},
        {"id": "katakana", "label": "片假名", "n": 86, "sample": list("アイウエオカガキギジヴン")},
        {"id": "bopomofo", "label": "注音", "n": 37, "sample": list(t["bopomofo"])},
    ]


def glyph_folder(ch: str) -> str:
    """338 style = Han; 295 target = Latin / kana / bopomofo. Never mix."""
    return "StyleImage" if "\u4e00" <= ch <= "\u9fff" else "TargetImage"


def assert_not_old(path: Path) -> None:
    p = path.resolve()
    old = OLD_A.resolve()
    if p == old or old in p.parents:
        raise SystemExit(f"refusing to write under old A disk: {p}")


def family_key(name: str) -> str:
    s = Path(name).stem
    s = re.sub(r"(JF|JW)$", "", s)
    for _ in range(3):
        s = re.sub(
            r"[-_](L|R|B|M|H|EB|SB|DB|EL|Te|Da|Cu|Zhong|Zhun|Xi|Italic)$",
            "",
            s,
            flags=re.I,
        )
    return s


def style_key(name: str) -> str:
    """Group weights of the same design. Keep JW/JF; strip _508R / -EL / _Te."""
    s = Path(name).stem
    s = re.sub(r"_\d{2,4}[A-Za-z]{1,3}$", "", s)
    for _ in range(4):
        s = re.sub(
            r"[-_](L|R|B|M|H|T|U|EL|UL|UB|EB|SB|DB|Te|Da|Cu|Zhong|Zhun|Xi|Italic|Regular|Bold|Light|Medium|Heavy)$",
            "",
            s,
            flags=re.I,
        )
    return s


def load_lines(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def sha256_file(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        if limit:
            h.update(f.read(limit))
        else:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def list_ttfs(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in root.iterdir():
        if p.suffix.lower() in {".ttf", ".otf", ".ttc"}:
            out[p.stem] = p
    return out


def build_inventory() -> dict:
    suiti = list_ttfs(SUITI)
    f50 = list_ttfs(FONT50)
    uniq: dict[str, dict] = {}
    for stem, p in suiti.items():
        uniq[stem] = {"stem": stem, "ttf": str(p), "source": "suiti"}
    f50_only = []
    overlap = []
    for stem, p in f50.items():
        if stem in uniq:
            overlap.append(stem)
            continue
        uniq[stem] = {"stem": stem, "ttf": str(p), "source": "font50_only"}
        f50_only.append(stem)

    names = set(uniq)
    jf_drop = []
    for n in list(names):
        if "JF" in n:
            alt = n.replace("JF", "JW")
            if alt in names:
                jf_drop.append(n)
    keep = []
    dropped = []
    for stem, rec in sorted(uniq.items()):
        reason = None
        if stem in DROP_STEMS:
            reason = "drop_incomplete_or_ink_gate"
        elif stem in jf_drop:
            reason = "jf_has_jw"
        if reason:
            dropped.append({**rec, "reason": reason})
            continue
        rec = dict(rec)
        rec["family_key"] = family_key(stem)
        rec["ttf_sha256"] = sha256_file(Path(rec["ttf"]))
        keep.append(rec)

    tr = set(load_lines(SPLIT_DIR / "pipeline_v3_train_stems.txt"))
    va = set(load_lines(SPLIT_DIR / "pipeline_v3_val_stems.txt"))
    te = set(load_lines(SPLIT_DIR / "pipeline_v3_test_stems.txt"))
    overlap260 = sorted(tr | va | te)
    fam_split: dict[str, str] = {}
    for stem, sp in [(s, "train") for s in tr] + [(s, "val") for s in va] + [(s, "test") for s in te]:
        fam_split[family_key(stem)] = sp

    by_split = {"train": [], "val": [], "test": []}
    for rec in keep:
        stem = rec["stem"]
        if stem in tr:
            rec["split"] = "train"
            rec["subset"] = "overlap260"
        elif stem in va:
            rec["split"] = "val"
            rec["subset"] = "overlap260"
        elif stem in te:
            rec["split"] = "test"
            rec["subset"] = "overlap260"
        elif rec["family_key"] in fam_split:
            rec["split"] = fam_split[rec["family_key"]]
            rec["subset"] = "new389_family_follow"
        else:
            rec["split"] = "train"
            rec["subset"] = "new389"
        by_split[rec["split"]].append(stem)

    inv = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "old_a": str(OLD_A),
        "out": str(OUT),
        "n_unique_pool": len(uniq),
        "n_keep": len(keep),
        "n_overlap260": len(overlap260),
        "n_new": len(keep) - len(overlap260),
        "split_counts": {k: len(v) for k, v in by_split.items()},
        "n_jf_drop": len(jf_drop),
        "n_f50_only": len(f50_only),
        "n_overlap_files": len(overlap),
        "fonts": keep,
        "dropped": dropped,
        "overlap260": overlap260,
    }
    return inv


def save_inventory(inv: dict) -> Path:
    assert_not_old(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "inventory.json"
    path.write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def snapshot_old_a() -> dict:
    pngs = list(OLD_A.rglob("*.png"))
    newest = max((p.stat().st_mtime for p in pngs), default=0.0)
    return {
        "n_png": len(pngs),
        "newest_mtime": newest,
        "newest_iso": datetime.fromtimestamp(newest, tz=timezone.utc).isoformat() if newest else None,
        "du_bytes": sum(p.stat().st_size for p in pngs),
    }


def render_content(out_root: Path) -> dict:
    assert_not_old(out_root)
    targets = target_chars()
    fs = find_size_A(str(NOTO), targets)
    tmp = out_root / "_content_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for ch in targets:
        render_glyph_AB(str(NOTO), ch, fs).save(tmp / f"{cp_name(ch)}.png")
    for split in ("train", "val", "test"):
        dst = out_root / split / "ContentImage"
        dst.mkdir(parents=True, exist_ok=True)
        for p in tmp.glob("*.png"):
            shutil.copy2(p, dst / p.name)
    shutil.rmtree(tmp)
    return {"font": str(NOTO), "n": len(targets), "size": fs}


def _render_font_job(job: dict) -> dict:
    stem = job["stem"]
    ttf = job["ttf"]
    split = job["split"]
    out_root = Path(job["out_root"])
    targets = job["targets"]
    styles = job["styles"]
    tdir = out_root / split / "TargetImage" / stem
    sdir = out_root / split / "StyleImage" / stem
    tdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)
    all_chars = targets + styles
    fs = find_size_A(ttf, all_chars)
    n_t = n_s = 0
    for ch in targets:
        render_glyph_AB(ttf, ch, fs).save(tdir / file_name(stem, ch))
        n_t += 1
    for ch in styles:
        render_glyph_AB(ttf, ch, fs).save(sdir / file_name(stem, ch))
        n_s += 1
    yong_p = sdir / file_name(stem, "永")
    yong = ink_yong(yong_p) if yong_p.exists() else {"h": 0, "w": 0, "area": 0, "touch": False}
    font_obj = ImageFont.truetype(ttf, fs)
    tmp = Image.new("L", (1, 1))
    dr = ImageDraw.Draw(tmp)
    max_w = max_h = 0
    wide_ch = tall_ch = ""
    for ch in all_chars:
        bb = dr.textbbox((0, 0), ch, font=font_obj)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if w > max_w:
            max_w, wide_ch = w, ch
        if h > max_h:
            max_h, tall_ch = h, ch
    return {
        "stem": stem,
        "split": split,
        "ttf": ttf,
        "subset": job.get("subset"),
        "n_target": n_t,
        "n_style": n_s,
        "size": fs,
        "max_w": max_w,
        "max_h": max_h,
        "wide": wide_ch,
        "tall": tall_ch,
        "yong": yong,
    }


def render_subset(inv: dict, subset: str, workers: int) -> list[dict]:
    assert_not_old(OUT)
    targets = target_chars()
    styles = style_chars()
    if subset == "overlap260":
        fonts = [f for f in inv["fonts"] if f["subset"] == "overlap260"]
    elif subset == "new389":
        fonts = [f for f in inv["fonts"] if f["subset"] != "overlap260"]
    else:
        fonts = list(inv["fonts"])
    jobs = [
        {
            "stem": f["stem"],
            "ttf": f["ttf"],
            "split": f["split"],
            "subset": f["subset"],
            "out_root": str(OUT),
            "targets": targets,
            "styles": styles,
        }
        for f in fonts
    ]
    print(f"[render] subset={subset} fonts={len(jobs)} workers={workers} -> {OUT}", flush=True)
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_render_font_job, j): j["stem"] for j in jobs}
        done = 0
        for fut in as_completed(futs):
            done += 1
            rows.append(fut.result())
            if done % 20 == 0 or done == len(jobs) or done <= 3:
                print(f"  {subset} {done}/{len(jobs)} elapsed={time.time()-t0:.0f}s", flush=True)
    rows.sort(key=lambda r: (r["split"], r["stem"]))
    return rows


def l1_pair(a: Path, b: Path) -> float | None:
    if not a.exists() or not b.exists():
        return None
    xa = np.asarray(Image.open(a).convert("L"), dtype=np.float32) / 255.0
    xb = np.asarray(Image.open(b).convert("L"), dtype=np.float32) / 255.0
    if xa.shape != xb.shape:
        return 1.0
    return float(np.abs(xa - xb).mean())


def find_old_font_dir(stem: str, kind: str) -> Path | None:
    for split in ("train", "val", "test"):
        d = OLD_A / split / kind / stem
        if d.is_dir():
            return d
    return None


def _compare_one_font(rec: dict) -> dict:
    stem = rec["stem"]
    split = rec["split"]
    old_t = find_old_font_dir(stem, "TargetImage")
    old_s = find_old_font_dir(stem, "StyleImage")
    new_t = OUT / split / "TargetImage" / stem
    new_s = OUT / split / "StyleImage" / stem
    diffs = []
    identical = 0
    missing = 0
    for kind, old_d, new_d in (("TargetImage", old_t, new_t), ("StyleImage", old_s, new_s)):
        if old_d is None or not old_d.is_dir():
            missing += 1
            continue
        for p in new_d.glob("*.png"):
            v = l1_pair(old_d / p.name, p)
            if v is None:
                missing += 1
                continue
            if v == 0.0:
                identical += 1
            else:
                diffs.append((v, kind, p.name))
    diffs.sort(reverse=True)
    return {
        "stem": stem,
        "split": split,
        "n_identical": identical,
        "n_diff": len(diffs),
        "n_missing": missing,
        "mean_l1_diff_only": float(np.mean([d[0] for d in diffs])) if diffs else 0.0,
        "max_l1": max((d[0] for d in diffs), default=0.0),
        "top": diffs[:5],
    }


def compare260(inv: dict, rows: list[dict], workers: int = 24) -> dict:
    REVIEW.mkdir(parents=True, exist_ok=True)
    overlap = [f for f in inv["fonts"] if f["subset"] == "overlap260"]
    print(f"[compare260] fonts={len(overlap)} workers={workers}", flush=True)
    per_font = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_compare_one_font, rec) for rec in overlap]
        done = 0
        for fut in as_completed(futs):
            per_font.append(fut.result())
            done += 1
            if done % 40 == 0 or done == len(futs):
                print(f"  compare {done}/{len(futs)}", flush=True)
    n_ident = sum(f["n_identical"] for f in per_font)
    n_diff = sum(f["n_diff"] for f in per_font)
    n_miss = sum(f["n_missing"] for f in per_font)
    worst = [
        {"stem": f["stem"], "split": f["split"], "top": f.get("top") or []}
        for f in sorted(per_font, key=lambda r: (-r["n_diff"], -r["max_l1"]))
        if f["n_diff"] > 0
    ][:20]

    per_font.sort(key=lambda r: (-r["n_diff"], -r["max_l1"], r["stem"]))
    probes = list("永Agん0书Wㄅ")
    sheet = build_compare_sheet(pick_compare_fonts(per_font), probes)
    summary = {
        "n_fonts": len(overlap),
        "png_identical": n_ident,
        "png_diff": n_diff,
        "png_missing_old_or_new": n_miss,
        "fonts_all_identical": sum(1 for f in per_font if f["n_diff"] == 0 and f["n_missing"] == 0),
        "fonts_any_diff": sum(1 for f in per_font if f["n_diff"] > 0),
        "sheet": str(sheet) if sheet else None,
        "per_font": per_font,
        "worst": worst[:20],
    }
    (REVIEW / "compare260.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_compare_html(summary, probes)
    return summary


def pick_compare_fonts(per_font: list[dict]) -> list[dict]:
    diffs = [f for f in per_font if f["n_diff"] > 0][:6]
    same = [f for f in per_font if f["n_diff"] == 0][:6]
    seen = {f["stem"] for f in diffs + same}
    # always include a few named demo fonts if present
    by = {f["stem"]: f for f in per_font}
    for stem in ("FZChuangHJW_DB", "FZLingFKSJW-B", "FZWuYTJW", "FZBaoHTJW_Da"):
        if stem in by and stem not in seen:
            same.append(by[stem])
            seen.add(stem)
    return (diffs + same)[:12]


def build_compare_sheet(fonts: list[dict], probes: list[str]) -> Path | None:
    if not fonts:
        return None
    cell, lab_w, head = 96, 72, 36
    cols = 1 + 2 * len(probes)
    rows = len(fonts)
    W = lab_w + cols * (cell + 4)
    # actually: label + for each probe: old|new
    W = lab_w + len(probes) * (2 * cell + 8) + 16
    H = head + rows * (cell + 8) + 16
    canvas = Image.new("RGB", (W, H), (238, 241, 245))
    dr = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    dr.text((8, 10), "old A  |  new p649   (same protocol A)", fill=(30, 30, 30), font=font)
    x0 = lab_w
    for i, ch in enumerate(probes):
        x = x0 + i * (2 * cell + 8)
        dr.text((x + 8, 18), f"{ch} old", fill=(80, 80, 80), font=font)
        dr.text((x + cell + 8, 18), f"{ch} new", fill=(80, 80, 80), font=font)
    for r, rec in enumerate(fonts):
        stem, split = rec["stem"], rec["split"]
        y = head + r * (cell + 8)
        dr.text((4, y + 40), stem[:14], fill=(20, 20, 20), font=font)
        for i, ch in enumerate(probes):
            x = x0 + i * (2 * cell + 8)
            cp = cp_name(ch)
            folder = glyph_folder(ch)
            old_d = find_old_font_dir(stem, folder)
            new_p = OUT / split / folder / stem / f"{stem}+{cp}.png"
            old_p = (old_d / f"{stem}+{cp}.png") if old_d else None
            for j, p in enumerate((old_p, new_p)):
                box = (x + j * cell, y)
                if p and Path(p).exists():
                    im = Image.open(p).convert("RGB").resize((cell, cell), Image.NEAREST)
                    canvas.paste(im, box)
                dr.rectangle((box[0], box[1], box[0] + cell - 1, box[1] + cell - 1), outline=(200, 204, 210))
    dest = REVIEW / "compare260_sheet.png"
    canvas.save(dest)
    return dest


def write_compare_html(summary: dict, probes: list[str]) -> None:
    rows = []
    for f in summary["per_font"]:
        tone = "same" if f["n_diff"] == 0 and f["n_missing"] == 0 else "diff"
        rows.append(
            f'<tr class="{tone}"><td>{f["stem"]}</td><td>{f["split"]}</td>'
            f'<td>{f["n_identical"]}</td><td>{f["n_diff"]}</td><td>{f["n_missing"]}</td>'
            f'<td>{f["max_l1"]:.6f}</td></tr>'
        )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>p649 vs 原260 像素对照</title>
<style>
body{{font:14px/1.45 system-ui,sans-serif;margin:24px;background:#eef1f5;color:#1a1a1a}}
table{{border-collapse:collapse;background:#fff}}
th,td{{border:1px solid #d5dbe3;padding:4px 8px;font-size:12px}}
tr.diff{{background:#fff4e5}} tr.same{{background:#f3faf3}}
img.sheet{{max-width:100%;background:#fff;border:1px solid #d5dbe3}}
.note{{color:#5c6570}}
</style></head><body>
<h1>新渲 260 vs 原 A 盘</h1>
<p>原盘只读 <code>{OLD_A}</code> · 新盘 <code>{OUT}</code></p>
<p>逐像素 L1（灰度/255）。identical={summary["png_identical"]} · diff={summary["png_diff"]} ·
全同字体 {summary["fonts_all_identical"]}/{summary["n_fonts"]} · 有差异字体 {summary["fonts_any_diff"]}</p>
<p class="note">文件压缩选项可能不同，这里比的是解码后的像素，不是文件 hash。</p>
<p><img class="sheet" src="compare260_sheet.png" alt="probe sheet"/></p>
<table>
<tr><th>stem</th><th>split</th><th>identical</th><th>diff</th><th>missing</th><th>max L1</th></tr>
{''.join(rows)}
</table>
</body></html>
"""
    (REVIEW / "compare260.html").write_text(html, encoding="utf-8")


def scan_font_ink(job: dict) -> dict:
    stem, split = job["stem"], job["split"]
    root = Path(job["root"])
    bboxes: list[float] = []
    fills: list[float] = []
    n_empty = 0
    touch_edge = False
    hashes: dict[str, int] = defaultdict(int)
    for sub in ("TargetImage", "StyleImage"):
        d = root / split / sub / stem
        if not d.exists():
            continue
        for p in d.glob("*.png"):
            try:
                a = np.asarray(Image.open(p).convert("L"))
            except Exception:
                continue
            h = hashlib.sha1(a.tobytes()).hexdigest()
            hashes[h] += 1
            ys, xs = np.where(a < T_INK)
            if len(ys) == 0:
                n_empty += 1
                bboxes.append(0.0)
                fills.append(0.0)
                continue
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            bboxes.append(float((y1 - y0 + 1) * (x1 - x0 + 1) / (CANVAS_96 * CANVAS_96)))
            fills.append(float(np.count_nonzero(a < T_INK)) / (CANVAS_96 * CANVAS_96))
            if y0 <= 0 or x0 <= 0 or y1 >= CANVAS_96 - 1 or x1 >= CANVAS_96 - 1:
                touch_edge = True
    arr_b = np.array(bboxes, dtype=np.float64) if bboxes else np.array([0.0])
    arr_f = np.array(fills, dtype=np.float64) if fills else np.array([0.0])
    n_tofu = sum(1 for n in hashes.values() if n >= 8)
    return {
        "stem": stem,
        "split": split,
        "n": len(bboxes),
        "n_empty": n_empty,
        "n_tofu_groups": n_tofu,
        "mean_ink_ratio": float(arr_b.mean()),
        "ink_ratio_p5": float(np.percentile(arr_b, 5)),
        "ink_ratio_p50": float(np.percentile(arr_b, 50)),
        "ink_ratio_p95": float(np.percentile(arr_b, 95)),
        "mean_ink_pixel_ratio": float(arr_f.mean()),
        "touch_edge": touch_edge,
    }


def build_review(inv: dict, font_rows: list[dict], workers: int) -> dict:
    assert_not_old(REVIEW)
    REVIEW.mkdir(parents=True, exist_ok=True)
    meta = {r["stem"]: r for r in font_rows}
    jobs = [{"stem": f["stem"], "split": f["split"], "root": str(OUT)} for f in inv["fonts"]]
    scans = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(scan_font_ink, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            scans.append(fut.result())
            if i % 50 == 0 or i == len(futs):
                print(f"  ink scan {i}/{len(futs)}", flush=True)
    rows = []
    for sc in scans:
        am = meta.get(sc["stem"], {})
        mean_r = sc["mean_ink_ratio"]
        flags = []
        if mean_r < ABS_MEAN_BBOX:
            flags.append(f"mean_bbox<{ABS_MEAN_BBOX}")
        if sc["touch_edge"]:
            flags.append("touch_edge")
        if sc["n_empty"] > 0:
            flags.append("n_empty>0")
        if sc["n_tofu_groups"] > 0:
            flags.append("tofu_hash")
        suggested = ""
        if mean_r < ABS_MEAN_BBOX:
            suggested = "drop"
        elif sc["touch_edge"] or sc["n_empty"] > 0 or sc["n_tofu_groups"] > 0:
            suggested = "rerender"
        rows.append({
            "stem": sc["stem"],
            "split": sc["split"],
            "subset": next((f["subset"] for f in inv["fonts"] if f["stem"] == sc["stem"]), ""),
            "family_key": next((f.get("family_key") for f in inv["fonts"] if f["stem"] == sc["stem"]), family_key(sc["stem"])),
            "style_key": style_key(sc["stem"]),
            "mean_ink_ratio": mean_r,
            "ink_ratio_p5": sc["ink_ratio_p5"],
            "ink_ratio_p50": sc["ink_ratio_p50"],
            "ink_ratio_p95": sc["ink_ratio_p95"],
            "mean_ink_pixel_ratio": sc["mean_ink_pixel_ratio"],
            "fs_A": am.get("size"),
            "tall_ch": am.get("tall"),
            "max_h": am.get("max_h"),
            "wide_ch": am.get("wide"),
            "max_w": am.get("max_w"),
            "touch_edge": sc["touch_edge"],
            "n_empty": sc["n_empty"],
            "n_tofu_groups": sc["n_tofu_groups"],
            "n_glyphs": sc["n"],
            "auto_flags": flags,
            "suggested_action": suggested,
            "review_candidate": bool(flags),
            "review_decision": "",
        })
    rows.sort(key=lambda r: (r["mean_ink_ratio"], r["stem"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    payload = {
        "dataset": OUT.name,
        "dataset_path": str(OUT),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ink_threshold": T_INK,
        "primary_metric": "mean(ink_bbox_area / 96^2)",
        "draft_thresholds": {"absolute_mean_bbox": ABS_MEAN_BBOX},
        "summary": {
            "n_fonts": len(rows),
            "n_review_candidates": sum(1 for r in rows if r["review_candidate"]),
            "n_mean_bbox_lt_020": sum(1 for r in rows if r["mean_ink_ratio"] < ABS_MEAN_BBOX),
            "n_empty": sum(1 for r in rows if r["n_empty"] > 0),
            "n_tofu": sum(1 for r in rows if r["n_tofu_groups"] > 0),
            "n_touch": sum(1 for r in rows if r["touch_edge"]),
        },
        "fonts": rows,
    }
    (REVIEW / "ink_ratio_rank.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    write_review_pages()
    return payload["summary"]


def write_review_pages() -> None:
    annotate_review_json()
    tpl = Path(__file__).with_name("p649_review_template.html").read_text(encoding="utf-8")
    html = (
        tpl.replace("SCRIPT_PROBES_PLACEHOLDER", json.dumps(script_probes(), ensure_ascii=False))
        .replace("PROBES_PLACEHOLDER", json.dumps(PROBES, ensure_ascii=False))
        .replace("REL_PLACEHOLDER", f"../{OUT.name}")
    )
    REVIEW.mkdir(parents=True, exist_ok=True)
    (REVIEW / "review.html").write_text(html, encoding="utf-8")
    index = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"/><title>p649 review hub</title></head>
<body style="font:14px/1.5 system-ui;margin:32px">
<h1>p649 协议 A 审查入口</h1>
<p>新盘只读对照：原 A 未覆盖。</p>
<ul>
<li><a href="compare260.html">260 套 vs 原 A 像素对照</a></li>
<li><a href="review.html">ink / 空图 / tofu 人工审查</a></li>
</ul>
<p>启动：<code>python -m http.server 8778 --directory {REVIEW.parent}</code>
然后打开 <code>http://127.0.0.1:8778/p649_v2a_review/review.html</code></p>
<p>进度会写入 <code>p649_v2a_review/decisions.json</code>。请用 <code>python scripts/serve_p649_review.py --port 8778</code>，不要用普通 http.server（不能写盘）。</p>
</body></html>
"""
    (REVIEW / "index.html").write_text(index, encoding="utf-8")


def annotate_review_json() -> None:
    path = REVIEW / "ink_ratio_rank.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    by = {}
    inv_path = OUT / "inventory.json"
    if inv_path.is_file():
        inv = json.loads(inv_path.read_text(encoding="utf-8"))
        by = {f["stem"]: f for f in inv.get("fonts", [])}
    for r in payload.get("fonts", []):
        rec = by.get(r["stem"], {})
        r["family_key"] = rec.get("family_key") or family_key(r["stem"])
        r["style_key"] = style_key(r["stem"])
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def check_export(jsonl_path: Path) -> dict:
    """Verify a drop/decisions JSONL maps to real Target+Style PNG dirs."""
    if not jsonl_path.is_file():
        raise SystemExit(f"missing jsonl: {jsonl_path}")
    recs = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    if not recs:
        raise SystemExit("empty jsonl")
    drops = [r for r in recs if r.get("review_decision", "drop") == "drop"]
    if not drops:
        drops = recs  # allow a stems-only / files-only dump
    by_stem = {}
    if (OUT / "inventory.json").exists():
        inv = json.loads((OUT / "inventory.json").read_text(encoding="utf-8"))
        by_stem = {f["stem"]: f for f in inv["fonts"]}
    rows = []
    n_ok = n_bad = 0
    for rec in drops:
        stem = rec.get("stem")
        split = rec.get("split") or (by_stem.get(stem) or {}).get("split")
        if not stem or not split:
            rows.append({"stem": stem, "ok": False, "error": "missing stem/split"})
            n_bad += 1
            continue
        target = OUT / split / "TargetImage" / stem
        style = OUT / split / "StyleImage" / stem
        nt = len(list(target.glob("*.png"))) if target.is_dir() else 0
        ns = len(list(style.glob("*.png"))) if style.is_dir() else 0
        expect_t, expect_s = 295, 338
        ok = target.is_dir() and style.is_dir() and nt == expect_t and ns == expect_s
        rows.append({
            "stem": stem,
            "split": split,
            "target_dir": str(target),
            "style_dir": str(style),
            "n_target_png": nt,
            "n_style_png": ns,
            "expect_target": 295,
            "expect_style": 338,
            "ok": ok,
        })
        n_ok += int(ok)
        n_bad += int(not ok)
    summary = {
        "jsonl": str(jsonl_path),
        "n_drop": len(drops),
        "n_ok": n_ok,
        "n_bad": n_bad,
        "rows": rows,
    }
    REVIEW.mkdir(parents=True, exist_ok=True)
    (REVIEW / "last_export_check.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("jsonl", "n_drop", "n_ok", "n_bad")}, ensure_ascii=False, indent=2))
    if n_bad:
        raise SystemExit(f"check-export failed: {n_bad} bad / {len(drops)}")
    return summary


def write_summary(inv: dict, content: dict, rows: list[dict], snap_before: dict, snap_after: dict) -> None:
    summary = {
        "dataset_id": OUT.name,
        "protocol": "A",
        "old_a_readonly": str(OLD_A),
        "n_fonts": len(inv["fonts"]),
        "split_counts": inv["split_counts"],
        "content": content,
        "fonts": rows,
        "old_a_snapshot_before": snap_before,
        "old_a_snapshot_after": snap_after,
        "old_a_unchanged": snap_before == snap_after,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "inventory", "render260", "compare", "render389", "review", "check-export"])
    ap.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 8))
    ap.add_argument("--jsonl", type=Path, default=None)
    args = ap.parse_args()
    assert_not_old(OUT)
    assert_not_old(REVIEW)

    if args.cmd == "check-export":
        if args.jsonl is None:
            raise SystemExit("check-export requires --jsonl")
        check_export(args.jsonl)
        return

    if args.cmd == "inventory":
        inv = build_inventory()
        save_inventory(inv)
        print(json.dumps({k: inv[k] for k in inv if k not in {"fonts", "dropped", "overlap260"}}, ensure_ascii=False, indent=2))
        return

    snap0 = snapshot_old_a()
    inv_path = OUT / "inventory.json"
    if inv_path.exists() and args.cmd != "inventory":
        inv = json.loads(inv_path.read_text(encoding="utf-8"))
    else:
        inv = build_inventory()
        save_inventory(inv)

    content = None
    rows_all: list[dict] = []
    rows_path = OUT / "render_rows.json"
    if rows_path.exists():
        rows_all = json.loads(rows_path.read_text(encoding="utf-8"))

    def merge_rows(new_rows: list[dict]) -> None:
        by = {r["stem"]: r for r in rows_all}
        for r in new_rows:
            by[r["stem"]] = r
        merged = sorted(by.values(), key=lambda r: (r["split"], r["stem"]))
        rows_path.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
        return merged

    if args.cmd in ("all", "render260"):
        if not (OUT / "train" / "ContentImage").exists():
            content = render_content(OUT)
            (OUT / "content_meta.json").write_text(json.dumps(content, indent=2), encoding="utf-8")
            print("[content]", content, flush=True)
        rows = render_subset(inv, "overlap260", args.workers)
        rows_all = merge_rows(rows)
        print(f"[render260] done n={len(rows)}", flush=True)

    if args.cmd in ("all", "compare"):
        cmp = compare260(inv, rows_all, workers=args.workers)
        print("[compare260]", {k: cmp[k] for k in cmp if k not in {"per_font", "worst"}}, flush=True)

    if args.cmd in ("all", "render389"):
        if not (OUT / "train" / "ContentImage").exists():
            content = render_content(OUT)
            (OUT / "content_meta.json").write_text(json.dumps(content, indent=2), encoding="utf-8")
        rows = render_subset(inv, "new389", args.workers)
        rows_all = merge_rows(rows)
        print(f"[render389] done n={len(rows)}", flush=True)

    if args.cmd in ("all", "review"):
        if not rows_all:
            raise SystemExit("no render_rows.json")
        summ = build_review(inv, rows_all, args.workers)
        print("[review]", summ, flush=True)

    snap1 = snapshot_old_a()
    if content is None and (OUT / "content_meta.json").exists():
        content = json.loads((OUT / "content_meta.json").read_text())
    if rows_all:
        write_summary(inv, content or {}, rows_all, snap0, snap1)
    print("[old_a_unchanged]", snap0 == snap1, flush=True)
    if snap0 != snap1:
        raise SystemExit("OLD A DISK CHANGED — abort")


if __name__ == "__main__":
    main()
