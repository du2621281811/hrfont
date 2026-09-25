#!/usr/bin/env python3
"""Build retrain_v2 GAN/LF assets: P1∪P2 english GT, CN style refs only, Demo-8 holdout."""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/root/projects/hrfont")
FONT_DIR = Path("/root/data/font_50")
OUT = ROOT / "data" / "retrain_v2"
SIZE = 128
CONTENT_LATIN = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
STYLE_REFS = list("永和书风骨韵天地")


def char_fname(ch: str) -> str:
    """Stable filesystem-safe name; A-Z keep FTrans A+.png convention."""
    if "A" <= ch <= "Z":
        return f"{ch}+.png"
    if "a" <= ch <= "z" or ("0" <= ch <= "9"):
        return f"{ch}.png"
    return f"u{ord(ch):04X}.png"


def has_glyph(font: ImageFont.FreeTypeFont, ch: str) -> bool:
    try:
        bbox = font.getmask(ch).getbbox()
        return bbox is not None
    except Exception:
        return False


def render(font_path: Path, ch: str, size: int = SIZE, index: int = 0) -> Image.Image:
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    path = str(font_path)
    lo, hi, best = 8, size, None
    for _ in range(14):
        fs = (lo + hi) // 2
        try:
            f = ImageFont.truetype(path, fs, index=index)
        except OSError:
            f = ImageFont.truetype(path, fs)
        bbox = draw.textbbox((0, 0), ch, font=f)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= size - 8 and h <= size - 8:
            best = (f, bbox)
            lo = fs + 1
        else:
            hi = fs - 1
    if best is None:
        try:
            f = ImageFont.truetype(path, max(10, size // 2), index=index)
        except OSError:
            f = ImageFont.truetype(path, max(10, size // 2))
        bbox = draw.textbbox((0, 0), ch, font=f)
    else:
        f, bbox = best
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - w) // 2 - bbox[0], (size - h) // 2 - bbox[1]), ch, font=f, fill=0)
    return img


def _render_font_job(args):
    phase, lang, font_path, chars, out_root = args
    font_path = Path(font_path)
    out_root = Path(out_root)
    out_dir = out_root / phase / lang / font_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        probe = ImageFont.truetype(str(font_path), 40)
    except Exception as e:
        return {"phase": phase, "lang": lang, "font": font_path.stem, "n": 0, "err": str(e)}
    n = 0
    missing = []
    for ch in chars:
        if not has_glyph(probe, ch):
            missing.append(ch)
            continue
        render(font_path, ch).save(out_dir / char_fname(ch))
        n += 1
    return {
        "phase": phase,
        "lang": lang,
        "font": font_path.stem,
        "n": n,
        "missing": [f"U+{ord(c):04X}" for c in missing],
    }


def main():
    meta = json.loads((ROOT / "data" / "unified_v1" / "meta.json").read_text(encoding="utf-8"))
    demo_stems = {Path(f).stem for f in meta["demo_fonts"]}
    p1 = list(meta["L_p1"])
    p2 = list(meta["L_p2"])
    assert len(p1) == 122 and len(p2) == 42
    english_chars = p1 + p2
    content_cjk = Path(meta.get("content_cjk", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))

    all_fonts = sorted(FONT_DIR.glob("*.ttf")) + sorted(FONT_DIR.glob("*.TTF"))
    # dedupe by stem
    by_stem = {}
    for p in all_fonts:
        by_stem[p.stem] = p
    train_fonts = [by_stem[s] for s in sorted(by_stem) if s not in demo_stems]
    test_fonts = [by_stem[s] for s in sorted(by_stem) if s in demo_stems]
    print(f"train_fonts={len(train_fonts)} test_fonts={len(test_fonts)} english={len(english_chars)}")

    font_root = OUT / "font"
    jobs = []
    for phase, fonts in [("train", train_fonts), ("test_unknown_style", test_fonts)]:
        for fp in fonts:
            jobs.append((phase, "chinese", str(fp), STYLE_REFS, str(font_root)))
            jobs.append((phase, "english", str(fp), english_chars, str(font_root)))

    coverage = {"train": {}, "test_unknown_style": {}, "source": {}, "meta": {
        "size": SIZE,
        "english_chars_n": len(english_chars),
        "p1_n": len(p1),
        "p2_n": len(p2),
        "style_refs": STYLE_REFS,
        "train_fonts": [p.stem for p in train_fonts],
        "test_fonts": [p.stem for p in test_fonts],
    }}

    workers = min(32, max(4, (os.cpu_count() or 8) // 2))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_render_font_job, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            phase = r["phase"]
            coverage[phase].setdefault(r["lang"], {})[r["font"]] = {
                "n": r["n"],
                "missing": r.get("missing", []),
                "err": r.get("err"),
            }
            if i % 20 == 0 or i == len(futs):
                print(f"render progress {i}/{len(futs)}")

    # source content skeletons
    for phase in ("train", "test_unknown_style"):
        src = font_root / phase / "source"
        src.mkdir(parents=True, exist_ok=True)
        for ch in p1:
            render(CONTENT_LATIN, ch).save(src / char_fname(ch))
        for ch in p2:
            render(content_cjk, ch, index=0).save(src / char_fname(ch))
        for ch in STYLE_REFS:
            # CN content fallback (unused in chinese2english)
            if train_fonts:
                render(train_fonts[0], ch).save(src / char_fname(ch))
        coverage["source"][phase] = len(list(src.glob("*.png")))

    # coverage summary
    def summarize(phase):
        eng = coverage[phase].get("english", {})
        ns = [v["n"] for v in eng.values()]
        miss_fonts = {k: v["missing"] for k, v in eng.items() if v.get("missing")}
        return {
            "fonts": len(eng),
            "english_png_total": sum(ns),
            "english_png_mean": (sum(ns) / len(ns)) if ns else 0,
            "fonts_with_missing": len(miss_fonts),
            "missing_by_font": miss_fonts,
        }

    summary = {
        "train": summarize("train"),
        "test_unknown_style": summarize("test_unknown_style"),
        "source": coverage["source"],
        "meta": coverage["meta"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coverage.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "meta.json").write_text(json.dumps(coverage["meta"], ensure_ascii=False, indent=2), encoding="utf-8")

    # LF char lists (P1∪P2 only; no CN→CN aux)
    lf = OUT / "lffont"
    lf.mkdir(parents=True, exist_ok=True)
    train_chars = english_chars
    (lf / "train_chars.json").write_text(json.dumps(train_chars, ensure_ascii=False), encoding="utf-8")
    (lf / "gen_chars.json").write_text(json.dumps(english_chars, ensure_ascii=False), encoding="utf-8")
    (lf / "ref_chars.json").write_text(json.dumps(STYLE_REFS, ensure_ascii=False), encoding="utf-8")
    (lf / "val_chars.json").write_text(json.dumps(p1[:16] + p2[:8], ensure_ascii=False), encoding="utf-8")

    # reuse / extend existing LF decomposition
    ffg = ROOT / "repos" / "fewshot-font-generation"
    dec = json.loads((ffg / "data/chn/decomposition.json").read_text(encoding="utf-8"))
    primals = json.loads((ffg / "data/chn/primals.json").read_text(encoding="utf-8"))
    primals_set = set(primals)
    for ch in english_chars + STYLE_REFS:
        if ch not in primals_set:
            primals.append(ch)
            primals_set.add(ch)
        if ch not in dec:
            dec[ch] = [ch]
    (lf / "decomposition.json").write_text(json.dumps(dec, ensure_ascii=False), encoding="utf-8")
    (lf / "primals.json").write_text(json.dumps(primals, ensure_ascii=False), encoding="utf-8")

    ttf_train, ttf_test = lf / "ttf_train", lf / "ttf_test"
    ttf_train.mkdir(exist_ok=True)
    ttf_test.mkdir(exist_ok=True)
    for p in train_fonts + test_fonts:
        dst = (ttf_test if p.stem in demo_stems else ttf_train) / p.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(p)
    src_link = lf / "source.ttf"
    if src_link.exists() or src_link.is_symlink():
        src_link.unlink()
    src_link.symlink_to(CONTENT_LATIN)
    # mixed-script content: prefer Noto CJK (covers P2; Latin often present)
    src_mixed = lf / "source_mixed.ttf"
    if src_mixed.exists() or src_mixed.is_symlink():
        src_mixed.unlink()
    # ttc cannot always symlink as .ttf for all loaders; keep path file
    (lf / "source_mixed.path").write_text(str(content_cjk), encoding="utf-8")
    if content_cjk.exists():
        src_mixed.symlink_to(content_cjk)

    # LF yaml configs
    cfg = lf / "cfgs"
    cfg.mkdir(exist_ok=True)
    (cfg / "train_data.yaml").write_text(
        f"""dset:
  train:
    data_dir: {lf / 'ttf_train'}
    chars: {lf / 'train_chars.json'}
    extension: ttf
  val:
    unseen_chars:
      data_dir: {lf / 'ttf_test'}
      extension: ttf
      n_gen: 8
      n_font: 2
      chars: {lf / 'val_chars.json'}
      source_path: {lf / 'source.ttf'}
      source_ext: ttf
    seen_chars:
      data_dir: {lf / 'ttf_train'}
      extension: ttf
      n_gen: 8
      n_font: 2
      chars: {lf / 'val_chars.json'}
      source_path: {lf / 'source.ttf'}
      source_ext: ttf
""",
        encoding="utf-8",
    )
    (cfg / "lf_p1_override.yaml").write_text(
        f"""decomposition: {lf / 'decomposition.json'}
primals: {lf / 'primals.json'}
max_iter: 40000
trainer:
  work_dir: {ROOT / 'runs_retrain_v2' / 'lffont' / 'train_p1'}
  print_freq: 200
  val_freq: 5000
  save_freq: 5000
dset:
  loader:
    num_workers: 8
    batch_size: 8
  train:
    source_path: {lf / 'source.ttf'}
    source_ext: ttf
""",
        encoding="utf-8",
    )
    (cfg / "lf_p2_override.yaml").write_text(
        f"""decomposition: {lf / 'decomposition.json'}
primals: {lf / 'primals.json'}
max_iter: 20000
trainer:
  work_dir: {ROOT / 'runs_retrain_v2' / 'lffont' / 'train_p2'}
  print_freq: 200
  val_freq: 5000
  save_freq: 5000
dset:
  loader:
    num_workers: 4
    batch_size: 1
  train:
    source_path: {lf / 'source.ttf'}
    source_ext: ttf
""",
        encoding="utf-8",
    )

    print(json.dumps(summary["train"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["test_unknown_style"], ensure_ascii=False, indent=2))
    print("done", OUT)


if __name__ == "__main__":
    main()
