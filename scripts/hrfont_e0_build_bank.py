#!/usr/bin/env python3
"""HR-Font E0: render 96/256 bank, drop fake Latin, cache encoders, freeze gap.

Demo-8 is rendered for eval only and is never used as an alpha-bank member.
Resumable: existing PNGs are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/root/projects/hrfont")
META = json.loads((ROOT / "data/retrain_v2/meta.json").read_text())
CHARSET_P1 = ROOT / "data/unified_v1/charset_P1_latin.txt"
OUT = ROOT / "data/hrfont/e0_bank"
REP = ROOT / "reports/hrfont_overnight"
REF8 = list("永和书风骨韵天地")
DONORS = list("口回日目田一二十土川人八乙了中申")
B0_TTF = ROOT / "data/FZKTJW.TTF"
DEJAVU = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
TTF_DIRS = [
    ROOT / "data/lffont_cn2latin/ttf_train",
    ROOT / "data/lffont_cn2latin/ttf_test",
    Path("/root/data/font_50"),
]


def log(msg: str) -> None:
    REP.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with (REP / "e0.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def p1_chars() -> list[str]:
    out: list[str] = []
    for raw in CHARSET_P1.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        code = raw.split()[0]
        if not code.startswith("U+"):
            continue
        ch = chr(int(code[2:], 16))
        if (not ch.isprintable()) or ch.isspace():
            continue
        out.append(ch)
    return out


def resolve_ttf(stem: str) -> Path | None:
    for d in TTF_DIRS:
        if not d.exists():
            continue
        for ext in (".ttf", ".TTF", ".otf", ".OTF"):
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
        hits = [h for h in d.glob(stem + ".*") if h.suffix.lower() in {".ttf", ".otf"}]
        if hits:
            return hits[0]
    return None


def cmap_ok(ttf: str, ch: str) -> bool:
    try:
        from fontTools.ttLib import TTFont

        font = TTFont(ttf, lazy=True, fontNumber=0)
        cmap = font.getBestCmap() or {}
        gid = cmap.get(ord(ch))
        font.close()
        return gid not in (None, ".notdef", ".null", "NULL")
    except Exception:
        return True


def render_fit(ttf: Path | str, ch: str, canvas: int) -> Image.Image | None:
    target_margin = canvas * 0.08
    best, best_score = None, 1e9
    for size in range(int(canvas * 0.50), int(canvas * 0.92), 2):
        try:
            font = ImageFont.truetype(str(ttf), size=size)
        except Exception:
            return None
        im = Image.new("RGB", (canvas, canvas), (255, 255, 255))
        dr = ImageDraw.Draw(im)
        bbox = dr.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= 0 or h <= 0:
            continue
        x = (canvas - w) // 2 - bbox[0]
        y = (canvas - h) // 2 - bbox[1]
        dr.text((x, y), ch, fill=(0, 0, 0), font=font)
        a = np.asarray(im.convert("L"))
        ink = a < 250
        if not ink.any():
            continue
        ys, xs = np.where(ink)
        m = min(xs.min(), ys.min(), canvas - 1 - xs.max(), canvas - 1 - ys.max())
        score = abs(m - target_margin)
        if score < best_score:
            best, best_score = im, score
    return best


def ink_ratio(im: Image.Image) -> float:
    a = np.asarray(im.convert("L"))
    return float((a < 250).mean())


def ncc(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("L"), dtype=np.float32).ravel()
    y = np.asarray(b.convert("L"), dtype=np.float32).ravel()
    x = x - x.mean()
    y = y - y.mean()
    d = float(np.linalg.norm(x) * np.linalg.norm(y)) + 1e-8
    return float(np.dot(x, y) / d)


def png_name(ch: str) -> str:
    return f"u{ord(ch):04X}.png"


def render_one(job: dict) -> dict:
    ttf = Path(job["ttf"])
    ch = job["ch"]
    canvas = int(job["canvas"])
    dest = Path(job["dest"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    rec = {"font": job["font"], "ch": ch, "canvas": canvas, "ok": False, "reason": ""}
    if dest.exists() and dest.stat().st_size > 0:
        rec.update(ok=True, reason="exists", path=str(dest))
        return rec
    if not cmap_ok(str(ttf), ch):
        rec["reason"] = "no_cmap"
        return rec
    im = render_fit(ttf, ch, canvas)
    if im is None:
        rec["reason"] = "render_fail"
        return rec
    if ink_ratio(im) < 0.004:
        rec["reason"] = "empty_ink"
        return rec
    im.save(dest)
    rec.update(ok=True, reason="wrote", path=str(dest), ink=ink_ratio(im))
    return rec


def collect_jobs(fonts: list[str], chars: list[str], canvases: list[int]):
    jobs, missing = [], []
    for stem in fonts:
        ttf = resolve_ttf(stem)
        if ttf is None:
            missing.append(stem)
            continue
        for canvas in canvases:
            for ch in chars:
                dest = OUT / f"r{canvas}" / stem / png_name(ch)
                jobs.append(
                    {
                        "font": stem,
                        "ttf": str(ttf),
                        "ch": ch,
                        "canvas": canvas,
                        "dest": str(dest),
                    }
                )
    return jobs, missing


def render_b0(chars: list[str], canvases: list[int]) -> None:
    if not B0_TTF.exists():
        log(f"B0 missing: {B0_TTF}")
        return
    for canvas in canvases:
        for ch in chars:
            dest = OUT / f"r{canvas}" / "_B0_" / png_name(ch)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                continue
            im = render_fit(B0_TTF, ch, canvas)
            if im is not None:
                im.save(dest)


def filter_fake_latin(train_fonts: list[str], latin_chars: list[str], canvas: int) -> dict:
    report: dict = {"dropped": [], "kept": 0, "canvas": canvas}
    dejavu_cache: dict[str, Image.Image] = {}
    if DEJAVU.exists():
        for ch in latin_chars:
            im = render_fit(DEJAVU, ch, canvas)
            if im is not None:
                dejavu_cache[ch] = im
    for stem in train_fonts:
        for ch in latin_chars:
            p = OUT / f"r{canvas}" / stem / png_name(ch)
            if not p.exists():
                report["dropped"].append({"font": stem, "ch": ch, "why": "missing_png"})
                continue
            im = Image.open(p)
            why = None
            if ink_ratio(im) < 0.004:
                why = "empty_ink"
            elif ch in dejavu_cache and ncc(im, dejavu_cache[ch]) > 0.985:
                why = "dejavu_clone"
            if why:
                report["dropped"].append({"font": stem, "ch": ch, "why": why})
                p.rename(p.with_suffix(".fake.png"))
            else:
                report["kept"] += 1
    return report


def freeze_gap(chars: list[str], canvas: int = 96) -> dict:
    if not B0_TTF.exists():
        return {"error": "no B0"}
    masks = {}
    for ch in list(chars) + REF8:
        im = render_fit(B0_TTF, ch, canvas)
        if im is None:
            continue
        masks[ch] = np.asarray(im.convert("L")) < 250

    def iou(a, b) -> float:
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum() + 1e-8
        return float(inter / union)

    rows = []
    for c in chars:
        if c not in masks:
            continue
        sims = [iou(masks[c], masks[r]) for r in REF8 if r in masks]
        g = 1.0 - max(sims) if sims else 1.0
        best = REF8[int(np.argmax(sims))] if sims else ""
        rows.append({"ch": c, "gap": g, "best_ref": best})
    rows.sort(key=lambda x: -x["gap"])
    return {"ref8": REF8, "rows": rows}


def pick_calib4(train_fonts: list[str]) -> list[str]:
    n = len(train_fonts)
    if n < 4:
        return list(train_fonts)
    return [train_fonts[i] for i in (0, n // 3, 2 * n // 3, n - 1)]


def cache_encoders(train_fonts: list[str], chars: list[str], canvas: int, ckpt: Path, device: str) -> dict:
    import torch
    from torchvision import transforms

    sys.path.insert(0, str(ROOT / "code/ours/FontDiffuser"))
    from configs.fontdiffuser import get_parser
    from src import build_content_encoder, build_style_encoder

    enc_size = 96
    args = get_parser().parse_args([])
    args.content_image_size = (enc_size, enc_size)
    args.style_image_size = (enc_size, enc_size)
    args.resolution = enc_size
    ce = build_content_encoder(args=args)
    se = build_style_encoder(args=args)
    ce.load_state_dict(torch.load(ckpt / "content_encoder.pth", map_location="cpu"))
    se.load_state_dict(torch.load(ckpt / "style_encoder.pth", map_location="cpu"))
    ce.eval().to(device)
    se.eval().to(device)
    tfm = transforms.Compose(
        [
            transforms.Resize((enc_size, enc_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def embed_style(path: Path):
        t = tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            f, _, _ = se(t)
        return f.flatten(1).mean(0).cpu()

    def embed_content(path: Path):
        t = tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            f, _ = ce(t)
        return f.flatten(1).mean(0).cpu()

    style_proto = {}
    with torch.no_grad():
        for stem in train_fonts:
            feats = []
            for r in REF8:
                p = OUT / f"r{canvas}" / stem / png_name(r)
                if p.exists():
                    feats.append(embed_style(p))
            if feats:
                style_proto[stem] = torch.stack(feats).mean(0)

    content_vec = {}
    with torch.no_grad():
        for ch in chars:
            p = OUT / f"r{canvas}" / "_B0_" / png_name(ch)
            if p.exists():
                content_vec[ch] = embed_content(p)

    cache_dir = OUT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "enc_size": enc_size,
        "canvas": canvas,
        "ckpt": str(ckpt),
        "n_style": len(style_proto),
        "n_content": len(content_vec),
    }
    torch.save({"style_proto": style_proto, "content_vec": content_vec, "meta": meta}, cache_dir / f"ec_es_r{canvas}.pt")
    (cache_dir / f"ec_es_r{canvas}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["render", "filter", "cache", "all"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--canvases", default="96,256")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    canvases = [int(x) for x in args.canvases.split(",") if x]
    train_fonts = list(META["train_fonts"])
    demo = list(META["test_fonts"])
    latin = p1_chars()
    chars = latin + REF8 + DONORS
    OUT.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    (OUT / "meta.json").write_text(
        json.dumps(
            {
                "train_fonts": train_fonts,
                "demo8": demo,
                "ref8": REF8,
                "donors": DONORS,
                "latin_n": len(latin),
                "canvases": canvases,
                "note": "Demo-8 rendered for eval only; excluded from alpha bank",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"E0 start phase={args.phase} fonts={len(train_fonts)} chars={len(chars)} canvases={canvases}")

    if args.phase in {"render", "all"}:
        render_b0(chars, canvases)
        jobs, missing = collect_jobs(train_fonts, chars, canvases)
        demo_jobs, demo_miss = collect_jobs(demo, chars, canvases)
        jobs = jobs + demo_jobs
        log(f"jobs={len(jobs)} missing_ttf={missing + demo_miss}")
        ok = fail = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(render_one, j) for j in jobs]
            done = 0
            for fut in as_completed(futs):
                rec = fut.result()
                done += 1
                if rec.get("ok"):
                    ok += 1
                else:
                    fail += 1
                if done % 500 == 0:
                    log(f"render {done}/{len(jobs)} ok={ok} fail={fail}")
        log(f"render done ok={ok} fail={fail}")
        (REP / "e0_render.json").write_text(
            json.dumps({"ok": ok, "fail": fail, "missing_ttf": missing + demo_miss}, indent=2),
            encoding="utf-8",
        )

    if args.phase in {"filter", "all"}:
        for canvas in canvases:
            rep = filter_fake_latin(train_fonts, latin, canvas=canvas)
            (REP / f"fake_latin_r{canvas}.json").write_text(
                json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            log(f"filter r{canvas} kept={rep['kept']} dropped={len(rep['dropped'])}")

    if args.phase in {"cache", "all"}:
        gap = freeze_gap(latin, canvas=96)
        (OUT / "gap_b0_iou.json").write_text(json.dumps(gap, ensure_ascii=False, indent=2), encoding="utf-8")
        calib = pick_calib4(train_fonts)
        (OUT / "calib4.json").write_text(json.dumps(calib, ensure_ascii=False, indent=2), encoding="utf-8")
        ckpt = ROOT / "runs/ft_cnstyle/global_step_25000"
        if not (ckpt / "content_encoder.pth").exists():
            ckpt = ROOT / "code/ours/FontDiffuser/ckpt"
        for canvas in canvases:
            try:
                info = cache_encoders(train_fonts, chars, canvas, ckpt, args.device)
                log(f"cache r{canvas} {info}")
            except Exception as e:
                log(f"cache r{canvas} FAILED: {type(e).__name__}: {e}")

    (REP / "e0_done.json").write_text(
        json.dumps({"t": time.time(), "phase": args.phase}, indent=2), encoding="utf-8"
    )
    log("E0 phase complete")


if __name__ == "__main__":
    main()
