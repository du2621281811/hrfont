#!/usr/bin/env python3
"""Rescore test16 FULL WESTERN (37 chars × 16 fonts = 592) under v0913 keep.

Scope:
  fonts = test16
  chars = digit + latin_upper + latin_lower + latin_ext (no kana/bopomofo)
  usability = manifests/v0913_clean/fonts.tsv (on test16 → all 592 kept)

Metrics:
  - E12 mem / Ours φ : ref8 style protocol (FROZEN)
  - L1 / SSIM / LPIPS : vs GT, same as eval_f03 (native size → [-1,1], NO forced 64)
  - CLIP / DINOv2 / LPIPS-Alex feature S01 : vs GT

    export HF_ENDPOINT=https://hf-mirror.com
    /root/miniforge3/envs/boogu/bin/python scripts/rescore_western_all.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/root/projects/hrfont")
sys.path.insert(0, str(ROOT / "scripts" / "eval_framework"))
from models import MembershipVerifier, load_phi_checkpoint  # noqa: E402

OUT = ROOT / "reports/f03_test16_strat"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
PAPER = ROOT / "reports/e12_paper"
PHI = ROOT / "runs/e12_phi_s2_b_s3407/best.pt"
MEM = ROOT / "runs/e12_membership_b_s3407/best.pt"
FONTS_TSV = ROOT / "manifests/v0913_clean/fonts.tsv"
REF8 = list("永和书风骨韵天地")
SEED = 3407
WEST = list("0123456789") + list("AGMQRWBCO") + list("aodpqbegcilnu") + list("àéüāě")
METHODS = [
    ("P1", "官方 P1", "baseline"),
    ("F0_100k", "F0 脏@100k", "dirty"),
    ("F0C_95000", "F0 净@95k(best)", "clean"),
    ("F0C_100000", "F0 净@100k", "clean"),
    ("F2_40000", "F2 脏@40k", "dirty"),
    ("F2C_40000", "F2 净@40k", "clean"),
    ("F2_80000", "F2 脏@80k", "dirty"),
    ("F2C_80000", "F2 净@80k", "clean"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cp_of(ch: str) -> str:
    return f"u{ord(ch):04X}"


def script_bucket(ch: str) -> str:
    if ch.isdigit():
        return "digit"
    name = unicodedata.name(ch, "")
    if "LATIN" in name and ch.isupper():
        return "latin_upper"
    if "LATIN" in name and ch.islower() and ord(ch) > 127:
        return "latin_ext"
    if "LATIN" in name and ch.islower():
        return "latin_lower"
    return "other"


def load_fonts() -> list[str]:
    return json.loads((OUT / "browse_index.json").read_text())["fonts"]


def load_usability() -> dict[str, str]:
    out = {}
    for line in FONTS_TSV.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0] != "stem":
            out[parts[0]] = parts[2]
    return out


def char_allowed(usability: str, ch: str) -> bool:
    b = script_bucket(ch)
    if usability == "exclude":
        return False
    if usability == "all_scripts":
        return True
    if usability == "no_bopomofo":
        return b != "bopomofo"
    if usability == "han_latin_digit":
        return b in ("digit", "latin_upper", "latin_lower", "latin_ext")
    return ch.isalnum() and ord(ch) < 128


def pred_path(mid: str, font: str, ch: str) -> Path:
    return OUT / "preds" / mid / "test" / font / f"test__{font}__{cp_of(ch)}__s{SEED}.png"


def gt_path(font: str, ch: str) -> Path | None:
    p = DATA / "test" / "TargetImage" / font / f"{font}+{cp_of(ch)}.png"
    return p if p.is_file() else None


def style_path(font: str, ch: str) -> Path:
    return DATA / "test" / "StyleImage" / font / f"{font}+{cp_of(ch)}.png"


def mean_std(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": None, "std": None}
    a = np.asarray(xs, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=0))}


def s01(c: float) -> float:
    return float((c + 1.0) / 2.0)


def fmt(x, d=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{d}f}"


def to_gray01(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    c1, c2 = 0.01**2, 0.03**2
    mu_a, mu_b = a.mean(), b.mean()
    sa, sb = a.var(), b.var()
    sab = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (sa + sb + c2) + 1e-12))


def load_rgb01(path: Path, size: int = 96) -> torch.Tensor:
    im = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None].repeat(3, 1, 1)


def load_rgb_pil(path: Path, size: int = 96) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)


def to_n11(im: Image.Image, device) -> torch.Tensor:
    arr = np.asarray(im, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)[None] * 2 - 1
    return t.to(device)


def iter_keys(fonts, usability):
    for font in fonts:
        u = usability.get(font, "")
        for ch in WEST:
            if char_allowed(u, ch):
                yield font, ch


def render_html(report: dict, path: Path) -> None:
    rows = []
    for mid, label, fam in METHODS:
        m = report["methods"][mid]
        n = m["n"]
        tds = "".join(
            f"<td>{fmt(m[k]['mean'] if isinstance(m.get(k), dict) else m.get(k))}</td>"
            for k in (
                "E12_mem",
                "Ours_phi",
                "CLIP_gt",
                "DINOv2_gt",
                "Alex_gt",
                "LPIPS",
                "L1",
                "SSIM",
            )
        )
        rows.append(f"<tr class='{fam}'><td>{label}</td><td>{fam}</td><td>{n}</td>{tds}</tr>")

    pairs = [
        ("F0C_100000", "F0_100k", "F0 净100k − 脏"),
        ("F0C_95000", "F0_100k", "F0 净95k − 脏"),
        ("F2C_40000", "F2_40000", "F2 净40k − 脏"),
        ("F2C_80000", "F2_80000", "F2 净80k − 脏"),
        ("F2C_40000", "P1", "F2 净40k − P1"),
    ]
    drows = []
    for a, b, name in pairs:
        cells = []
        for k in ("E12_mem", "Ours_phi", "CLIP_gt", "DINOv2_gt", "Alex_gt", "LPIPS"):
            va = report["methods"][a][k]["mean"]
            vb = report["methods"][b][k]["mean"]
            cells.append(fmt(None if va is None or vb is None else va - vb))
        drows.append(f"<tr><td>{name}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>全西文 592 · test16 重测</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:24px;background:#f5f3ef;color:#1a1a1a;line-height:1.45}}
h1{{font-size:1.35rem;margin:0 0 .35rem}} h2{{font-size:1.1rem;margin:1.4rem 0 .5rem}}
.meta,.note{{color:#555;font-size:.92rem;max-width:92ch}}
table{{border-collapse:collapse;background:#fff;margin:12px 0;font-size:13px}}
th,td{{border:1px solid #ddd;padding:6px 8px;text-align:right}}
th:first-child,td:first-child,td:nth-child(2){{text-align:left}}
tr.clean{{background:#f3faf3}} tr.dirty{{background:#fffaf3}} tr.baseline{{background:#f3f5fa}}
th.sty,td.sty{{background:#eef6ff}} th.gtm,td.gtm{{background:#f7f7f7}}
.links a{{margin-right:12px}}
</style></head><body>
<h1>全西文重测 · test16 × 37 西文</h1>
<p class="meta">生成 {report['computed_at']} · n={report['n_keys']}（期望 592）·
字体 test16 · 可用性 v0913_clean · 字集 digit+拉丁大小写+扩展（无假名/注音）</p>
<p class="note">
<strong>蓝列固定：</strong>E12 mem / Ours φ = 中文 ref8 → 西文 query。<br>
<strong>灰列 vs GT：</strong>CLIP/DINO/Alex/LPIPS/L1/SSIM；LPIPS 与 f03 同协议（不强制 64）。
</p>
<p class="links">
  <a href="clean_dirty_style_metrics.html">旧子集板</a>
  <a href="timeline_f2_clean.html">时间线</a>
</p>
<table>
<thead><tr><th>方法</th><th>臂</th><th>n</th>
<th class="sty">E12mem↑</th><th class="sty">Oursφ↑</th>
<th class="gtm">CLIP↑</th><th class="gtm">DINO↑</th><th class="gtm">Alex↑</th>
<th class="gtm">LPIPS↓</th><th class="gtm">L1↓</th><th class="gtm">SSIM↑</th>
</tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>成对差值</h2>
<table>
<thead><tr><th>对比</th><th class="sty">ΔE12</th><th class="sty">ΔOursφ</th>
<th class="gtm">ΔCLIP</th><th class="gtm">ΔDINO</th><th class="gtm">ΔAlex</th><th class="gtm">ΔLPIPS</th></tr></thead>
<tbody>{''.join(drows)}</tbody></table>
</body></html>
"""
    path.write_text(html, encoding="utf-8")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--skip-clip", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    fonts = load_fonts()
    usability = load_usability()
    keys = list(iter_keys(fonts, usability))
    print(f"[west] fonts={len(fonts)} chars={len(WEST)} keys={len(keys)}", flush=True)

    # --- E12 mem + Ours φ (frozen protocol) ---
    print("[west] load E12 φ+mem", flush=True)
    phi = load_phi_checkpoint(PHI, 3, 512, str(device)).to(device).eval()
    payload = torch.load(MEM, map_location=device, weights_only=False)
    mem = MembershipVerifier(phi, 512, 256, freeze_encoder=True).to(device)
    mem.load_state_dict(payload["model"])
    mem.eval()
    temp = float(payload.get("temperature", 1.0))

    refs = {}
    protos = {}
    for font in fonts:
        imgs = torch.stack([load_rgb01(style_path(font, ch)) for ch in REF8]).to(device)
        refs[font] = imgs
        protos[font] = F.normalize(phi(imgs).mean(0), dim=0)

    by: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    items = []

    def flush_style(mid, qs, fonts_b, metas):
        if not qs:
            return
        q = torch.stack(qs).to(device)
        r = torch.stack([refs[f] for f in fonts_b]).to(device)
        logits = mem(q, r)
        probs = torch.sigmoid(logits / temp)
        zq = F.normalize(phi(q), dim=1)
        for i, meta in enumerate(metas):
            proto = protos[meta["font"]]
            cos = float((zq[i] * proto).sum().item())
            rec = {
                **meta,
                "e12_mem_prob": float(probs[i]),
                "ours_phi_S01": s01(cos),
            }
            items.append(rec)
            by[mid]["E12_mem"].append(rec["e12_mem_prob"])
            by[mid]["Ours_phi"].append(rec["ours_phi_S01"])

    for mid, _, _ in METHODS:
        print(f"[west] style score {mid}", flush=True)
        qs, fs, metas = [], [], []
        for font, ch in keys:
            if mid == "P1" or mid.startswith("F"):
                pp = pred_path(mid, font, ch)
            else:
                continue
            if not pp.is_file():
                continue
            qs.append(load_rgb01(pp))
            fs.append(font)
            metas.append({"method": mid, "font": font, "char": ch, "cp": cp_of(ch), "bucket": script_bucket(ch)})
            if len(qs) >= args.batch:
                flush_style(mid, qs, fs, metas)
                qs, fs, metas = [], [], []
        flush_style(mid, qs, fs, metas)

    # --- pixel + LPIPS (f03 protocol) ---
    import lpips

    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    print("[west] pixel+LPIPS vs GT", flush=True)
    for mid, _, _ in METHODS:
        for font, ch in keys:
            pp = pred_path(mid, font, ch)
            gtp = gt_path(font, ch)
            if not pp.is_file() or gtp is None:
                continue
            pred = load_rgb_pil(pp)
            gt = load_rgb_pil(gtp)
            # match sizes
            if pred.size != gt.size:
                gt = gt.resize(pred.size, Image.Resampling.BILINEAR)
            pa, ga = to_gray01(pred), to_gray01(gt)
            l1 = float(np.abs(pa - ga).mean())
            ss = ssim(pa, ga)
            d = float(lpips_fn(to_n11(pred, device), to_n11(gt, device)).item())
            by[mid]["L1"].append(l1)
            by[mid]["SSIM"].append(ss)
            by[mid]["LPIPS"].append(d)
            # attach to matching item if exists
            for it in items:
                if it["method"] == mid and it["font"] == font and it["char"] == ch:
                    it["L1"] = l1
                    it["SSIM"] = ss
                    it["LPIPS"] = d
                    break

    # --- CLIP / DINO / Alex vs GT ---
    if not args.skip_clip:
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
        from torchvision import transforms

        print("[west] CLIP/DINO/Alex vs GT", flush=True)
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        dino = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
        dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        alex_lp = lpips.LPIPS(net="alex").to(device).eval()
        alex_tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        def embed_clip(ims: list[Image.Image]):
            inp = clip_proc(images=ims, return_tensors="pt")
            vision = clip.vision_model(pixel_values=inp["pixel_values"].to(device))
            return F.normalize(clip.visual_projection(vision.pooler_output), dim=1)

        def embed_dino(ims: list[Image.Image]):
            inp = dino_proc(images=ims, return_tensors="pt")
            out = dino(pixel_values=inp["pixel_values"].to(device))
            return F.normalize(out.last_hidden_state[:, 0], dim=1)

        def embed_alex(ims: list[Image.Image]):
            ts = []
            for im in ims:
                arr = np.asarray(im, dtype=np.float32) / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1)
                ts.append(alex_tf(t))
            x = torch.stack(ts).to(device)
            feats = alex_lp.net.forward(x)
            flat = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]
            return F.normalize(torch.cat(flat, dim=1), dim=1)

        for mid, _, _ in METHODS:
            print(f"[west] feat-gt {mid}", flush=True)
            buf_p, buf_g, metas = [], [], []

            def flush():
                nonlocal buf_p, buf_g, metas
                if not buf_p:
                    return
                zc_p, zc_g = embed_clip(buf_p), embed_clip(buf_g)
                zd_p, zd_g = embed_dino(buf_p), embed_dino(buf_g)
                za_p, za_g = embed_alex(buf_p), embed_alex(buf_g)
                for i, meta in enumerate(metas):
                    c_clip = float((zc_p[i] * zc_g[i]).sum().item())
                    c_dino = float((zd_p[i] * zd_g[i]).sum().item())
                    c_alex = float((za_p[i] * za_g[i]).sum().item())
                    by[mid]["CLIP_gt"].append(s01(c_clip))
                    by[mid]["DINOv2_gt"].append(s01(c_dino))
                    by[mid]["Alex_gt"].append(s01(c_alex))
                buf_p, buf_g, metas = [], [], []

            for font, ch in keys:
                pp = pred_path(mid, font, ch)
                gtp = gt_path(font, ch)
                if not pp.is_file() or gtp is None:
                    continue
                buf_p.append(load_rgb_pil(pp))
                buf_g.append(load_rgb_pil(gtp))
                metas.append((font, ch))
                if len(buf_p) >= args.batch:
                    flush()
            flush()

    methods_out = {}
    for mid, label, fam in METHODS:
        methods_out[mid] = {
            "label": label,
            "family": fam,
            "n": len(by[mid]["E12_mem"]) or len(by[mid]["L1"]),
            "E12_mem": mean_std(by[mid]["E12_mem"]),
            "Ours_phi": mean_std(by[mid]["Ours_phi"]),
            "CLIP_gt": mean_std(by[mid]["CLIP_gt"]),
            "DINOv2_gt": mean_std(by[mid]["DINOv2_gt"]),
            "Alex_gt": mean_std(by[mid]["Alex_gt"]),
            "LPIPS": mean_std(by[mid]["LPIPS"]),
            "L1": mean_std(by[mid]["L1"]),
            "SSIM": mean_std(by[mid]["SSIM"]),
        }

    report = {
        "computed_at": utc_now(),
        "scope": {
            "split": "test16",
            "fonts_n": len(fonts),
            "western_chars": WEST,
            "western_n": len(WEST),
            "keys": len(keys),
            "usability": "v0913_clean fonts.tsv",
            "note": "full western available scripts on test16; no kana/bopomofo",
        },
        "protocol": {
            "E12_mem_Ours_phi": "FROZEN ref8 style",
            "LPIPS_L1_SSIM": "vs GT; LPIPS alex native resolution (f03), no forced 64",
            "CLIP_DINO_Alex": "vs GT feature cosine → S01",
        },
        "n_keys": len(keys),
        "methods": methods_out,
    }
    out_json = OUT / "western_all_metrics.json"
    out_items = OUT / "western_all_items.json"
    out_html = OUT / "western_all_metrics.html"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    out_items.write_text(json.dumps(items, ensure_ascii=False) + "\n")
    render_html(report, out_html)

    print(f"\n{'method':<12} {'n':>4} {'E12':>7} {'Ours':>7} {'CLIP':>7} {'DINO':>7} {'Alex':>7} {'LPIPS':>7} {'L1':>7}")
    for mid, _, _ in METHODS:
        m = methods_out[mid]

        def g(k):
            v = m[k]["mean"]
            return f"{v:.4f}" if v is not None else "  —  "

        print(
            f"{mid:<12} {m['n']:4d} {g('E12_mem')} {g('Ours_phi')} {g('CLIP_gt')} "
            f"{g('DINOv2_gt')} {g('Alex_gt')} {g('LPIPS')} {g('L1')}"
        )
    print(f"wrote {out_html}")


if __name__ == "__main__":
    main()
