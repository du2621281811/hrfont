#!/usr/bin/env python3
"""Compute L1/SSIM/LPIPS for FontDiffuser preds vs GT; build overview grids."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/root/projects/hrfont")


def to_arr(im: Image.Image, size=96):
    im = im.convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def ssim(a, b):
    # simple SSIM
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = a.mean(), b.mean()
    sig_a, sig_b = a.var(), b.var()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)) / ((mu_a**2 + mu_b**2 + C1) * (sig_a + sig_b + C2)))


def decode_char_tag(tag: str) -> str:
    """Map pred filename tag back to character (handles u00E0-style accents)."""
    if tag.startswith("u") and len(tag) == 5:
        try:
            return chr(int(tag[1:], 16))
        except ValueError:
            return tag
    if tag.startswith("U+") and len(tag) >= 6:
        try:
            return chr(int(tag[2:], 16))
        except ValueError:
            return tag
    return tag


def resolve_gt(gt_root: Path, stem: str, ch: str):
    folder = gt_root / stem
    for name in [f"{ch}.png", f"u{ord(ch):04X}.png", f"U+{ord(ch):04X}.png", f"U+{ord(ch):04X}_{ch}.png"]:
        p = folder / name
        if p.exists():
            return p
    return None


def eval_phase(phase: str):
    pred_dir = ROOT / "runs" / "fontdiffuser" / phase / "pred"
    gt_root = ROOT / "data" / "canonical" / "gt_latin"
    rows = []
    for p in sorted(pred_dir.glob("*.png")):
        stem, tag = p.stem.split("__", 1)
        ch = decode_char_tag(tag)
        gt = resolve_gt(gt_root, stem, ch)
        if gt is None:
            continue
        pa, ga = to_arr(Image.open(p)), to_arr(Image.open(gt))
        l1 = float(np.abs(pa - ga).mean())
        rows.append({"font": stem, "char": ch, "L1": l1, "SSIM": ssim(pa, ga)})
    if not rows:
        return None
    mean_l1 = sum(r["L1"] for r in rows) / len(rows)
    mean_ssim = sum(r["SSIM"] for r in rows) / len(rows)
    out = {
        "phase": phase,
        "n": len(rows),
        "L1_mean": mean_l1,
        "SSIM_mean": mean_ssim,
        "per_item": rows,
    }
    outp = ROOT / "runs" / "fontdiffuser" / phase / "metrics.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    # csv summary
    (ROOT / "runs" / "fontdiffuser" / phase / "metrics_summary.csv").write_text(
        f"phase,n,L1_mean,SSIM_mean\n{phase},{len(rows)},{mean_l1:.6f},{mean_ssim:.6f}\n",
        encoding="utf-8",
    )
    return out


def build_grid(phase: str, fonts=None, chars=None):
    cmp_dir = ROOT / "runs" / "fontdiffuser" / phase / "compare"
    files = sorted(cmp_dir.glob("*.png"))
    if fonts:
        files = [f for f in files if any(f.name.startswith(ft + "__") for ft in fonts)]
    if chars:
        files = [f for f in files if any(f.name.endswith("__" + c + ".png") for c in chars)]
    if not files:
        return
    # take up to 24
    files = files[:24]
    imgs = [Image.open(f).convert("RGB") for f in files]
    w, h = imgs[0].size
    cols = 2
    rows = (len(imgs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w + 20, rows * h + 40), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 10), f"FontDiffuser CN->Latin | {phase}", fill=(0, 0, 0), font=font)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas.paste(im, (10 + c * w, 35 + r * h))
    out = ROOT / "reports" / f"fontdiffuser_{phase}_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print("saved", out, canvas.size)


def main():
    for phase in ["phase0", "phase1a"]:
        if (ROOT / "runs" / "fontdiffuser" / phase / "pred").exists():
            m = eval_phase(phase)
            print(phase, m and {k: m[k] for k in ["n", "L1_mean", "SSIM_mean"]})
            build_grid(phase, chars=list("AaBgRQ0"))


if __name__ == "__main__":
    main()
