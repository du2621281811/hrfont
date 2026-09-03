#!/usr/bin/env python3
"""Run FontDiffuser on L-accent only, append into phase1a outputs, then re-eval full set."""
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/root/projects/hrfont")
sys.path.insert(0, str(ROOT / "scripts"))
# import helpers from batch script with __file__ set
import importlib.util

spec = importlib.util.spec_from_file_location(
    "fd_batch", ROOT / "scripts" / "run_fontdiffuser_batch.py"
)
fd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fd)

from accelerate.utils import set_seed  # noqa: E402

ACCENT = list("àáèéêìíòóùúü")


def main():
    device = "cuda:0"
    data = ROOT / "data" / "canonical"
    meta = json.loads((data / "meta.json").read_text())
    out_root = ROOT / "runs" / "fontdiffuser" / "phase1a"
    pred_dir = out_root / "pred"
    cmp_dir = out_root / "compare"
    pred_dir.mkdir(parents=True, exist_ok=True)
    cmp_dir.mkdir(parents=True, exist_ok=True)

    fonts = meta["demo_fonts"]
    style_char = meta["ref16"][0]
    style_uname = f"U+{ord(style_char):04X}"

    args = fd.build_args(str(ROOT / "repos" / "FontDiffuser" / "ckpt"), device)
    set_seed(args.seed)
    print("Loading FontDiffuser...")
    pipe = fd.load_pipe(args)
    print("loaded")

    def resolve(ch, folder):
        for c in [
            folder / f"{ch}.png",
            folder / f"u{ord(ch):04X}.png",
            folder / f"U+{ord(ch):04X}.png",
            folder / f"U+{ord(ch):04X}_{ch}.png",
        ]:
            if c.exists():
                return c
        return None

    n = 0
    t0 = time.time()
    for fi, font_file in enumerate(fonts):
        stem = Path(font_file).stem
        style_path = data / "style_cn" / stem / f"{style_uname}.png"
        style_img = Image.open(style_path).convert("RGB")
        style_t = fd.to_tensor(style_img, args.style_image_size).to(device)
        for ch in ACCENT:
            content_path = resolve(ch, data / "content_latin")
            gt_path = resolve(ch, data / "gt_latin" / stem)
            if content_path is None or gt_path is None:
                print("skip", stem, ch, content_path, gt_path)
                continue
            tag = f"u{ord(ch):04X}"
            outp = pred_dir / f"{stem}__{tag}.png"
            if outp.exists():
                print("exists", outp.name)
                n += 1
                continue
            content_img = Image.open(content_path).convert("RGB")
            content_t = fd.to_tensor(content_img, args.content_image_size).to(device)
            with torch.no_grad():
                images = pipe.generate(
                    content_images=content_t,
                    style_images=style_t,
                    batch_size=1,
                    order=args.order,
                    num_inference_step=args.num_inference_steps,
                    content_encoder_downsample_size=args.content_encoder_downsample_size,
                    t_start=args.t_start,
                    t_end=args.t_end,
                    dm_size=args.content_image_size,
                    algorithm_type=args.algorithm_type,
                    skip_type=args.skip_type,
                    method=args.method,
                    correcting_x0_fn=args.correcting_x0_fn,
                )
            pred = images[0]
            if not isinstance(pred, Image.Image):
                pred = fd.tensor_to_pil(pred)
            else:
                pred = pred.convert("RGB")
            pred.save(outp)
            gt = Image.open(gt_path).convert("RGB")
            fd.make_compare(style_img, content_img, pred, gt, f"{stem} | char={ch}").save(
                cmp_dir / f"{stem}__{tag}.png"
            )
            n += 1
            print(f"[{fi+1}/{len(fonts)}] {stem} {ch} ok")
    print("DONE accents", n, "elapsed", time.time() - t0)


if __name__ == "__main__":
    main()
