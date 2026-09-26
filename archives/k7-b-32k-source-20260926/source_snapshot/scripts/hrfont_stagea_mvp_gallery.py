#!/usr/bin/env python3
"""Build qualitative Control-vs-Delta sheets from saved paired predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = Path("/root/projects/hrfont")
DATA = ROOT / "data/fontdiffuser"
GT_ROOT = ROOT / "data/unified_v1/renders/128/gt_latin"
SIZE = 96
LABEL_W = 210
HEAD_H = 36
ROW_H = 112


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size
    )


def glyph_path(split: str, kind: str, face: str, ch: str) -> Path:
    code = f"u{ord(ch):04X}"
    if kind == "ContentImage":
        return DATA / split / kind / f"{code}.jpg"
    return DATA / split / kind / face / f"{face}+{code}.jpg"


def gt_path(face: str, ch: str) -> Path:
    target = glyph_path("val", "TargetImage", face, ch)
    if target.exists():
        return target
    name = ch if ch.isalnum() and ord(ch) < 128 else f"u{ord(ch):04X}"
    return GT_ROOT / face / f"{name}.png"


def tile(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)


def error_tile(prediction: Image.Image, target: Image.Image) -> Image.Image:
    error = ImageChops.difference(
        prediction.convert("L"), target.convert("L")
    ).point(lambda value: min(255, value * 3))
    return Image.merge("RGB", (error, Image.new("L", error.size), Image.new("L", error.size)))


def prediction_path(root: Path, face: str, ch: str) -> Path:
    return root / f"{face}__u{ord(ch):04X}.png"


def render_sheet(
    rows: list[tuple[str, str, float]],
    control_dir: Path,
    delta_dir: Path,
    output: Path,
    title: str,
) -> None:
    columns = ["Content", "Style: Yong", "Ground truth", "Control", "Delta", "|Err C| x3", "|Err D| x3"]
    width = LABEL_W + len(columns) * SIZE
    canvas = Image.new("RGB", (width, HEAD_H + len(rows) * ROW_H), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), title, fill="black", font=font(16))
    for index, label in enumerate(columns):
        draw.text((LABEL_W + index * SIZE + 5, 18), label, fill="black", font=font(10))

    for row_index, (face, ch, difference) in enumerate(rows):
        y = HEAD_H + row_index * ROW_H
        draw.text((8, y + 22), face, fill="black", font=font(11))
        draw.text(
            (8, y + 43),
            f"char={repr(ch)}  Delta-Control={difference:+.4f}",
            fill="black",
            font=font(11),
        )
        content = tile(glyph_path("val", "ContentImage", "", ch))
        style = tile(glyph_path("val", "StyleImage", face, "永"))
        target = tile(gt_path(face, ch))
        control = tile(prediction_path(control_dir, face, ch))
        delta = tile(prediction_path(delta_dir, face, ch))
        images = [
            content,
            style,
            target,
            control,
            delta,
            error_tile(control, target),
            error_tile(delta, target),
        ]
        for column_index, image in enumerate(images):
            canvas.paste(image, (LABEL_W + column_index * SIZE, y))
        draw.line((0, y + SIZE, width, y + SIZE), fill=(220, 220, 220), width=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-metrics", type=Path, required=True)
    parser.add_argument("--delta-metrics", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--delta-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8)
    args = parser.parse_args()

    control = json.loads(args.control_metrics.read_text(encoding="utf-8"))
    delta = json.loads(args.delta_metrics.read_text(encoding="utf-8"))
    control_l1 = {(row["font"], row["char"]): row["l1"] for row in control["rows"]}
    delta_l1 = {(row["font"], row["char"]): row["l1"] for row in delta["rows"]}
    available = []
    for face, ch in sorted(set(control_l1) & set(delta_l1)):
        if prediction_path(args.control_dir, face, ch).exists() and prediction_path(
            args.delta_dir, face, ch
        ).exists():
            available.append((face, ch, delta_l1[(face, ch)] - control_l1[(face, ch)]))
    if not available:
        raise RuntimeError("no paired predictions found")

    available.sort(key=lambda row: row[2])
    render_sheet(
        available[: args.rows],
        args.control_dir,
        args.delta_dir,
        args.output_dir / "delta_improvements.png",
        "Largest Delta improvements in the selected qualitative set (lower L1 is better)",
    )
    render_sheet(
        list(reversed(available[-args.rows :])),
        args.control_dir,
        args.delta_dir,
        args.output_dir / "delta_regressions.png",
        "Largest Delta regressions in the selected qualitative set (lower L1 is better)",
    )


if __name__ == "__main__":
    main()
