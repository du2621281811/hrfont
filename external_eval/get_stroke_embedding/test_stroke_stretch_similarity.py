#!/usr/bin/env python3
"""Compare stroke embeddings before and after non-uniform image stretching."""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image


def load_rgb_image(image_path: Path, size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.asarray(image.resize((size, size), Image.Resampling.LANCZOS), dtype=np.uint8)


def stretch_on_fixed_canvas(image: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    height, width = image.shape[:2]
    transform = np.array(
        [
            [scale_x, 0.0, (1.0 - scale_x) * width / 2.0],
            [0.0, scale_y, (1.0 - scale_y) * height / 2.0],
        ],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def encode_png(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def save_comparison(original, stretched, similarity: float, output_path: Path) -> None:
    height, width = original.shape[:2]
    title_height = 48
    canvas = np.full((height + title_height, width * 2, 3), 255, dtype=np.uint8)
    canvas[title_height:, :width] = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    canvas[title_height:, width:] = cv2.cvtColor(stretched, cv2.COLOR_RGB2BGR)
    cv2.putText(canvas, "Original", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "Stretched", (width + 12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    text = f"cosine={similarity:.6f}"
    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
    cv2.putText(
        canvas,
        text,
        (width - text_width // 2, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (30, 80, 190),
        2,
    )
    cv2.imwrite(str(output_path), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description="测试图片拉伸前后的笔触 embedding 相似度")
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument("--api-url", default="http://127.0.0.1:8899", help="embedding API 地址")
    parser.add_argument("--scale-x", type=float, default=1.5, help="横向拉伸倍率")
    parser.add_argument("--scale-y", type=float, default=1.0, help="纵向拉伸倍率")
    parser.add_argument("--size", type=int, default=256, help="测试画布尺寸")
    parser.add_argument("--output-dir", default="./stretch_similarity_result", help="结果目录")
    parser.add_argument("--timeout", type=int, default=120, help="API 超时秒数")
    args = parser.parse_args()

    if args.scale_x <= 0 or args.scale_y <= 0:
        raise ValueError("scale-x 和 scale-y 必须大于 0")

    image_path = Path(args.image)
    if not image_path.is_file():
        raise ValueError(f"输入图片不存在: {image_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original = load_rgb_image(image_path, args.size)
    stretched = stretch_on_fixed_canvas(original, args.scale_x, args.scale_y)

    payload = {
        "left": {"image_base64": encode_png(original)},
        "right": {"image_base64": encode_png(stretched)},
    }
    endpoint = args.api_url.rstrip("/") + "/similarity/stroke"
    response = requests.post(endpoint, json=payload, timeout=args.timeout)
    response.raise_for_status()
    result = response.json()
    similarity = float(result["cosine_similarity"])

    Image.fromarray(original).save(output_dir / "original.png")
    Image.fromarray(stretched).save(output_dir / "stretched.png")
    save_comparison(original, stretched, similarity, output_dir / "comparison.png")
    (output_dir / "result.json").write_text(
        json.dumps(
            {
                "input_image": str(image_path.resolve()),
                "scale_x": args.scale_x,
                "scale_y": args.scale_y,
                "api_response": result,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"stroke cosine similarity: {similarity:.6f}")
    print(f"original: {output_dir / 'original.png'}")
    print(f"stretched: {output_dir / 'stretched.png'}")
    print(f"comparison: {output_dir / 'comparison.png'}")
    print(f"json: {output_dir / 'result.json'}")


if __name__ == "__main__":
    main()

