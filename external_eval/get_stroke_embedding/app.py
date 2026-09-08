#!/usr/bin/env python3
"""Standalone skeleton and stroke embedding HTTP API."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from models import StrokeEmbedder
from skeleton import SkeletonExtractor


def render_char(ttf_path, char, size=256):
    if len(char) != 1:
        raise ValueError("char must contain exactly one character")
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    font_size = int(size * 0.8)
    while font_size >= 12:
        font = ImageFont.truetype(ttf_path, font_size)
        bbox = draw.textbbox((0, 0), char, font=font)
        if bbox[2] - bbox[0] <= size * 0.82 and bbox[3] - bbox[1] <= size * 0.82:
            break
        font_size -= 4
    x = (size - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (size - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), char, font=font, fill="black")
    return np.asarray(image, dtype=np.uint8)


def decode_image(payload, size=256):
    if payload.get("image_base64"):
        encoded = payload["image_base64"]
        if encoded.strip().startswith("data:") and "," in encoded:
            encoded = encoded.split(",", 1)[1]
        raw = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        array = np.asarray(image)
    elif payload.get("image_path"):
        image_path = Path(payload["image_path"])
        if not image_path.is_file():
            raise ValueError(f"image not found: {image_path}")
        array = np.asarray(Image.open(image_path).convert("RGB"))
    elif payload.get("ttf_path") and payload.get("char"):
        array = render_char(payload["ttf_path"], payload["char"], size=size)
    else:
        raise ValueError("provide image_base64, image_path, or ttf_path + char")
    return cv2.resize(array, (size, size), interpolation=cv2.INTER_AREA)


def cosine_similarity(left, right):
    left_tensor = torch.as_tensor(left, dtype=torch.float32).reshape(1, -1)
    right_tensor = torch.as_tensor(right, dtype=torch.float32).reshape(1, -1)
    return float(F.cosine_similarity(left_tensor, right_tensor, dim=1).item())


def dice_similarity(left, right, eps=1e-8):
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    return float((2.0 * (left * right).sum() + eps) / (left.sum() + right.sum() + eps))


class FeatureService:
    def __init__(self, weights, device, stroke_mode, skeleton_url, skeleton_size, allow_fallback):
        self.device = device
        self.stroke_mode = stroke_mode
        self.skeleton_size = int(skeleton_size)
        self.stroke_embedder = StrokeEmbedder(weights, device=device, mode=stroke_mode)
        self.skeleton_extractor = SkeletonExtractor(
            skeleton_url,
            timeout=60,
            allow_fallback=allow_fallback,
        )
        self.stroke_lock = threading.Lock()

    def embed_stroke(self, payload):
        image = decode_image(payload)
        with self.stroke_lock:
            embedding = self.stroke_embedder.extract(image)
        return {
            "embedding": embedding.tolist(),
            "embedding_dim": int(embedding.size),
            "mode": self.stroke_mode,
        }, embedding

    def embed_skeleton(self, payload):
        image = decode_image(payload)
        skeleton, source = self.skeleton_extractor.extract(image)
        gray = cv2.cvtColor(skeleton, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(
            gray,
            (self.skeleton_size, self.skeleton_size),
            interpolation=cv2.INTER_AREA,
        )
        mask = (small < 250).astype(np.float32)
        embedding = mask.reshape(-1)
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm
        return {
            "embedding": embedding.tolist(),
            "embedding_dim": int(embedding.size),
            "mask_sum": float(mask.sum()),
            "skeleton_source": source,
        }, mask

    def similarity(self, payload, feature_type):
        if feature_type == "stroke":
            left_response, left = self.embed_stroke(payload["left"])
            right_response, right = self.embed_stroke(payload["right"])
            return {
                "feature_type": "stroke",
                "cosine_similarity": cosine_similarity(left, right),
                "left": left_response,
                "right": right_response,
            }
        left_response, left = self.embed_skeleton(payload["left"])
        right_response, right = self.embed_skeleton(payload["right"])
        return {
            "feature_type": "skeleton",
            "cosine_similarity": cosine_similarity(left, right),
            "dice_similarity": dice_similarity(left, right),
            "left": left_response,
            "right": right_response,
        }


SERVICE = None


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "StandaloneFeatureEmbeddingAPI/1.0"

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > 32 * 1024 * 1024:
            raise ValueError("request body exceeds 32 MB")
        raw = self.rfile.read(content_length) if content_length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self.send_json(
                {
                    "status": "ok",
                    "device": SERVICE.device,
                    "stroke_mode": SERVICE.stroke_mode,
                    "skeleton_embedding_dim": SERVICE.skeleton_size ** 2,
                }
            )
            return
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        try:
            path = urlparse(self.path).path
            payload = self.read_json()
            if path == "/embed/stroke":
                response, _ = SERVICE.embed_stroke(payload)
            elif path == "/embed/skeleton":
                response, _ = SERVICE.embed_skeleton(payload)
            elif path == "/similarity/stroke":
                response = SERVICE.similarity(payload, "stroke")
            elif path == "/similarity/skeleton":
                response = SERVICE.similarity(payload, "skeleton")
            else:
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            self.send_json(response)
        except Exception as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, message_format, *args):
        print(f"[{self.log_date_time_string()}] {message_format % args}", flush=True)


def resolve_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is visible in the container")
    return requested


def main():
    parser = argparse.ArgumentParser(description="独立骨架/笔触 embedding API")
    parser.add_argument("--weights", required=True, help="export_weights.py 导出的轻量权重")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stroke-mode", default="style_token", choices=("style_token", "pooled_feature"))
    parser.add_argument("--skeleton-url", default=os.getenv("SKELETON_API_URL", "http://172.19.52.231:28000/extract_skeleton"))
    parser.add_argument("--skeleton-size", type=int, default=64)
    parser.add_argument("--no-skeleton-fallback", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    global SERVICE
    SERVICE = FeatureService(
        weights=args.weights,
        device=device,
        stroke_mode=args.stroke_mode,
        skeleton_url=args.skeleton_url,
        skeleton_size=args.skeleton_size,
        allow_fallback=not args.no_skeleton_fallback,
    )
    print(f"API listening on http://{args.host}:{args.port}", flush=True)
    print(f"device={device}, stroke_mode={args.stroke_mode}", flush=True)
    ThreadingHTTPServer((args.host, args.port), RequestHandler).serve_forever()


if __name__ == "__main__":
    main()

