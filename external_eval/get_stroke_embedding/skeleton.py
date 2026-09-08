"""Remote skeleton extraction with a local morphology fallback."""

from __future__ import annotations

import io

import cv2
import numpy as np
import requests
from PIL import Image


class SkeletonExtractor:
    def __init__(self, api_url, timeout=60, allow_fallback=True):
        self.api_url = api_url
        self.timeout = int(timeout)
        self.allow_fallback = bool(allow_fallback)

    def extract(self, image_rgb_255):
        try:
            image_bytes = io.BytesIO()
            Image.fromarray(image_rgb_255.astype(np.uint8)).save(image_bytes, format="PNG")
            response = requests.post(
                self.api_url,
                files={"img_file": ("image.png", image_bytes.getvalue(), "image/png")},
                timeout=self.timeout,
            )
            response.raise_for_status()
            skeleton = np.asarray(Image.open(io.BytesIO(response.content)).convert("RGB"))
            return skeleton, "remote"
        except Exception:
            if not self.allow_fallback:
                raise
            return extract_skeleton_fallback(image_rgb_255), "fallback"


def extract_skeleton_fallback(image_rgb_255):
    gray = cv2.cvtColor(image_rgb_255, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    skeleton = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while cv2.countNonZero(binary) > 0:
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(binary, opened))
        binary = cv2.erode(binary, element)

    return cv2.cvtColor(255 - skeleton, cv2.COLOR_GRAY2RGB)

