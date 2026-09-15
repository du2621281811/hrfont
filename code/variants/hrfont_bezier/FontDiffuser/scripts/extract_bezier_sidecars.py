#!/usr/bin/env python3
"""Extract editable cubic outlines from a TTF/OTF into HR-Font sidecars.

Quadratic TrueType curves are converted exactly to cubic Beziers.  The outline
is uniformly scaled using the A-protocol font size and centred by vector bounds,
matching HR-Font's bbox-centred raster semantics without tracing the PNG.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont


class CubicCollectorPen(BasePen):
    def __init__(self, glyph_set):
        super().__init__(glyph_set)
        self.points = []
        self.point_types = []
        self.contour_ids = []
        self.segments = []
        self.segment_contours = []
        self.contour_segments = []
        self._current = None
        self._first = None
        self._active_segments = []
        self._contour = -1

    def _add(self, point, point_type):
        self.points.append((float(point[0]), float(point[1])))
        self.point_types.append(point_type)
        self.contour_ids.append(self._contour)
        return len(self.points) - 1

    def _moveTo(self, p0):
        self._contour += 1
        self._active_segments = []
        self._current = self._add(p0, 0)
        self._first = self._current

    def _append_cubic(self, c1, c2, end, end_index=None):
        i1 = self._add(c1, 1)
        i2 = self._add(c2, 1)
        i3 = self._add(end, 0) if end_index is None else end_index
        self.segments.append((self._current, i1, i2, i3))
        self.segment_contours.append(self._contour)
        self._active_segments.append(len(self.segments) - 1)
        self._current = i3

    def _lineTo(self, p1):
        p0 = np.asarray(self.points[self._current], dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        self._append_cubic(p0 + (p1 - p0) / 3, p0 + 2 * (p1 - p0) / 3, p1)

    def _curveToOne(self, p1, p2, p3):
        self._append_cubic(p1, p2, p3)

    def _qCurveToOne(self, p1, p2):
        p0 = np.asarray(self.points[self._current], dtype=np.float64)
        q = np.asarray(p1, dtype=np.float64)
        end = np.asarray(p2, dtype=np.float64)
        c1 = p0 + (2.0 / 3.0) * (q - p0)
        c2 = end + (2.0 / 3.0) * (q - end)
        self._append_cubic(c1, c2, end)

    def _closePath(self):
        if self._current != self._first:
            p0 = np.asarray(self.points[self._current], dtype=np.float64)
            p1 = np.asarray(self.points[self._first], dtype=np.float64)
            i1 = self._add(p0 + (p1 - p0) / 3, 1)
            i2 = self._add(p0 + 2 * (p1 - p0) / 3, 1)
            self.segments.append((self._current, i1, i2, self._first))
            self.segment_contours.append(self._contour)
            self._active_segments.append(len(self.segments) - 1)
        self.contour_segments.append(tuple(self._active_segments))
        self._current = self._first = None

    def _endPath(self):
        # Fonts should normally use closed contours; keep open contours explicit.
        self.contour_segments.append(tuple(self._active_segments))
        self._current = self._first = None

    def adjacency(self):
        pairs = []
        for group in self.contour_segments:
            pairs.extend((group[i], group[i + 1]) for i in range(len(group) - 1))
            if len(group) > 1:
                pairs.append((group[-1], group[0]))
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def codepoint_name(char):
    return f"u{ord(char):04X}"


def extract_one(font, glyph_set, cmap, char, font_size, canvas):
    cp = ord(char)
    if cp not in cmap:
        raise KeyError(f"font has no U+{cp:04X}")
    glyph_name = cmap[cp]
    pen = CubicCollectorPen(glyph_set)
    glyph_set[glyph_name].draw(pen)
    if not pen.points or not pen.segments:
        raise ValueError(f"empty outline for U+{cp:04X}")

    upm = float(font["head"].unitsPerEm)
    points = np.asarray(pen.points, dtype=np.float32)
    lo, hi = points.min(0), points.max(0)
    centre = (lo + hi) / 2
    scale = float(font_size) / upm / float(canvas)
    points = (points - centre) * scale + 0.5

    advance, lsb = font["hmtx"].metrics[glyph_name]
    glyph_width = hi[0] - lo[0]
    rsb = advance - lsb - glyph_width
    metrics = np.asarray([advance / upm, lsb / upm, rsb / upm], dtype=np.float32)
    return {
        "points": points,
        "point_types": np.asarray(pen.point_types, dtype=np.float32),
        "contour_ids": np.asarray(pen.contour_ids, dtype=np.int64),
        "segments": np.asarray(pen.segments, dtype=np.int64),
        "segment_contours": np.asarray(pen.segment_contours, dtype=np.int64),
        "adjacent_segments": pen.adjacency(),
        "metrics": metrics,
        "render_transform": np.asarray([scale, centre[0], centre[1], canvas], dtype=np.float32),
        "upm": np.asarray([upm], dtype=np.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--font", required=True)
    parser.add_argument("--font-id", required=True)
    parser.add_argument("--role", choices=("content", "target"), default="target",
                        help="content writes <cp>.npz; target writes <font-id>+<cp>.npz")
    parser.add_argument("--characters", help="literal character string")
    parser.add_argument("--characters-json", help="JSON file containing target_string")
    parser.add_argument("--font-size", required=True, type=int,
                        help="the frozen per-font A-protocol Pillow font size")
    parser.add_argument("--canvas", type=int, default=96)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if bool(args.characters) == bool(args.characters_json):
        parser.error("provide exactly one of --characters or --characters-json")

    chars = args.characters
    if args.characters_json:
        payload = json.loads(Path(args.characters_json).read_text(encoding="utf-8"))
        chars = payload["target_string"]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    font = TTFont(args.font, recalcBBoxes=False, recalcTimestamp=False)
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()
    manifest = {"font_id": args.font_id, "source": str(Path(args.font).resolve()), "written": [], "missing": []}
    for char in chars:
        cp = codepoint_name(char)
        try:
            outline = extract_one(font, glyph_set, cmap, char, args.font_size, args.canvas)
        except (KeyError, ValueError) as exc:
            manifest["missing"].append({"cp": cp, "reason": str(exc)})
            continue
        filename = f"{cp}.npz" if args.role == "content" else f"{args.font_id}+{cp}.npz"
        path = out / filename
        np.savez_compressed(path, **outline)
        manifest["written"].append(cp)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"written": len(manifest["written"]), "missing": len(manifest["missing"]), "output": str(out)}))


if __name__ == "__main__":
    main()
