"""Export predicted cubic contours as editable SVG or GLIF."""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np


def _ordered_contours(points, segments, segment_contours):
    contours = []
    for contour_id in sorted(set(int(x) for x in segment_contours)):
        ids = np.flatnonzero(segment_contours == contour_id)
        if len(ids):
            contours.append([(points[segments[i]], int(i)) for i in ids])
    return contours


def export_svg(path, points, segments, segment_contours, canvas=1000):
    points = np.asarray(points, dtype=np.float64)
    segments = np.asarray(segments, dtype=np.int64)
    segment_contours = np.asarray(segment_contours, dtype=np.int64)
    commands = []
    for contour in _ordered_contours(points, segments, segment_contours):
        first = contour[0][0][0]
        commands.append(f"M {first[0]*canvas:.3f} {(1-first[1])*canvas:.3f}")
        for segment, _ in contour:
            _, c1, c2, end = segment
            commands.append(
                f"C {c1[0]*canvas:.3f} {(1-c1[1])*canvas:.3f} "
                f"{c2[0]*canvas:.3f} {(1-c2[1])*canvas:.3f} "
                f"{end[0]*canvas:.3f} {(1-end[1])*canvas:.3f}"
            )
        commands.append("Z")
    data = " ".join(commands)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {canvas} {canvas}">\n'
        f'  <path d="{escape(data)}" fill="black" fill-rule="nonzero"/>\n</svg>\n'
    )
    Path(path).write_text(svg, encoding="utf-8")


def export_glif(path, glyph_name, unicode_value, points, segments, segment_contours, advance_width=1000):
    """Write a UFO-compatible GLIF with editable cubic off-curve points."""
    points = np.asarray(points, dtype=np.float64)
    segments = np.asarray(segments, dtype=np.int64)
    segment_contours = np.asarray(segment_contours, dtype=np.int64)
    lines = [f'<glyph name="{escape(glyph_name)}" format="2">', f'  <advance width="{advance_width:.3f}"/>']
    lines.append(f'  <unicode hex="{int(unicode_value):04X}"/>')
    lines.append("  <outline>")
    for contour in _ordered_contours(points, segments, segment_contours):
        lines.append("    <contour>")
        start = contour[0][0][0]
        # These are closed contours. In GLIF, ``move`` denotes an open contour;
        # the first on-curve point must therefore be a regular line point.
        lines.append(f'      <point x="{start[0]*1000:.3f}" y="{start[1]*1000:.3f}" type="line"/>')
        for segment, _ in contour:
            _, c1, c2, end = segment
            lines.append(f'      <point x="{c1[0]*1000:.3f}" y="{c1[1]*1000:.3f}"/>')
            lines.append(f'      <point x="{c2[0]*1000:.3f}" y="{c2[1]*1000:.3f}"/>')
            lines.append(f'      <point x="{end[0]*1000:.3f}" y="{end[1]*1000:.3f}" type="curve"/>')
        lines.append("    </contour>")
    lines.extend(["  </outline>", "</glyph>"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
