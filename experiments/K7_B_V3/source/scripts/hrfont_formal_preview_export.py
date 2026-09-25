#!/usr/bin/env python3
"""Export mentor report for offline / email — single HTML + zip bundle.

Mentor cannot open a lone downloaded index.html because:
  - viz_overview.png / viz_callouts.png are separate files
  - Google Fonts CDN may be blocked offline
  - file:// links to ../mentor_preview/ break

Outputs under reports/hrfont_overnight/formal_preview/export/:
  HR-Font_Mentor汇报_离线单页.html   — one file, images embedded, no CDN
  HR-Font_Mentor汇报_完整包.zip      — index + png + audit json
  打开说明.txt
"""
from __future__ import annotations

import base64
import re
import zipfile
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
PREVIEW = ROOT / "reports/hrfont_overnight/formal_preview"
EXPORT = PREVIEW / "export"

SINGLE_NAME = "HR-Font_Mentor汇报_离线单页.html"
ZIP_NAME = "HR-Font_Mentor汇报_完整包.zip"
README_NAME = "打开说明.txt"

README = """HR-Font Mentor 汇报 — 离线打开说明
================================

方式一（推荐）：双击「HR-Font_Mentor汇报_离线单页.html」
  - 仅一个文件，图片已内嵌，无需联网
  - 用 Chrome / Edge / Firefox 打开均可

方式二：解压「HR-Font_Mentor汇报_完整包.zip」后，双击里面的 index.html
  - 请保持 zip 内所有文件在同一文件夹，不要只拷贝 html

若仍打不开：
  1. 不要从微信/邮件里直接点预览，先「另存为」到电脑再双击
  2. 右键 → 打开方式 → 选择 Chrome 或 Edge
  3. 在线版（需内网/VPN）：见项目组 READ_ME_FIRST.md 中的 URL

注意：旧版单独下载的「HR-Font Mentor 汇报.html」缺少图片，请改用本包内文件。
"""


def _b64_data_uri(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _embed_images(html: str, preview: Path) -> str:
    for name in ("viz_overview.png", "viz_callouts.png"):
        p = preview / name
        if not p.exists():
            print(f"WARN missing {p}")
            continue
        uri = _b64_data_uri(p)
        html = html.replace(f'src="{name}"', f'src="{uri}"')
        html = html.replace(f"src='{name}'", f"src='{uri}'")
    return html


def _offline_fonts(html: str) -> str:
    html = re.sub(
        r'<link[^>]*fonts\.googleapis\.com[^>]*/>\s*',
        "",
        html,
        flags=re.IGNORECASE,
    )
    html = html.replace(
        '"IBM Plex Sans","Noto Sans SC",sans-serif',
        '"Segoe UI","Microsoft YaHei","PingFang SC","Helvetica Neue",Arial,sans-serif',
    )
    return html


def _fix_links(html: str) -> str:
    html = html.replace(
        '<a href="../mentor_preview/index.html">mentor_preview</a>',
        "mentor_preview（旧页，需服务器）",
    )
    html = html.replace(
        '<a href="PROTOCOL_AUDIT.json">PROTOCOL_AUDIT.json</a>',
        "PROTOCOL_AUDIT.json（见完整包 zip）",
    )
    banner = (
        '<div class="warnbox" style="margin-bottom:14px">'
        "<b>离线版</b>：图片已内嵌，无需联网。"
        "完整数据见同目录 zip 包内 PROTOCOL_AUDIT.json。</div>"
    )
    html = html.replace('<div class="wrap">', f'<div class="wrap">\n{banner}', 1)
    return html


def build_single_html(preview: Path, export_dir: Path) -> Path:
    src = preview / "index.html"
    if not src.exists():
        raise SystemExit(f"missing {src} — run hrfont_formal_preview_build.py first")
    html = src.read_text(encoding="utf-8")
    html = _offline_fonts(html)
    html = _embed_images(html, preview)
    html = _fix_links(html)
    out = export_dir / SINGLE_NAME
    out.write_text(html, encoding="utf-8")
    return out


def build_zip(preview: Path, export_dir: Path) -> Path:
    out = export_dir / ZIP_NAME
    files = [
        ("index.html", preview / "index.html"),
        ("viz_overview.png", preview / "viz_overview.png"),
        ("viz_callouts.png", preview / "viz_callouts.png"),
        ("PROTOCOL_AUDIT.json", preview / "PROTOCOL_AUDIT.json"),
        ("viz_meta.json", preview / "viz_meta.json"),
        (README_NAME, None),
    ]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for arc, path in files:
            if path is None:
                zf.writestr(arc, README.encode("utf-8"))
            elif path.exists():
                zf.write(path, arc)
            else:
                print(f"WARN zip skip missing {path}")
    return out


def main() -> None:
    EXPORT.mkdir(parents=True, exist_ok=True)
    single = build_single_html(PREVIEW, EXPORT)
    zpath = build_zip(PREVIEW, EXPORT)
    (EXPORT / README_NAME).write_text(README, encoding="utf-8")
    print(f"wrote {single} ({single.stat().st_size // 1024} KB)")
    print(f"wrote {zpath} ({zpath.stat().st_size // 1024} KB)")
    print(f"wrote {EXPORT / README_NAME}")


if __name__ == "__main__":
    main()
