#!/usr/bin/env python3
"""Build the report hub from reports/REPORT_CATALOG.json (single source of truth).

Writes:
  reports/hub/index.html     — clickable hub
  reports/REPORT_HUB.md      — same content for git/search
  reports/hub/unregistered.json — HTML pages under reports/ not in catalog

Usage:
  python3 scripts/build_report_hub.py           # build
  python3 scripts/build_report_hub.py --check   # build + exit 1 if unregistered HTML
  python3 scripts/build_report_hub.py --serve   # build + serve :8780
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
CATALOG = REP / "REPORT_CATALOG.json"
HUB_DIR = REP / "hub"
HUB_HTML = HUB_DIR / "index.html"
HUB_MD = REP / "REPORT_HUB.md"
UNREG = HUB_DIR / "unregistered.json"

# Filenames that look like report pages (scanned for leak detection).
SCAN_NAMES = {
    "index.html",
    "timeline.html",
    "timeline_f2.html",
    "timeline_f2_clean.html",
    "core_shot_board.html",
    "multi_shot_board.html",
    "pi_highlights.html",
    "f2vec_eye_board.html",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_catalog() -> dict:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def page_by_id(cat: dict) -> dict[str, dict]:
    return {p["id"]: p for p in cat["pages"]}


def resolve_path(rel: str) -> Path:
    return REP / rel


def scan_unregistered(cat: dict) -> list[dict]:
    registered = set()
    for p in cat["pages"]:
        registered.add(Path(p["path"]).as_posix())
    found = []
    for p in REP.rglob("*.html"):
        if p.name not in SCAN_NAMES and not p.name.endswith("_board.html"):
            # still catch *board*.html
            if "board" not in p.name and p.name not in SCAN_NAMES:
                continue
        rel = p.relative_to(REP).as_posix()
        # skip generated hub itself + root redirect
        if rel.startswith("hub/") or rel == "index.html":
            continue
        if rel in registered:
            continue
        # skip large export one-offs under nested mentor exports
        if "/export/" in rel or rel.endswith("standalone.html"):
            continue
        found.append({"path": rel, "mtime": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()})
    found.sort(key=lambda x: x["path"])
    return found


def href_for(page: dict) -> str:
    """Relative from hub/index.html to the report file."""
    return "../" + page["path"]


def url_hint(page: dict) -> str:
    port = page.get("port")
    if not port:
        return ""
    path = page["path"]
    # when served from report-subdir servers, path may be filename-only
    if path.startswith("f03_test16_strat/") and port == 8791:
        return f"http://127.0.0.1:{port}/{path.split('/', 1)[1]}"
    if path.startswith("tonight_20260914_dashboard/") and port == 8790:
        return f"http://127.0.0.1:{port}/{path.split('/', 1)[1]}"
    return f"http://127.0.0.1:{port}/"


def build_md(cat: dict, unreg: list[dict]) -> str:
    lines = [
        f"# {cat['hub_title']}",
        "",
        f"> {cat['hub_blurb']}",
        "",
        f"生成：`{utcnow()}` · 源：`reports/REPORT_CATALOG.json` · 重建：`python3 scripts/build_report_hub.py`",
        "",
        "## 怎么用（防漏更新）",
        "",
        "1. **新增/改报告** → 只改 `REPORT_CATALOG.json` 一条（title / blurb / path / port / status）",
        "2. **重建导航** → `python3 scripts/build_report_hub.py`",
        "3. **查漏** → `python3 scripts/build_report_hub.py --check`（未登记 HTML 会列在 `hub/unregistered.json`）",
        "4. **打开** → `reports/hub/index.html` 或 http://127.0.0.1:8780/",
        "",
    ]
    by_id = page_by_id(cat)
    for g in cat.get("groups", []):
        lines.append(f"## {g['title']}")
        lines.append("")
        lines.append("| 报告 | 简介 | 打开 |")
        lines.append("|------|------|------|")
        for pid in g["page_ids"]:
            p = by_id[pid]
            exists = "✓" if resolve_path(p["path"]).exists() else "✗ 缺文件"
            link = p["path"]
            port = url_hint(p)
            open_col = f"[文件]({link})"
            if port:
                open_col += f" · [:{p['port']}]({port})"
            open_col += f" · {exists}"
            lines.append(f"| **{p['title']}** | {p['blurb']} | {open_col} |")
        lines.append("")
    # orphan catalog pages not in any group
    grouped = {pid for g in cat.get("groups", []) for pid in g["page_ids"]}
    orphans = [p for p in cat["pages"] if p["id"] not in grouped]
    if orphans:
        lines.append("## 未分组")
        lines.append("")
        for p in orphans:
            lines.append(f"- **{p['title']}** — {p['blurb']} (`{p['path']}`)")
        lines.append("")
    if unreg:
        lines.append("## ⚠ 未登记 HTML（可能漏更新目录）")
        lines.append("")
        for u in unreg[:40]:
            lines.append(f"- `{u['path']}`")
        if len(unreg) > 40:
            lines.append(f"- … 另有 {len(unreg) - 40} 个")
        lines.append("")
    return "\n".join(lines)


def build_html(cat: dict, unreg: list[dict]) -> str:
    by_id = page_by_id(cat)
    sections = []
    for g in cat.get("groups", []):
        cards = []
        for pid in g["page_ids"]:
            p = by_id[pid]
            exists = resolve_path(p["path"]).exists()
            port = url_hint(p)
            links = [f'<a href="{href_for(p)}">打开文件</a>']
            if port:
                links.append(f'<a href="{port}">端口 {p["port"]}</a>')
            miss = "" if exists else '<span class="miss">缺文件</span>'
            tags = " ".join(f'<span class="tag">{t}</span>' for t in p.get("tags", [])[:5])
            cards.append(
                f"""<article class="card">
  <h3>{p['title']} {miss}</h3>
  <p class="blurb">{p['blurb']}</p>
  <p class="meta">{tags} · {p.get('status','')} · 更新 {p.get('updated','—')}</p>
  <p class="links">{' · '.join(links)}</p>
  <p class="path"><code>{p['path']}</code></p>
</article>"""
            )
        sections.append(f"<section><h2>{g['title']}</h2><div class='grid'>{''.join(cards)}</div></section>")

    unreg_html = ""
    if unreg:
        items = "".join(f"<li><code>{u['path']}</code></li>" for u in unreg[:30])
        more = f"<li>… +{len(unreg)-30}</li>" if len(unreg) > 30 else ""
        unreg_html = f"<section class='warn'><h2>未登记 HTML（{len(unreg)}）</h2><ul>{items}{more}</ul><p class='meta'>请补进 REPORT_CATALOG.json 后重新 build</p></section>"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{cat['hub_title']}</title>
<style>
:root {{ --bg:#eef1f5; --card:#fff; --line:#d5dbe3; --muted:#5c6570; --ink:#1a1a1a; --accent:#1f4a6f; --warn:#9a6b12; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:14px/1.45 system-ui,sans-serif; background:var(--bg); color:var(--ink); }}
header {{ background:var(--card); border-bottom:1px solid var(--line); padding:18px 20px; }}
h1 {{ margin:0; font-size:1.35rem; }}
.lede {{ color:var(--muted); margin:.5rem 0 0; max-width:52rem; }}
.how {{ background:#fff8e8; border:1px solid #e6d7a8; margin:14px 20px; padding:10px 12px; font-size:13px; }}
main {{ max-width:1100px; margin:0 auto; padding:8px 16px 48px; }}
h2 {{ font-size:1.05rem; margin:22px 0 10px; border-bottom:1px solid var(--line); padding-bottom:6px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:10px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:6px; padding:12px 14px; }}
.card h3 {{ margin:0 0 6px; font-size:0.98rem; }}
.blurb {{ margin:0; color:var(--ink); }}
.meta,.path {{ color:var(--muted); font-size:12px; margin:6px 0 0; }}
.tag {{ display:inline-block; border:1px solid var(--line); padding:0 6px; margin-right:4px; font-size:11px; }}
.links a {{ color:var(--accent); margin-right:8px; }}
.miss {{ color:#8b2e2e; font-size:12px; }}
.warn {{ background:#fbeeee; border:1px solid #e0b4b4; padding:10px 14px; border-radius:6px; margin-top:20px; }}
code {{ font-size:12px; }}
</style>
</head>
<body>
<header>
  <h1>{cat['hub_title']}</h1>
  <p class="lede">{cat['hub_blurb']}</p>
  <p class="lede">生成 {utcnow()} · 源 REPORT_CATALOG.json</p>
</header>
<div class="how">
  <b>防漏更新：</b>改报告 → 改 <code>reports/REPORT_CATALOG.json</code> 一条 →
  <code>python3 scripts/build_report_hub.py</code> → 本页自动更新。
  查漏：<code>python3 scripts/build_report_hub.py --check</code>
</div>
<main>
{''.join(sections)}
{unreg_html}
</main>
</body>
</html>
"""


def build(check: bool = False) -> int:
    cat = load_catalog()
    HUB_DIR.mkdir(parents=True, exist_ok=True)
    unreg = scan_unregistered(cat)
    HUB_MD.write_text(build_md(cat, unreg), encoding="utf-8")
    HUB_HTML.write_text(build_html(cat, unreg), encoding="utf-8")
    UNREG.write_text(json.dumps(unreg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Convenience: top-level reports/index.html → hub
    (REP / "index.html").write_text(
        '<!doctype html><meta charset=utf-8><meta http-equiv="refresh" content="0;url=hub/">'
        "<title>HR-Font reports</title><p><a href='hub/'>进入报告导航</a></p>\n",
        encoding="utf-8",
    )
    print(f"wrote {HUB_HTML}")
    print(f"wrote {HUB_MD}")
    print(f"unregistered HTML: {len(unreg)}")
    if check and unreg:
        print("CHECK FAIL: register pages in REPORT_CATALOG.json or ignore intentionally")
        for u in unreg[:15]:
            print(" ", u["path"])
        return 1
    return 0


def serve(port: int = 8780) -> None:
    os.chdir(REP)
    httpd = ThreadingHTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler)
    print(f"serving reports/ on http://127.0.0.1:{port}/hub/")
    httpd.serve_forever()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8780)
    args = ap.parse_args()
    rc = build(check=args.check)
    if args.serve:
        # detach note: for long-run use nohup outside
        serve(args.port)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
