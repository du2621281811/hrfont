#!/usr/bin/env python3
"""V100 local hub: training board + F2/F2-RL/F2-VEC compare.

This container's :8791 / :19000 are NOT published on the host NIC.
From a laptop, SSH-tunnel or Cursor port-forward:

  ssh -L 19000:127.0.0.1:19000 root@172.18.41.23
  open http://127.0.0.1:19000/

Do not use http://0.0.0.0:8791 or http://172.18.41.23:19000 — those will fail.
"""
from __future__ import annotations

import argparse
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path("/root/projects/hrfont")
SNAP = ROOT / "reports/v0913_dashboard"
GSNAP = ROOT / "reports/g_dashboard"
GSHOT = ROOT / "reports/g_v0913_shot"
BOARD = ROOT / "reports/f03_test16_strat"
HUB = ROOT / "reports/v100_hub"

INDEX = """<!doctype html>
<meta charset="utf-8">
<title>V100 看板枢纽</title>
<style>
body{font-family:ui-sans-serif,system-ui,sans-serif;margin:32px;background:#111;color:#eee;max-width:72ch;line-height:1.5}
a{color:#9cf} .warn{color:#fd6} .ok{color:#7d7} code{color:#c8a24b}
.hero{font-size:18px;margin:18px 0;padding:14px 16px;border:1px solid #345;background:#16202a}
</style>
<h1>V100 看板枢纽</h1>
<p class="hero ok"><b>G 系 0913 对照看板（直接点）：</b>
  <a href="/g_shot/">打开 G v0913 看板</a>
  · <a href="/g_shot/index.html">备用直链</a></p>
<p class="note">服务端已就绪。在 Cursor 里打开本页后点上方链接即可（相对路径，图片会正常加载）。
若外置浏览器空白：Ports 面板确认 <code>19000</code> 已转发，再开
<code>http://127.0.0.1:19000/g_shot/</code>。</p>
<h2>页</h2>
<ul>
<li><a href="/g_shot/"><b>G v0913</b> one/few-shot · vs dirty F2/F2-RL</a>（主看板）</li>
<li><a href="/f2vec_shot_board.html">F2 / F2-RL / F2-VEC 对照</a></li>
<li><a href="/g/">Group G 训练进度</a></li>
<li><a href="/v0913/">v0913_clean 训练看板</a></li>
<li><a href="/core_shot_board.html">核心模型 Demo-8 对照</a></li>
</ul>
"""


def _safe(base: Path, relative: str) -> Path:
    root = base.resolve()
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        return root / "__forbidden__"
    return target


class HubHandler(SimpleHTTPRequestHandler):
    server_version = "HRFontV100Hub/1.0"

    def end_headers(self) -> None:
        path = self.path.split("?")[0]
        if path in {"/", "/v0913/", "/v0913/status.json", "/v0913/index.html",
                    "/g/", "/g/status.json", "/g/index.html",
                    "/g_shot/", "/g_shot/index.html"}:
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        request_path = unquote(urlsplit(self.path).path)
        if request_path in {"/v0913", "/v0913/"}:
            self.send_response(302)
            self.send_header("Location", "/v0913/index.html")
            self.end_headers()
            return
        if request_path in {"/g", "/g/"}:
            self.send_response(302)
            self.send_header("Location", "/g/index.html")
            self.end_headers()
            return
        if request_path in {"/g_shot", "/g_shot/"}:
            self.send_response(302)
            self.send_header("Location", "/g_shot/index.html")
            self.end_headers()
            return
        super().do_GET()

    def translate_path(self, path: str) -> str:
        request_path = unquote(urlsplit(path).path)
        if request_path in {"/", "/index.html"}:
            return str(HUB / "index.html")
        if request_path == "/v0913" or request_path.startswith("/v0913/"):
            rel = request_path[len("/v0913") :].lstrip("/")
            return str(_safe(SNAP, rel or "index.html"))
        if request_path == "/g_shot" or request_path.startswith("/g_shot/"):
            rel = request_path[len("/g_shot") :].lstrip("/")
            return str(_safe(GSHOT, rel or "index.html"))
        if request_path == "/g" or request_path.startswith("/g/"):
            rel = request_path[len("/g") :].lstrip("/")
            return str(_safe(GSNAP, rel or "index.html"))
        return str(_safe(BOARD, request_path.lstrip("/")))

    def log_message(self, fmt: str, *args: object) -> None:
        return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=19000)
    args = ap.parse_args()
    HUB.mkdir(parents=True, exist_ok=True)
    (HUB / "index.html").write_text(INDEX, encoding="utf-8")
    httpd = ThreadingHTTPServer((args.host, args.port), HubHandler)
    print(
        f"V100 hub http://127.0.0.1:{args.port}/ pid={os.getpid()} "
        f"(tunnel: ssh -L {args.port}:127.0.0.1:{args.port} root@172.18.41.23)",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
