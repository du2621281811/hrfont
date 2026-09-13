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
BOARD = ROOT / "reports/f03_test16_strat"
HUB = ROOT / "reports/v100_hub"

INDEX = """<!doctype html>
<meta charset="utf-8">
<title>V100 看板枢纽</title>
<style>
body{font-family:ui-sans-serif,system-ui,sans-serif;margin:32px;background:#111;color:#eee;max-width:72ch;line-height:1.5}
a{color:#9cf} .warn{color:#fd6} .ok{color:#7d7} code{color:#c8a24b}
</style>
<h1>V100 看板枢纽</h1>
<p class="warn">宿主机 <code>172.18.41.23</code> 只开了 22，没有映射 8791/19000。
在笔记本浏览器里直接打开 <code>http://0.0.0.0:8791</code> 或 <code>http://172.18.41.23:19000</code> 都会失败。</p>
<p class="ok">请任选一种：</p>
<ol>
<li><b>Cursor 端口转发</b>：Ports 面板转发 <code>19000</code>，然后打开
  <a href="http://127.0.0.1:19000/">http://127.0.0.1:19000/</a></li>
<li><b>SSH 隧道</b>（你 SSH 进的就是这台容器）：
  <pre>ssh -L 19000:127.0.0.1:19000 root@172.18.41.23</pre>
  再打开 <a href="http://127.0.0.1:19000/">http://127.0.0.1:19000/</a></li>
</ol>
<h2>页</h2>
<ul>
<li><a href="/f2vec_shot_board.html"><b>F2 / F2-RL / F2-VEC</b> 1-shot 与 8-shot 对照</a></li>
<li><a href="/g/"><b>Group G</b> 训练看板</a>（G0 → G2 → G1+G2-PRL）</li>
<li><a href="/v0913/">v0913_clean 训练看板</a>（F0-clean 进度）</li>
<li><a href="/core_shot_board.html">完整核心模型 Demo-8 对照</a></li>
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
                    "/g/", "/g/status.json", "/g/index.html"}:
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
        super().do_GET()

    def translate_path(self, path: str) -> str:
        request_path = unquote(urlsplit(path).path)
        if request_path in {"/", "/index.html"}:
            return str(HUB / "index.html")
        if request_path == "/v0913" or request_path.startswith("/v0913/"):
            rel = request_path[len("/v0913") :].lstrip("/")
            return str(_safe(SNAP, rel or "index.html"))
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
