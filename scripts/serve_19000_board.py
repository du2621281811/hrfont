#!/usr/bin/env python3
"""Replace `python3 -m http.server 19000` without changing mentor-facing URLs.

Keeps the existing glyph board / PI briefing at http://172.19.45.13:19000/
and mounts p649 ink review on the same port:

  /                                 reports/f03_test16_strat (unchanged)
  /PI_BRIEFING_20260909.html        briefing
  /p649_v2a_review/review.html      ink review (mentor-facing)
  /p649_v2a_layers/review.html      layered L2/L3 (hidden; not linked from /)
  /fontdiffuser-p649-.../           review PNGs
  PUT /p649_v2a_review/decisions.json
  PUT /p649_v2a_layers/layer2.json and layer3.json only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve_p649_layers import PUT_STORES as LAYER_PUT_STORES  # noqa: E402
from serve_p649_layers import ensure_files as ensure_layer_files  # noqa: E402
from serve_p649_review import DECISIONS_JSON, PUT_STORES, apply_incoming  # noqa: E402

ROOT = Path("/root/projects/hrfont")
BOARD = ROOT / "reports" / "f03_test16_strat"
DATA = ROOT / "data"
REVIEW_PREFIX = "/p649_v2a_review"
LAYERS_PREFIX = "/p649_v2a_layers"
DATASET_PREFIX = "/fontdiffuser-p649-"
REPORTS = ROOT / "reports"
ALL_PUT = {**PUT_STORES, **LAYER_PUT_STORES}


def _safe_target(base: Path, relative: str) -> Path:
    base_resolved = base.resolve()
    target = (base_resolved / relative).resolve()
    if target != base_resolved and base_resolved not in target.parents:
        return base_resolved / "__forbidden__"
    return target


class BoardHandler(SimpleHTTPRequestHandler):
    server_version = "HRFont19000/1.0"

    def end_headers(self) -> None:
        path = self.path.split("?")[0]
        if path in set(ALL_PUT) | {
            "/p649_v2a_review/review.html",
            "/p649_v2a_layers/review.html",
            "/p649_v2a_layers/l1_snapshot.json",
        }:
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        request_path = unquote(urlsplit(self.path).path)
        if request_path in {REVIEW_PREFIX, REVIEW_PREFIX + "/"}:
            self.send_response(302)
            self.send_header("Location", "/p649_v2a_review/review.html")
            self.end_headers()
            return
        if request_path in {LAYERS_PREFIX, LAYERS_PREFIX + "/"}:
            self.send_response(302)
            self.send_header("Location", "/p649_v2a_layers/review.html")
            self.end_headers()
            return
        super().do_GET()

    def do_OPTIONS(self) -> None:
        request_path = unquote(urlsplit(self.path).path)
        store = ALL_PUT.get(request_path)
        if store is None:
            self.send_error(403)
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, PUT, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_PUT(self) -> None:
        request_path = unquote(urlsplit(self.path).path)
        store = ALL_PUT.get(request_path)
        if store is None:
            self.send_error(403, "PUT only allowed for L1 decisions or layer2/layer3")
            return
        json_path, jsonl_path = store
        if request_path.startswith(LAYERS_PREFIX) and json_path.resolve() == DECISIONS_JSON.resolve():
            self.send_error(403, "refusing to write layer-1 decisions")
            return
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        try:
            data = json.loads(raw.decode("utf-8"))
            payload = apply_incoming(data, json_path=json_path, jsonl_path=jsonl_path)
        except json.JSONDecodeError:
            self.send_error(400, "invalid json")
            return
        except ValueError as exc:
            self.send_error(400, str(exc))
            return
        body = json.dumps({"ok": True, "n": payload["n"], "updated_at": payload["updated_at"]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def list_directory(self, path: str):  # type: ignore[override]
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(DATA.resolve())
        except ValueError:
            return super().list_directory(path)
        self.send_error(403, "Directory listing disabled")
        return None

    def translate_path(self, path: str) -> str:
        request_path = unquote(urlsplit(path).path)
        if (
            request_path.startswith(REVIEW_PREFIX)
            or request_path.startswith(LAYERS_PREFIX)
            or request_path.startswith(DATASET_PREFIX)
        ):
            return str(_safe_target(DATA, request_path.lstrip("/")))
        if request_path == "/reports" or request_path.startswith("/reports/"):
            return str(_safe_target(REPORTS, request_path[len("/reports") :].lstrip("/")))
        return str(_safe_target(BOARD, request_path.lstrip("/")))

    def log_message(self, fmt: str, *args: object) -> None:
        path = str(args[0]) if args else ""
        if any(k in path for k in ("PUT ", "review.html", "decisions.json", "layer2", "layer3", "PI_BRIEFING")):
            super().log_message(fmt, *args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=19000)
    args = ap.parse_args()
    if not BOARD.is_dir():
        raise FileNotFoundError(BOARD)
    if not (DATA / "p649_v2a_review" / "review.html").is_file():
        raise FileNotFoundError(DATA / "p649_v2a_review" / "review.html")
    ensure_layer_files()
    httpd = ThreadingHTTPServer((args.host, args.port), BoardHandler)
    print(
        f"19000 board+review http://{args.host}:{args.port}/ "
        f"(pid={os.getpid()}) board={BOARD} "
        f"layers=http://{args.host}:{args.port}/p649_v2a_layers/review.html",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
