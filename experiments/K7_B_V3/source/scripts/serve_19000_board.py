#!/usr/bin/env python3
"""Replace `python3 -m http.server 19000` without changing mentor-facing URLs.

Keeps the existing glyph board / PI briefing at http://172.19.45.13:19000/
and mounts p649 ink review + G shot board on the same port:

  /                                 reports/f03_test16_strat (unchanged)
  /PI_BRIEFING_20260909.html        briefing
  /g_shot/                          G v0913 1/8-shot board (reports/g_v0913_shot)
  /reports/...                      full reports tree
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
G_SHOT_PREFIX = "/g_shot"
REPORTS = ROOT / "reports"
G_SHOT = REPORTS / "g_v0913_shot"
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

    def do_OPTIONS(self) -> None:
        request_path = unquote(urlsplit(self.path).path)
        if request_path == "/api/ratings":
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            return
        store = ALL_PUT.get(request_path)
        if store is None:
            self.send_error(403)
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, PUT, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        request_path = unquote(urlsplit(self.path).path)
        if request_path == "/api/ratings":
            live = REPORTS / "e12_paper" / "human_ratings_live.json"
            if live.is_file():
                data = json.loads(live.read_text(encoding="utf-8"))
            else:
                data = {"rater": None, "scores": {}, "n": 0}
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return
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
        if request_path == G_SHOT_PREFIX:
            self.send_response(302)
            self.send_header("Location", "/g_shot/")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self) -> None:
        request_path = unquote(urlsplit(self.path).path)
        if request_path != "/api/ratings":
            self.send_error(405)
            return
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "invalid json")
            return
        rater = str(payload.get("rater") or "R1")
        scores_in = payload.get("scores") or {}
        clean = {}
        for k, v in scores_in.items():
            try:
                s = int(v)
            except (TypeError, ValueError):
                continue
            if 1 <= s <= 5:
                clean[str(k)] = s
        out_dir = REPORTS / "e12_paper"
        out_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime, timezone

        doc = {
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rater": rater,
            "n": len(clean),
            "scores": clean,
        }
        (out_dir / "human_ratings_live.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        items_path = out_dir / "human_board" / "items.json"
        items = json.loads(items_path.read_text(encoding="utf-8"))["items"] if items_path.is_file() else []
        lines = ["item_id,font,char,method,attention_check,rater,score"]
        for it in items:
            iid = it["item_id"]
            if iid not in clean:
                continue
            lines.append(
                f"{iid},{it['font']},{it['char']},{it['method']},"
                f"{1 if it.get('attention_check') else 0},{rater},{clean[iid]}"
            )
        (out_dir / "human_ratings_partial.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        body = json.dumps({"ok": True, "n": len(clean)}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
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
            or request_path.startswith("/data/")
            or request_path == "/data"
        ):
            rel = request_path[len("/data/") :] if request_path.startswith("/data/") else request_path.lstrip("/")
            if request_path.startswith("/data/") or request_path == "/data":
                return str(_safe_target(DATA, rel))
            return str(_safe_target(DATA, request_path.lstrip("/")))
        if request_path == G_SHOT_PREFIX or request_path.startswith(G_SHOT_PREFIX + "/"):
            rel = request_path[len(G_SHOT_PREFIX) :].lstrip("/")
            return str(_safe_target(G_SHOT, rel))
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
    if not (G_SHOT / "index.html").is_file():
        raise FileNotFoundError(G_SHOT / "index.html")
    ensure_layer_files()
    httpd = ThreadingHTTPServer((args.host, args.port), BoardHandler)
    print(
        f"19000 board+review http://{args.host}:{args.port}/ "
        f"(pid={os.getpid()}) board={BOARD} "
        f"g_shot=http://{args.host}:{args.port}/g_shot/ "
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
