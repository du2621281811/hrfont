#!/usr/bin/env python3
"""Human-board static + ratings sync on :8793 (repo root).

    python scripts/serve_e12_human_board.py
POST /api/ratings  JSON {"rater":"R1","scores":{item_id: int, ...}}
GET  /api/ratings
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path("/root/projects/hrfont")
OUT = ROOT / "reports/e12_paper"
LIVE = OUT / "human_ratings_live.json"
CSV = OUT / "human_ratings_partial.csv"
PORT = 8793


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt, *args):
        print(f"[8793] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/ratings":
            if LIVE.is_file():
                data = json.loads(LIVE.read_text(encoding="utf-8"))
            else:
                data = {"rater": None, "scores": {}, "n": 0}
            return self._json(200, data)
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/ratings":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return self._json(400, {"ok": False, "error": "invalid json"})
        rater = str(payload.get("rater") or "R1")
        scores = payload.get("scores") or {}
        # coerce
        clean = {}
        for k, v in scores.items():
            try:
                s = int(v)
            except Exception:
                continue
            if 1 <= s <= 5:
                clean[str(k)] = s
        doc = {
            "updated_at": utc_now(),
            "rater": rater,
            "n": len(clean),
            "scores": clean,
        }
        OUT.mkdir(parents=True, exist_ok=True)
        LIVE.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # also write CSV aligned with items.json for spearman script
        items_path = OUT / "human_board" / "items.json"
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
        CSV.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return self._json(200, {"ok": True, "n": len(clean), "path": str(CSV)})


def main():
    # replace previous plain http.server if any
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"serving {ROOT} on http://127.0.0.1:{PORT}/  (API POST /api/ratings)", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
