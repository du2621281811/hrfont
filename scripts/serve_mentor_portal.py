#!/usr/bin/env python3
"""Serve an allowlisted, read-only mentor review portal.

Routes:
  /             curated portal pages
  /assets/      selected weekly-report figures
  /bbox/        rendering/bounding-box explanation
  /e1/          E1 formal interactive evaluation
"""
from __future__ import annotations

import argparse
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
PORTAL = ROOT / "public" / "mentor_portal"
ASSETS = ROOT / "reports" / "weekly_20260905"
BBOX = ROOT / "data" / "cn2west_v2_abc_review" / "bbox_explain"
E1 = (
    ROOT
    / "runs"
    / "E1-FTV2-A-S3407"
    / "eval_formal"
    / "E1-formal-strat-20260904T041520Z"
)


class MentorHandler(SimpleHTTPRequestHandler):
    server_version = "HRFontMentor/1.0"

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def list_directory(self, path: str):  # type: ignore[override]
        self.send_error(403, "Directory listing disabled")
        return None

    def translate_path(self, path: str) -> str:
        request_path = unquote(urlsplit(path).path)
        routes = (
            ("/e1", E1),
            ("/bbox", BBOX),
            ("/assets", ASSETS),
        )
        for prefix, base in routes:
            if request_path == prefix or request_path.startswith(prefix + "/"):
                relative = request_path[len(prefix) :].lstrip("/")
                return str(self._safe_target(base, relative))

        relative = request_path.lstrip("/") or "index.html"
        return str(self._safe_target(PORTAL, relative))

    @staticmethod
    def _safe_target(base: Path, relative: str) -> Path:
        base_resolved = base.resolve()
        target = (base_resolved / relative).resolve()
        if target != base_resolved and base_resolved not in target.parents:
            return base_resolved / "__forbidden__"
        return target

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def verify_inputs() -> None:
    required = (
        PORTAL / "index.html",
        PORTAL / "rendering.html",
        ASSETS / "fig1_six_protocols.png",
        BBOX / "index.html",
        BBOX / "logic_vs_ink_zoom.png",
        E1 / "index.html",
        E1 / "browse_index.json",
        E1 / "VERIFY.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("mentor portal inputs missing:\n" + "\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19001)
    args = parser.parse_args()

    verify_inputs()
    server = ThreadingHTTPServer((args.host, args.port), MentorHandler)
    print(
        f"HR-Font mentor portal listening on http://{args.host}:{args.port} "
        f"(pid={os.getpid()})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
