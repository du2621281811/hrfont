#!/usr/bin/env python3
"""Serve the charset picker on http://127.0.0.1:8766/"""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "reports" / "charset_picker"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"charset picker → http://{args.host}:{args.port}/", flush=True)
    print(f"serving {ROOT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
