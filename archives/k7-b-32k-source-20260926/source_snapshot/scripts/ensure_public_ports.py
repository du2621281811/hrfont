#!/usr/bin/env python3
"""Keep mentor-facing public HTTP ports alive outside Cursor/chat sessions."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

PY = Path("/root/miniforge3/envs/boogu/bin/python")
SCRIPTS = Path("/root/projects/hrfont/scripts")
LOGDIR = Path("/var/log/hrfont-public")
RUNDIR = Path("/var/run")

SERVICES = (
    {
        "name": "board19000",
        "port": 19000,
        "cwd": SCRIPTS,
        "argv": [str(PY), str(SCRIPTS / "serve_v100_hub.py"), "--host", "0.0.0.0", "--port", "19000"],
    },
    {
        "name": "portal19001",
        "port": 19001,
        "cwd": SCRIPTS,
        "argv": [str(PY), str(SCRIPTS / "serve_mentor_portal.py"), "--host", "0.0.0.0", "--port", "19001"],
    },
    {
        "name": "layers19003",
        "port": 19003,
        "cwd": SCRIPTS,
        "argv": [str(PY), str(SCRIPTS / "serve_p649_layers.py"), "--host", "0.0.0.0", "--port", "19003"],
    },
    {
        "name": "f03_8767",
        "port": 8767,
        "cwd": Path("/root/projects/hrfont/reports/f03_test16_strat"),
        "argv": [
            "python3",
            "-m",
            "http.server",
            "8767",
            "--bind",
            "0.0.0.0",
            "--directory",
            "/root/projects/hrfont/reports/f03_test16_strat",
        ],
    },
    {
        "name": "styleprobe8772",
        "port": 8772,
        "cwd": Path("/root/projects/hrfont/reports/style_domain_probe"),
        "argv": ["python3", "-m", "http.server", "8772", "--bind", "0.0.0.0"],
    },
)


def port_up(port: int) -> bool:
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        if not os.path.exists(path):
            continue
        with open(path) as f:
            next(f)
            for line in f:
                parts = line.split()
                listen_port = int(parts[1].split(":")[1], 16)
                if parts[3] == "0A" and listen_port == port:
                    return True
    return False


def start(svc: dict) -> bool:
    name = svc["name"]
    port = svc["port"]
    if port_up(port):
        print(f"{name} :{port} already up")
        return True
    LOGDIR.mkdir(parents=True, exist_ok=True)
    RUNDIR.mkdir(parents=True, exist_ok=True)
    log = open(LOGDIR / f"{name}.log", "ab", buffering=0)
    proc = subprocess.Popen(
        svc["argv"],
        cwd=str(svc["cwd"]),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    (RUNDIR / f"hrfont-{name}.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
    for _ in range(20):
        time.sleep(0.25)
        if port_up(port):
            print(f"{name} started :{port} pid={proc.pid}")
            return True
        if proc.poll() is not None:
            break
    print(f"{name} FAILED :{port} pid={proc.pid} rc={proc.poll()}", file=sys.stderr)
    tail = (LOGDIR / f"{name}.log").read_bytes()[-800:]
    if tail:
        sys.stderr.buffer.write(tail + b"\n")
    return False


def main() -> int:
    ok = True
    for svc in SERVICES:
        ok = start(svc) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
