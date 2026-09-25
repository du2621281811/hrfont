#!/usr/bin/env python3
"""Session-independent watchdog for F1 + F3b + E12 membership.

Policy (frozen):
  - Never degrade batch/lr/steps/seed.
  - Never create STOP files (except the one-shot cancel of the old stroke-bank F3b,
    which is done out-of-band; this process must not relaunch that run).
  - Adopt already-running train PIDs; only relaunch when the process is dead
    and DONE/test_metrics is missing.
  - Resume F1 from stopped_step if present else last_state.
  - New F3b (`f3b_topology_ownfont_s3407`, topology bank, own-font Ec) launches
    only after F1@80k is DONE. Do not resume the old stroke-bank run.
  - Survive Cursor/SSH disconnects (caller must launch with setsid/nohup).

Network blips are ignored — all work is local disk + GPU.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
PY = "/root/miniforge3/envs/boogu/bin/python"
VARIANT = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"
PARENT = ROOT / "runs/F0-RSIFREE-FT-A-S3407/best"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
SPLIT = ROOT / "manifests/split_v3_228_16_16.json"
ES = ROOT / "artifacts/f0/es_spatial_f0"
EC = ROOT / "artifacts/f0/ec_multiscale_f0"
BANK = ROOT / "artifacts/f0/support_bank_f3b_topology.json"
OLD_F3B = ROOT / "runs/f3b_joint_crossbank_s3407"
NEW_F3B = ROOT / "runs/f3b_topology_ownfont_s3407"
WD = ROOT / "reports/watchdog_f1_f3b_e12"
INCIDENTS = WD / "incidents.jsonl"
STATUS = WD / "status.json"
PIDFILE = WD / "watchdog.pid"
POLL_S = 60

COMMON = [
    "--data_root", str(DATA),
    "--split_manifest", str(SPLIT),
    "--es_cache_path", str(ES),
    "--ec_cache_path", str(EC),
    "--phase_1_ckpt_dir", str(PARENT),
    "--resolution", "96",
    "--style_image_size", "96",
    "--content_image_size", "96",
    "--train_batch_size", "8",
    "--gradient_accumulation_steps", "1",
    "--max_train_steps", "80000",
    "--learning_rate", "1e-05",
    "--lr_scheduler", "linear",
    "--lr_warmup_steps", "5000",
    "--mixed_precision", "fp16",
    "--ckpt_interval", "5000",
    "--state_interval", "1000",
    "--log_interval", "100",
    "--drop_prob", "0.1",
    "--perceptual_coefficient", "0.01",
    "--offset_coefficient", "0.5",
    "--parity_check",
]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_incident(kind: str, detail: dict) -> None:
    WD.mkdir(parents=True, exist_ok=True)
    rec = {"ts": utcnow(), "kind": kind, **detail}
    with INCIDENTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[INCIDENT] {kind} {detail}", flush=True)


def write_status(payload: dict) -> None:
    WD.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({"ts": utcnow(), **payload}, ensure_ascii=False, indent=2) + "\n")


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def pgrep_cmd(pattern: str) -> int | None:
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True).strip()
    except subprocess.CalledProcessError:
        return None
    for line in out.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        # Ignore this watchdog and shells that embed the pattern in argv.
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except Exception:
            continue
        if "watchdog_f1_f3b_e12" in cmd:
            continue
        if "pgrep" in cmd:
            continue
        return pid
    return None


def read_step(run_dir: Path) -> int:
    best = 0
    for name in ("stopped_step", "last_state"):
        p = run_dir / name / "trainer_state.pt"
        if not p.is_file():
            continue
        try:
            import torch
            st = torch.load(str(p), map_location="cpu", weights_only=False)
            best = max(best, int(st.get("step", 0)))
        except Exception:
            pass
    for p in run_dir.glob("global_step_*"):
        try:
            best = max(best, int(p.name.split("_")[-1]))
        except ValueError:
            pass
    hb = run_dir / "heartbeat.json"
    if hb.is_file():
        try:
            best = max(best, int(json.loads(hb.read_text()).get("step") or 0))
        except Exception:
            pass
    return best


def resume_dir(run_dir: Path) -> Path | None:
    stopped = run_dir / "stopped_step" / "trainer_state.pt"
    last = run_dir / "last_state" / "trainer_state.pt"
    candidates = []
    for p in (stopped, last):
        if p.is_file():
            candidates.append(p.parent)
    if not candidates:
        return None
    # Prefer higher step; tie-break newer mtime.
    scored = []
    for d in candidates:
        try:
            import torch
            st = torch.load(str(d / "trainer_state.pt"), map_location="cpu", weights_only=False)
            step = int(st.get("step", 0))
        except Exception:
            step = 0
        scored.append((step, (d / "trainer_state.pt").stat().st_mtime, d))
    scored.sort(reverse=True)
    return scored[0][2]


def f1_done() -> bool:
    p = ROOT / "runs/F1-OFFRSI-A-S3407/DONE.json"
    if not p.is_file():
        return False
    try:
        d = json.loads(p.read_text())
        return int(d.get("global_step", 0)) >= 80000 and d.get("status") == "completed"
    except Exception:
        return False


def f3b_done() -> bool:
    p = NEW_F3B / "DONE.json"
    if not p.is_file():
        return False
    try:
        d = json.loads(p.read_text())
        return int(d.get("global_step", 0)) >= 80000 and d.get("status") == "completed"
    except Exception:
        return False


def old_f3b_alive() -> int | None:
    pid = pgrep_cmd(r"train\.py --arm F3b .*f3b_joint_crossbank_s3407")
    return pid if pid and alive(pid) else None


def membership_done() -> bool:
    out = ROOT / "runs/e12_membership_v51_train228_s3407"
    return (out / "test_metrics.json").is_file() and (out / "best.pt").is_file()


def launch(cmd: list[str], cwd: Path, gpu: int, log_path: Path) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n===== WATCHDOG LAUNCH {utcnow()} =====\n")
        logf.write("CMD " + " ".join(cmd) + "\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc.pid


def ensure_f1() -> dict:
    run = ROOT / "runs/F1-OFFRSI-A-S3407"
    if f1_done():
        return {"job": "F1", "state": "done", "step": read_step(run)}
    # Remove accidental STOP so resume is not immediately halted.
    stop = run / "STOP"
    if stop.exists():
        stop.unlink()
        log_incident("removed_stop", {"job": "F1"})
    pid = pgrep_cmd(r"train\.py --arm F1 .*F1-OFFRSI-A-S3407")
    if pid and alive(pid):
        return {"job": "F1", "state": "running", "pid": pid, "step": read_step(run)}
    rd = resume_dir(run)
    if rd is None:
        log_incident("f1_no_resume", {"step": 0})
        return {"job": "F1", "state": "blocked_no_ckpt"}
    cmd = [
        PY, str(VARIANT / "train.py"),
        "--arm", "F1", "--rsi_source", "official", "--no-support",
        "--support_drop", "0.2", "--support_k", "8", "--source_drop", "0.25",
        "--seed", "3407",
        "--experience_name", "F1-OFFRSI-A-S3407",
        "--output_dir", str(run),
        *COMMON,
        "--resume_from", str(rd),
    ]
    pid = launch(cmd, VARIANT, gpu=2, log_path=ROOT / "logs/f1/F1-OFFRSI-A-S3407.log")
    log_incident("relaunch_f1", {"pid": pid, "resume": str(rd), "step": read_step(run)})
    return {"job": "F1", "state": "relaunched", "pid": pid, "resume": str(rd)}


def ensure_f3b() -> dict:
    """New topology F3b after F1; never resume the old stroke-bank run."""
    old_pid = old_f3b_alive()
    if old_pid:
        return {
            "job": "F3b",
            "state": "waiting_old_stop",
            "old_pid": old_pid,
            "old_step": read_step(OLD_F3B),
            "note": "stroke-bank F3b still alive; not relaunching it",
        }
    if not f1_done():
        return {
            "job": "F3b",
            "state": "waiting_f1",
            "f1_step": read_step(ROOT / "runs/F1-OFFRSI-A-S3407"),
            "run": str(NEW_F3B),
            "bank": str(BANK),
        }
    run = NEW_F3B
    if f3b_done():
        return {"job": "F3b", "state": "done", "step": read_step(run)}
    stop = run / "STOP"
    if stop.exists():
        stop.unlink()
        log_incident("removed_stop", {"job": "F3b", "run": str(run)})
    pid = pgrep_cmd(r"train\.py --arm F3b .*f3b_topology_ownfont_s3407")
    if pid and alive(pid):
        return {"job": "F3b", "state": "running", "pid": pid, "step": read_step(run)}
    rd = resume_dir(run)
    cmd = [
        PY, str(VARIANT / "train.py"),
        "--arm", "F3b", "--rsi_source", "delta", "--support",
        "--support_drop", "0.2", "--support_k", "8", "--source_drop", "0.25",
        "--seed", "3407",
        "--experience_name", "f3b_topology_ownfont_s3407",
        "--output_dir", str(run),
        "--support_bank", str(BANK),
        *COMMON,
    ]
    if rd is None:
        pid = launch(cmd, VARIANT, gpu=0, log_path=ROOT / "logs/f3b/f3b_topology.log")
        log_incident("launch_f3b_topology_fresh", {"pid": pid, "bank": str(BANK)})
        return {"job": "F3b", "state": "launched_fresh", "pid": pid, "run": str(run)}
    cmd = cmd + ["--resume_from", str(rd)]
    pid = launch(cmd, VARIANT, gpu=0, log_path=ROOT / "logs/f3b/f3b_topology.log")
    log_incident("relaunch_f3b_topology", {"pid": pid, "resume": str(rd), "step": read_step(run)})
    return {"job": "F3b", "state": "relaunched", "pid": pid, "resume": str(rd)}


def ensure_membership() -> dict:
    out = ROOT / "runs/e12_membership_v51_train228_s3407"
    if membership_done():
        return {"job": "E12_membership", "state": "done"}
    phi_best = ROOT / "runs/e12_phi_s2_v51_train228_s3407/best.pt"
    if not phi_best.is_file():
        return {"job": "E12_membership", "state": "waiting_phi"}
    pid = pgrep_cmd(r"train_membership\.py --config .*e12_membership_v51_train228")
    if pid and alive(pid):
        step = 0
        curves = out / "curves.jsonl"
        if curves.is_file():
            try:
                last = curves.read_text().strip().splitlines()[-1]
                step = int(json.loads(last).get("step", 0))
            except Exception:
                pass
        return {"job": "E12_membership", "state": "running", "pid": pid, "step": step}
    # Membership has no resume — restart from scratch if crashed mid-run.
    cfg = ROOT / "configs/e12_membership_v51_train228_s3407.yaml"
    cmd = [PY, "-u", str(ROOT / "scripts/eval_framework/train_membership.py"), "--config", str(cfg)]
    pid = launch(cmd, ROOT / "scripts/eval_framework", gpu=1,
                 log_path=ROOT / "logs/e12/membership_v51_train228_s3407.log")
    log_incident("relaunch_membership", {"pid": pid})
    return {"job": "E12_membership", "state": "relaunched", "pid": pid}


def all_done() -> bool:
    return f1_done() and f3b_done() and membership_done()


def main() -> int:
    WD.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()) + "\n")
    log_incident("watchdog_start", {"pid": os.getpid(), "poll_s": POLL_S})
    while not all_done():
        try:
            snap = {
                "F1": ensure_f1(),
                "F3b": ensure_f3b(),
                "E12_membership": ensure_membership(),
            }
            write_status({"jobs": snap})
        except Exception as e:
            log_incident("watchdog_exception", {"error": str(e), "tb": traceback.format_exc()})
        time.sleep(POLL_S)
    write_status({"phase": "completed", "jobs": {
        "F1": ensure_f1(), "F3b": ensure_f3b(), "E12_membership": ensure_membership()
    }})
    log_incident("watchdog_complete", {})
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    raise SystemExit(main())
