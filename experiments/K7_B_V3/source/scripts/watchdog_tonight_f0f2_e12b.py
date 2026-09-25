#!/usr/bin/env python3
"""Session-independent overnight supervisor for F0/F2-CLEAN + E12-b.

Pipeline (GPU0 serial):
  F0-CLEAN 100k → scan best → Es/Ec cache → F2 smoke → F2-CLEAN 80k

Parallel (GPU1):
  E12-b membership (after phi best.pt)

Policy:
  - Never overwrite dirty F0/F2/Es/Ec or v51 E12 runs.
  - Never use rebuild_g0_caches.py.
  - Do not touch Group G / V100 jobs.
  - Network blips ignored — local disk + GPU only.
  - Adopt running PIDs; relaunch only when dead and not done.
  - Survive Cursor/SSH disconnect (launch with setsid/nohup).

Writes:
  reports/watchdog_tonight_20260914/status.json
  reports/watchdog_tonight_20260914/incidents.jsonl
  reports/watchdog_tonight_20260914/MORNING.md
  reports/e12_b/STATUS.md (refreshed)
  reports/f0f2_clean_v0913/STATUS.md (via status script)
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
WD = ROOT / "reports/watchdog_tonight_20260914"
INCIDENTS = WD / "incidents.jsonl"
STATUS = WD / "status.json"
PIDFILE = WD / "watchdog.pid"
MORNING = WD / "MORNING.md"
STOP = WD / "STOP"
POLL_S = 90

F0_ID = "F0-CLEAN-V0913-A-S3407"
F2_ID = "F2-CLEAN-V0913-A-S3407"
F0_RUN = ROOT / "runs" / F0_ID
F2_RUN = ROOT / "runs" / F2_ID
F0_VARIANT = ROOT / "code/variants/cn2west_f0_rsifree/FontDiffuser"
F123_VARIANT = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"
CLEAN_MAP = ROOT / "manifests/v0913_clean"
CACHE = ROOT / "artifacts/f0_clean_v0913"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
SPLIT = ROOT / "manifests/split_v3_228_16_16.json"
P1 = ROOT / "code/official/FontDiffuser/ckpt"

PHI_RUN = ROOT / "runs/e12_phi_s2_b_s3407"
MEM_RUN = ROOT / "runs/e12_membership_b_s3407"
MEM_CFG = ROOT / "configs/e12_membership_b_s3407.yaml"
E12_CACHE = ROOT / "artifacts/e12/cache_v0913_b"

LOG_F0 = ROOT / "reports/f0f2_clean_v0913/logs/train_f0.console.log"
LOG_F2 = ROOT / "reports/f0f2_clean_v0913/logs/train_f2.console.log"
LOG_SCAN = ROOT / "reports/f0f2_clean_v0913/logs/scan_f0.console.log"
LOG_CACHE = ROOT / "reports/f0f2_clean_v0913/logs/cache.console.log"
LOG_SMOKE = ROOT / "reports/f0f2_clean_v0913/logs/smoke_f2.console.log"
LOG_MEM = ROOT / "reports/e12_b/membership.console.log"

MARK_SCAN = WD / "mark_scan_ok"
MARK_CACHE = WD / "mark_cache_ok"
MARK_SMOKE = WD / "mark_smoke_f2_ok"
LAUNCHED_SCAN = WD / "launched_scan"
LAUNCHED_CACHE = WD / "launched_cache"
LAUNCHED_SMOKE = WD / "launched_smoke_f2"
LAUNCHED_MEM = WD / "launched_membership"

GPU_MAIN = 0  # F0/F2/cache
# E12 uses physical cuda:1 via yaml (no CUDA_VISIBLE_DEVICES remap)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_incident(kind: str, detail: dict | None = None) -> None:
    WD.mkdir(parents=True, exist_ok=True)
    rec = {"ts": utcnow(), "kind": kind, **(detail or {})}
    with INCIDENTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[INCIDENT] {kind} {detail or {}}", flush=True)


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
        out = subprocess.check_output(["pgrep", "-af", pattern], text=True).strip()
    except subprocess.CalledProcessError:
        return None
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1]
        if "watchdog_tonight_f0f2_e12b" in cmd:
            continue
        if "pgrep" in cmd:
            continue
        return pid
    return None


def launch(cmd: list[str], cwd: Path, log_path: Path, *, cuda_visible: str | None) -> int:
    env = os.environ.copy()
    if cuda_visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
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


def read_f0_step() -> int:
    step = 0
    log = F0_RUN / "fontdiffuser_training.log"
    if log.is_file():
        try:
            for line in log.read_text(errors="ignore").splitlines()[-50:]:
                if "Global Step" in line:
                    try:
                        step = max(step, int(line.split("Global Step")[1].split()[0]))
                    except Exception:
                        pass
        except Exception:
            pass
    for p in F0_RUN.glob("global_step_*"):
        try:
            step = max(step, int(p.name.split("_")[-1]))
        except Exception:
            pass
    # Prefer directory markers; avoid torch.load on every poll
    return step


def read_f2_step() -> int:
    step = 0
    last = F2_RUN / "last_state" / "trainer_state.pt"
    if last.is_file():
        try:
            import torch

            blob = torch.load(last, map_location="cpu", weights_only=False)
            step = max(step, int(blob.get("global_step", blob.get("step", 0))))
        except Exception:
            pass
    for p in F2_RUN.glob("global_step_*"):
        try:
            step = max(step, int(p.name.split("_")[-1]))
        except Exception:
            pass
    hb = F2_RUN / "heartbeat.json"
    if hb.is_file():
        try:
            step = max(step, int(json.loads(hb.read_text()).get("step", 0)))
        except Exception:
            pass
    return step


def f0_done() -> bool:
    done = F0_RUN / "DONE.json"
    if done.is_file():
        try:
            d = json.loads(done.read_text())
            return d.get("status") == "completed" and int(d.get("global_step", 0)) >= 100000
        except Exception:
            return False
    return (F0_RUN / "global_step_100000" / "unet.pth").is_file()


def f2_done() -> bool:
    done = F2_RUN / "DONE.json"
    if done.is_file():
        try:
            d = json.loads(done.read_text())
            return d.get("status") == "completed" and int(d.get("global_step", 0)) >= 80000
        except Exception:
            return False
    return (F2_RUN / "global_step_80000" / "unet.pth").is_file()


def f0_resume_dir() -> Path | None:
    last = F0_RUN / "last_state" / "trainer_state.pt"
    if last.is_file():
        return F0_RUN / "last_state"
    cks = sorted(F0_RUN.glob("global_step_*"), key=lambda p: int(p.name.split("_")[-1]))
    for p in reversed(cks):
        if (p / "trainer_state.pt").is_file() or (p / "unet.pth").is_file():
            return p
    return None


def f2_resume_dir() -> Path | None:
    last = F2_RUN / "last_state"
    if (last / "trainer_state.pt").is_file() or (last / "unet.pth").is_file():
        return last
    cks = sorted(F2_RUN.glob("global_step_*"), key=lambda p: int(p.name.split("_")[-1]))
    for p in reversed(cks):
        if (p / "trainer_state.pt").is_file() or (p / "unet.pth").is_file():
            return p
    return None


def ensure_f0() -> dict:
    if f0_done():
        return {"job": "F0", "state": "done", "step": read_f0_step()}
    pid = pgrep_cmd(rf"train\.py .*--experience_name {F0_ID}|train\.py .*--output_dir .*{F0_ID}")
    if pid and alive(pid):
        return {"job": "F0", "state": "running", "pid": pid, "step": read_f0_step()}
    # Also match via launch wrapper child
    pid2 = pgrep_cmd(rf"FontDiffuser/train\.py .*{F0_ID}")
    if pid2 and alive(pid2):
        return {"job": "F0", "state": "running", "pid": pid2, "step": read_f0_step()}

    if not F0_RUN.is_dir():
        # Fresh launch via orchestrator (creates dir)
        cmd = [
            PY, str(ROOT / "scripts/run_f0f2_clean_v0913.py"), "train-f0", "--gpu", str(GPU_MAIN),
        ]
        pid = launch(cmd, ROOT, LOG_F0, cuda_visible=None)
        log_incident("launch_f0_fresh", {"pid": pid})
        return {"job": "F0", "state": "launched_fresh", "pid": pid}

    rd = f0_resume_dir()
    if rd is None and read_f0_step() == 0:
        # Dir exists but no ckpt yet — may have crashed at start; refuse overwrite via launcher.
        # Call train.py directly; train will start from scratch into existing dir.
        log_incident("f0_crash_early_no_ckpt", {"step": 0})
    cmd = [
        PY, str(F0_VARIANT / "train.py"),
        "--seed", "3407",
        "--experience_name", F0_ID,
        "--output_dir", str(F0_RUN),
        "--data_root", str(DATA),
        "--v0913_clean_map", str(CLEAN_MAP),
        "--resolution", "96",
        "--style_image_size", "96",
        "--content_image_size", "96",
        "--train_batch_size", "8",
        "--gradient_accumulation_steps", "1",
        "--max_train_steps", "100000",
        "--learning_rate", "1e-05",
        "--lr_scheduler", "linear",
        "--lr_warmup_steps", "5000",
        "--phase_1_ckpt_dir", str(P1),
        "--mixed_precision", "fp16",
        "--ckpt_interval", "5000",
        "--log_interval", "100",
        "--drop_prob", "0.1",
        "--perceptual_coefficient", "0.01",
        "--offset_coefficient", "0.0",
    ]
    # train.py auto-resumes from last_state when resume_from unset
    pid = launch(cmd, F0_VARIANT, LOG_F0, cuda_visible=str(GPU_MAIN))
    log_incident("relaunch_f0", {"pid": pid, "resume": str(rd) if rd else "auto", "step": read_f0_step()})
    return {"job": "F0", "state": "relaunched", "pid": pid, "resume": str(rd) if rd else "auto"}


def ensure_scan() -> dict:
    if not f0_done():
        return {"job": "scan", "state": "waiting_f0"}
    if (F0_RUN / "best" / "unet.pth").is_file():
        MARK_SCAN.write_text(utcnow() + "\n")
        return {"job": "scan", "state": "done"}
    if pgrep_cmd(r"scan_f0_val_loss\.py|run_f0f2_clean_v0913\.py scan-f0"):
        return {"job": "scan", "state": "running"}
    if pgrep_cmd(rf"train\.py .*{F0_ID}"):
        return {"job": "scan", "state": "waiting_f0_exit"}
    if LAUNCHED_SCAN.is_file():
        # Previous attempt finished without best — log and retry once after clearing launch mark age
        age = time.time() - LAUNCHED_SCAN.stat().st_mtime
        if age < 600:
            return {"job": "scan", "state": "waiting_scan_finish", "age_s": int(age)}
        log_incident("scan_retry", {"age_s": int(age)})
        LAUNCHED_SCAN.unlink(missing_ok=True)
    cmd = [PY, str(ROOT / "scripts/run_f0f2_clean_v0913.py"), "scan-f0", "--gpu", str(GPU_MAIN)]
    pid = launch(cmd, ROOT, LOG_SCAN, cuda_visible=None)
    LAUNCHED_SCAN.write_text(f"{utcnow()} pid={pid}\n")
    log_incident("launch_scan_f0", {"pid": pid})
    return {"job": "scan", "state": "launched", "pid": pid}


def ensure_cache() -> dict:
    if not f0_done() or not (F0_RUN / "best" / "unet.pth").is_file():
        return {"job": "cache", "state": "waiting_scan"}
    es_m = CACHE / "es_spatial" / "manifest.json"
    ec_m = CACHE / "ec_multiscale" / "manifest.json"
    if es_m.is_file() and ec_m.is_file():
        MARK_CACHE.write_text(utcnow() + "\n")
        return {"job": "cache", "state": "done"}
    if pgrep_cmd(r"rebuild_f0_clean_v0913_caches\.py|run_f0f2_clean_v0913\.py cache"):
        return {"job": "cache", "state": "running"}
    if pgrep_cmd(rf"train\.py .*{F0_ID}") or pgrep_cmd(r"scan_f0_val_loss\.py"):
        return {"job": "cache", "state": "waiting_gpu0"}
    if LAUNCHED_CACHE.is_file():
        age = time.time() - LAUNCHED_CACHE.stat().st_mtime
        # Ec build can take many hours
        if age < 6 * 3600:
            return {"job": "cache", "state": "waiting_cache_finish", "age_s": int(age)}
        log_incident("cache_retry", {"age_s": int(age)})
        LAUNCHED_CACHE.unlink(missing_ok=True)
    cmd = [PY, str(ROOT / "scripts/run_f0f2_clean_v0913.py"), "cache", "--gpu", str(GPU_MAIN)]
    pid = launch(cmd, ROOT, LOG_CACHE, cuda_visible=None)
    LAUNCHED_CACHE.write_text(f"{utcnow()} pid={pid}\n")
    log_incident("launch_cache", {"pid": pid})
    return {"job": "cache", "state": "launched", "pid": pid}


def ensure_smoke_f2() -> dict:
    es_m = CACHE / "es_spatial" / "manifest.json"
    ec_m = CACHE / "ec_multiscale" / "manifest.json"
    if not (es_m.is_file() and ec_m.is_file()):
        return {"job": "smoke_f2", "state": "waiting_cache"}
    if MARK_SMOKE.is_file() or F2_RUN.is_dir() or f2_done():
        MARK_SMOKE.write_text(utcnow() + "\n")
        return {"job": "smoke_f2", "state": "done"}
    if pgrep_cmd(r"run_f0f2_clean_v0913\.py smoke-f2"):
        return {"job": "smoke_f2", "state": "running"}
    if pgrep_cmd(r"rebuild_f0_clean_v0913_caches\.py"):
        return {"job": "smoke_f2", "state": "waiting_cache"}
    # Detect completed smoke run dirs
    smokes = list((ROOT / "runs").glob("smoke*F2*")) + list((ROOT / "runs").glob("smoke-F2*"))
    for s in smokes:
        if (s / "DONE.json").is_file() or list(s.glob("**/unet.pth")) or list(s.glob("global_step_*")):
            MARK_SMOKE.write_text(utcnow() + f" from={s.name}\n")
            return {"job": "smoke_f2", "state": "done", "from": s.name}
    if LAUNCHED_SMOKE.is_file():
        age = time.time() - LAUNCHED_SMOKE.stat().st_mtime
        if age < 1800:
            return {"job": "smoke_f2", "state": "waiting_smoke_finish", "age_s": int(age)}
        log_incident("smoke_f2_retry", {"age_s": int(age)})
        LAUNCHED_SMOKE.unlink(missing_ok=True)
    cmd = [PY, str(ROOT / "scripts/run_f0f2_clean_v0913.py"), "smoke-f2", "--gpu", str(GPU_MAIN)]
    pid = launch(cmd, ROOT, LOG_SMOKE, cuda_visible=None)
    LAUNCHED_SMOKE.write_text(f"{utcnow()} pid={pid}\n")
    log_incident("launch_smoke_f2", {"pid": pid})
    return {"job": "smoke_f2", "state": "launched", "pid": pid}

def ensure_f2() -> dict:
    if f2_done():
        return {"job": "F2", "state": "done", "step": read_f2_step()}
    es = CACHE / "es_spatial"
    ec = CACHE / "ec_multiscale"
    parent = F0_RUN / "best"
    if not (parent / "unet.pth").is_file() or not (es / "manifest.json").is_file() or not (ec / "manifest.json").is_file():
        return {"job": "F2", "state": "waiting_preconditions"}
    if not F2_RUN.is_dir() and not MARK_SMOKE.is_file():
        return {"job": "F2", "state": "waiting_smoke"}

    pid = pgrep_cmd(rf"train\.py .*--experience_name {F2_ID}|train\.py .*--output_dir .*{F2_ID}")
    if pid and alive(pid):
        return {"job": "F2", "state": "running", "pid": pid, "step": read_f2_step()}
    pid2 = pgrep_cmd(rf"FontDiffuser/train\.py .*{F2_ID}")
    if pid2 and alive(pid2):
        return {"job": "F2", "state": "running", "pid": pid2, "step": read_f2_step()}

    # Don't steal GPU while cache/smoke/F0 still running
    if pgrep_cmd(r"rebuild_f0_clean_v0913_caches\.py|scan_f0_val_loss\.py|run_f0f2_clean_v0913\.py smoke-f2") or pgrep_cmd(rf"train\.py .*{F0_ID}"):
        return {"job": "F2", "state": "waiting_gpu0"}

    if not F2_RUN.is_dir():
        cmd = [
            PY, str(ROOT / "scripts/run_f0f2_clean_v0913.py"), "train-f2", "--gpu", str(GPU_MAIN),
        ]
        pid = launch(cmd, ROOT, LOG_F2, cuda_visible=None)
        log_incident("launch_f2_fresh", {"pid": pid})
        return {"job": "F2", "state": "launched_fresh", "pid": pid}

    rd = f2_resume_dir()
    if rd is None:
        log_incident("f2_no_resume", {"step": read_f2_step()})
        return {"job": "F2", "state": "blocked_no_ckpt"}
    cmd = [
        PY, str(F123_VARIANT / "train.py"),
        "--arm", "F2",
        "--rsi_source", "delta",
        "--no-support",
        "--support_drop", "0.2",
        "--support_k", "8",
        "--source_drop", "0.25",
        "--seed", "3407",
        "--experience_name", F2_ID,
        "--output_dir", str(F2_RUN),
        "--data_root", str(DATA),
        "--split_manifest", str(SPLIT),
        "--v0913_clean_map", str(CLEAN_MAP),
        "--es_cache_path", str(es),
        "--es_local_cache_path", str(CACHE / "es_local"),
        "--ec_cache_path", str(ec),
        "--phase_1_ckpt_dir", str(parent),
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
        "--resume_from", str(rd),
    ]
    pid = launch(cmd, F123_VARIANT, LOG_F2, cuda_visible=str(GPU_MAIN))
    log_incident("relaunch_f2", {"pid": pid, "resume": str(rd), "step": read_f2_step()})
    return {"job": "F2", "state": "relaunched", "pid": pid, "resume": str(rd)}


def membership_done() -> bool:
    return (MEM_RUN / "test_metrics.json").is_file() and (MEM_RUN / "best.pt").is_file()


def ensure_membership() -> dict:
    if membership_done():
        return {"job": "E12_membership", "state": "done"}
    if not (PHI_RUN / "best.pt").is_file():
        return {"job": "E12_membership", "state": "waiting_phi"}
    if not (E12_CACHE / "manifest.json").is_file() and not any(E12_CACHE.glob("*")):
        return {"job": "E12_membership", "state": "waiting_cache"}
    pid = pgrep_cmd(r"train_membership\.py --config .*e12_membership_b_s3407")
    if pid and alive(pid):
        step = 0
        curves = MEM_RUN / "curves.jsonl"
        if curves.is_file():
            try:
                last = curves.read_text().strip().splitlines()[-1]
                step = int(json.loads(last).get("step", 0))
            except Exception:
                pass
        return {"job": "E12_membership", "state": "running", "pid": pid, "step": step}
    if LAUNCHED_MEM.is_file() and MEM_RUN.is_dir() and not membership_done():
        age = time.time() - LAUNCHED_MEM.stat().st_mtime
        # Membership ~1–3h; allow finish window before retry
        if age < 4 * 3600 and (MEM_RUN / "curves.jsonl").is_file():
            # Process died mid-run — retry after short grace
            if age < 180:
                return {"job": "E12_membership", "state": "waiting_exit", "age_s": int(age)}
        if age < 180:
            return {"job": "E12_membership", "state": "waiting_start", "age_s": int(age)}
        log_incident("membership_restart_partial", {"dir": str(MEM_RUN), "age_s": int(age)})
    cmd = [
        PY, "-u", str(ROOT / "scripts/eval_framework/train_membership.py"),
        "--config", str(MEM_CFG),
    ]
    # Match phi: use yaml device cuda:1, do not remap CUDA_VISIBLE_DEVICES
    pid = launch(cmd, ROOT / "scripts/eval_framework", LOG_MEM, cuda_visible=None)
    LAUNCHED_MEM.write_text(f"{utcnow()} pid={pid}\n")
    log_incident("launch_membership", {"pid": pid})
    return {"job": "E12_membership", "state": "launched", "pid": pid}


def refresh_e12_status() -> None:
    out = ROOT / "reports/e12_b"
    out.mkdir(parents=True, exist_ok=True)
    auc_points = []
    best = None
    curves = PHI_RUN / "curves.jsonl"
    if curves.is_file():
        for line in curves.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "val_auc" in r:
                auc_points.append((r.get("step"), r["val_auc"]))
        if auc_points:
            best = max(auc_points, key=lambda x: x[1])
    mem_state = "done" if membership_done() else ("running" if pgrep_cmd(r"train_membership\.py --config .*e12_membership_b") else "pending")
    val_m = mem_t = None
    if (MEM_RUN / "val_metrics.json").is_file():
        try:
            val_m = json.loads((MEM_RUN / "val_metrics.json").read_text())
        except Exception:
            pass
    if (MEM_RUN / "test_metrics.json").is_file():
        try:
            mem_t = json.loads((MEM_RUN / "test_metrics.json").read_text())
        except Exception:
            pass
    n_fonts = "?"
    man = E12_CACHE / "manifest.json"
    if man.is_file():
        try:
            n_fonts = len(json.loads(man.read_text()).get("fonts", []))
        except Exception:
            pass
    lines = [
        "# E12-b STATUS",
        f"- updated: {utcnow()}",
        f"- cache: artifacts/e12/cache_v0913_b (fonts≈{n_fonts})",
        f"- phi run: runs/e12_phi_s2_b_s3407",
        f"- phi latest val_auc points: {auc_points[-8:]}",
        f"- phi curve best: {best}",
        f"- phi best.pt: {(PHI_RUN / 'best.pt').is_file()}",
        f"- membership: {mem_state}",
        f"- membership val: {val_m}",
        f"- membership test: {mem_t}",
        "- contract: reports/EXPERIMENT_E12B_V0913_20260914.md",
        "",
    ]
    (out / "STATUS.md").write_text("\n".join(lines), encoding="utf-8")


def refresh_f0f2_status() -> None:
    try:
        subprocess.run(
            [PY, str(ROOT / "scripts/status_f0f2_clean_v0913.py")],
            cwd=str(ROOT),
            capture_output=True,
            timeout=120,
        )
    except Exception as e:
        log_incident("status_f0f2_refresh_fail", {"error": str(e)})


def write_morning(jobs: dict) -> None:
    lines = [
        "# 明早看板 · F0/F2-CLEAN + E12-b",
        f"- 更新：`{utcnow()}`",
        f"- 监督器：`reports/watchdog_tonight_20260914/`（`status.json` / `incidents.jsonl`）",
        "",
        "## 阶段快照",
    ]
    for k, v in jobs.items():
        lines.append(f"- **{k}**: `{json.dumps(v, ensure_ascii=False)}`")
    lines += [
        "",
        "## 入口文件",
        "- `reports/f0f2_clean_v0913/STATUS.md`",
        "- `reports/e12_b/STATUS.md`",
        "- F0 log: `reports/f0f2_clean_v0913/logs/train_f0.console.log`",
        "- F2 log: `reports/f0f2_clean_v0913/logs/train_f2.console.log`",
        "- membership: `reports/e12_b/membership.console.log`",
        "- 事故：`reports/watchdog_tonight_20260914/incidents.jsonl`",
        "",
        "## 完成判据",
        "- F0: `runs/F0-CLEAN-V0913-A-S3407/DONE.json` + `best/`",
        "- Cache: `artifacts/f0_clean_v0913/{es_spatial,ec_multiscale}/manifest.json`",
        "- F2: `runs/F2-CLEAN-V0913-A-S3407/DONE.json`（80k）",
        "- E12-b: `runs/e12_membership_b_s3407/{best.pt,val_metrics.json,test_metrics.json}`",
        "",
        "## 注意",
        "- 脏臂未动；eval_f03 仍硬编码脏 Es/Ec（干净对比图等 F2 后再接）",
        "- 网络/会话断开不影响本监督器（setsid）",
        "",
    ]
    MORNING.write_text("\n".join(lines), encoding="utf-8")
    # Mirror into tonight execution note
    exec_path = ROOT / "reports/EXECUTION_TONIGHT_20260914.md"
    try:
        base = exec_path.read_text(encoding="utf-8") if exec_path.is_file() else ""
        marker = "\n\n## 自动监督（通宵）\n"
        block = marker + "\n".join(lines) + "\n"
        if marker in base:
            base = base.split(marker)[0] + block
        else:
            base = base.rstrip() + block
        exec_path.write_text(base, encoding="utf-8")
    except Exception as e:
        log_incident("write_execution_fail", {"error": str(e)})


def all_done(jobs: dict) -> bool:
    return (
        jobs.get("F0", {}).get("state") == "done"
        and jobs.get("scan", {}).get("state") == "done"
        and jobs.get("cache", {}).get("state") == "done"
        and jobs.get("F2", {}).get("state") == "done"
        and jobs.get("E12_membership", {}).get("state") == "done"
    )


def tick() -> dict:
    jobs = {
        "F0": ensure_f0(),
        "scan": ensure_scan(),
        "cache": ensure_cache(),
        "smoke_f2": ensure_smoke_f2(),
        "F2": ensure_f2(),
        "E12_membership": ensure_membership(),
    }
    refresh_e12_status()
    refresh_f0f2_status()
    write_morning(jobs)
    write_status({"phase": "running", "jobs": jobs, "all_done": all_done(jobs)})
    return jobs


def main() -> int:
    WD.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()) + "\n")
    log_incident("watchdog_start", {"pid": os.getpid(), "poll_s": POLL_S})
    while True:
        if STOP.is_file():
            log_incident("watchdog_stop_file", {})
            break
        try:
            jobs = tick()
            if all_done(jobs):
                write_status({"phase": "completed", "jobs": jobs, "all_done": True})
                log_incident("watchdog_complete", {})
                write_morning(jobs)
                break
        except Exception as e:
            log_incident("watchdog_exception", {"error": str(e), "tb": traceback.format_exc()})
        time.sleep(POLL_S)
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    raise SystemExit(main())
