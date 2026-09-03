#!/usr/bin/env python3
"""Supervise formal v2 Stage A until 80k + official eval, then auto Stage B.

Writes:
  reports/hrfont_overnight/SUPERVISOR.json
  reports/hrfont_overnight/formal_eval/metrics_step_*.json
  reports/hrfont_overnight/NEXT_STEPS_FORMAL.md
  reports/hrfont_overnight/STAGE_B_PLAN.json
  reports/hrfont_overnight/NEXT_STEPS_COMPLETE.md (A+B done)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
REP = ROOT / "reports/hrfont_overnight"
FORMAL_OUT = ROOT / "runs/e2_stageA96_formal"
STAGEB_OUT = ROOT / "runs/e2_stageB96"
PY = "/root/miniforge3/envs/boogu/bin/python"
MAX_STEPS = 80_000
STAGE_B_DEFAULT = 30_000
EVAL_MILESTONES = [10_000, 25_000, 50_000, 70_000, 80_000]
POLL_SEC = 120


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    REP.mkdir(parents=True, exist_ok=True)
    with (REP / "supervisor.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_json(p: Path) -> dict:
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save_supervisor(st: dict) -> None:
    st["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (REP / "SUPERVISOR.json").write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")


def current_step() -> int:
    st = load_json(REP / "STATUS_FORMAL.json")
    if "step" in st:
        return int(st["step"])
    last = FORMAL_OUT / "last.pt"
    if last.exists():
        try:
            import torch

            blob = torch.load(last, map_location="cpu", weights_only=False)
            return int(blob.get("step", 0))
        except Exception:
            pass
    cks = sorted(FORMAL_OUT.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if cks:
        return int(cks[-1].stem.split("_")[1])
    return 0


def train_alive() -> bool:
    r = subprocess.run(["pgrep", "-f", "hrfont_e2_stageA96_formal_train.py"], capture_output=True)
    return r.returncode == 0


def keepalive_alive() -> bool:
    r = subprocess.run(["pgrep", "-f", "hrfont_formal_keepalive.sh"], capture_output=True)
    return r.returncode == 0


def start_keepalive() -> None:
    if (REP / "STOP_FORMAL").exists():
        log("STOP_FORMAL set — not starting keepalive")
        return
    if keepalive_alive():
        return
    log("starting formal keepalive")
    subprocess.Popen(
        ["bash", str(ROOT / "scripts/hrfont_formal_keepalive.sh")],
        stdout=open(REP / "keepalive_formal.log", "a"),
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
        start_new_session=True,
    )


def resolve_ckpt_for_step(step: int) -> Path | None:
    exact = FORMAL_OUT / f"step_{step}.pt"
    if exact.exists():
        return exact
    cks = sorted(FORMAL_OUT.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    best = None
    for p in cks:
        s = int(p.stem.split("_")[1])
        if s <= step:
            best = p
    if best:
        return best
    last = FORMAL_OUT / "last.pt"
    return last if last.exists() else None


def run_official_eval(step: int, ckpt: Path) -> dict | None:
    out = REP / "formal_eval" / f"metrics_step_{step}.json"
    if out.exists():
        log(f"eval step {step} already exists — skip")
        return load_json(out)
    log(f"official eval @ step {step} ckpt={ckpt.name}")
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "3")
    env["HRFONT_STAGEA_CKPT"] = str(ckpt)
    r = subprocess.run(
        [PY, str(ROOT / "scripts/hrfont_e2_official_eval.py"), "--ckpt", str(ckpt), "--out", str(out)],
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        log(f"eval failed step={step} rc={r.returncode}\n{r.stderr[-800:]}")
        return None
    log(f"eval done step={step} -> {out}")
    return load_json(out) if out.exists() else None


def write_next_steps(st: dict) -> None:
    best = st.get("best_eval") or {}
    md = f"""# HR-Font Formal v2 — 完成 & 下一步

更新时间：{time.strftime('%Y-%m-%d %H:%M:%S')}

## Formal Stage A 结果

| 项 | 值 |
|----|-----|
| 最终 step | {st.get('final_step')} |
| 最佳 eval step | {best.get('step', '—')} |
| ft_cnstyle L1 | {best.get('ft_l1', '—')} |
| Stage A L1 | {best.get('stageA_l1', '—')} |
| Δ (ft − A) | {best.get('delta', '—')} |

协议：Demo-8 × P1 全表 · DejaVu content · DPM++ 20 · CFG 7.5 · style=「永」

## 判定

- **idea 验证**：Stage A L1 < ft_cnstyle → Δ-RSI 接线有效
- 详细：`reports/hrfont_overnight/formal_eval/`

## 下一步（自动规划）

1. **锁 best ckpt** → `runs/e2_stageA96_formal/step_<best>.pt`
2. **E3 ablation**（同协议）：零样本 / ft / StageA / 无Δ（可选）
3. **Stage B**：冻 A，训 SupportAdapter + gap 采样（§4.5）
4. **E4/E6**：gap 分层 eval
5. **更新 mentor 页**：official 栏 + 图

## 命令

```bash
# 最终评测已写入 formal_eval/metrics_step_*.json
cat reports/hrfont_overnight/SUPERVISOR.json
```
"""
    (REP / "NEXT_STEPS_FORMAL.md").write_text(md, encoding="utf-8")
    log("wrote NEXT_STEPS_FORMAL.md")


def resolve_best_stageA_ckpt(st: dict) -> Path:
    best = st.get("best_eval") or {}
    step = int(best.get("step", 0))
    if step:
        ck = resolve_ckpt_for_step(step)
        if ck and ck.exists():
            return ck
    last = FORMAL_OUT / "last.pt"
    if last.exists():
        return last
    cks = sorted(FORMAL_OUT.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return cks[-1] if cks else last


def plan_stage_b(st: dict) -> dict:
    best = st.get("best_eval") or {}
    ft = best.get("ft_l1")
    a = best.get("stageA_l1")
    ckpt = str(resolve_best_stageA_ckpt(st))
    plan = {"stageA_ckpt": ckpt, "t": time.strftime("%Y-%m-%d %H:%M:%S")}
    if ft is None or a is None:
        plan.update(max_steps=20_000, reason="无完整 eval；默认短训 Stage B")
    else:
        delta = float(ft) - float(a)
        plan["ft_l1"] = ft
        plan["stageA_l1"] = a
        plan["delta"] = delta
        if a < ft:
            plan.update(max_steps=40_000, reason=f"Stage A 胜 ft (Δ={delta:.4f}) → 全量 Stage B")
        elif a <= ft * 1.08:
            plan.update(max_steps=25_000, reason=f"Stage A 接近 ft ({a:.4f} vs {ft:.4f}) → 标准 Stage B")
        else:
            plan.update(max_steps=15_000, reason=f"Stage A 落后 ft → 短训 Stage B 验证支撑分支")
    (REP / "STAGE_B_PLAN.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Stage B plan: {plan['reason']} max={plan['max_steps']}")
    return plan


def stage_b_step() -> int:
    st = load_json(REP / "STATUS_STAGE_B.json")
    if "step" in st:
        return int(st["step"])
    last = STAGEB_OUT / "last.pt"
    if last.exists():
        try:
            import torch

            return int(torch.load(last, map_location="cpu", weights_only=False).get("step", 0))
        except Exception:
            pass
    return 0


def stage_b_train_alive() -> bool:
    return subprocess.run(["pgrep", "-f", "hrfont_e2_stageB96_train.py"], capture_output=True).returncode == 0


def stage_b_keepalive_alive() -> bool:
    return subprocess.run(["pgrep", "-f", "hrfont_stageB_keepalive.sh"], capture_output=True).returncode == 0


def start_stage_b(plan: dict) -> None:
    if (REP / "STOP_STAGE_B").exists() or (REP / "STOP_ALL").exists():
        log("STOP_STAGE_B/ALL — skip Stage B launch")
        return
    if stage_b_keepalive_alive():
        return
    env = os.environ.copy()
    env["HRFONT_STAGE_B_MAX"] = str(plan.get("max_steps", STAGE_B_DEFAULT))
    env["HRFONT_STAGEA_CKPT"] = plan["stageA_ckpt"]
    log(f"launch Stage B keepalive max={env['HRFONT_STAGE_B_MAX']} init={env['HRFONT_STAGEA_CKPT']}")
    subprocess.Popen(
        ["bash", str(ROOT / "scripts/hrfont_stageB_keepalive.sh")],
        env=env,
        stdout=open(REP / "keepalive_stageB.log", "a"),
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
        start_new_session=True,
    )


def write_complete_handoff(st: dict) -> None:
    best = st.get("best_eval") or {}
    plan = st.get("stage_b_plan") or {}
    md = f"""# HR-Font Stage A + B — 自动完成

更新时间：{time.strftime('%Y-%m-%d %H:%M:%S')}

## Stage A（Formal v2）

| 项 | 值 |
|----|-----|
| 最终 step | {st.get('final_step')} |
| 最佳 eval step | {best.get('step', '—')} |
| ft L1 | {best.get('ft_l1', '—')} |
| Stage A L1 | {best.get('stageA_l1', '—')} |
| Δ (ft − A) | {best.get('delta', '—')} |

## Stage B（SupportAdapter）

| 项 | 值 |
|----|-----|
| 计划 | {plan.get('reason', '—')} |
| max steps | {plan.get('max_steps', '—')} |
| 实际 step | {st.get('stage_b_final_step', '—')} |
| init ckpt | `{plan.get('stageA_ckpt', '—')}` |
| 输出 | `runs/e2_stageB96/last.pt` |

## 判定

- Stage A L1 < ft → Δ-RSI 接线有效（见 formal_eval/）
- Stage B 训完 → 下一步 **E4/E6 gap 分层 eval**、更新 mentor 页

## 文件

- `SUPERVISOR.json` · `STAGE_B_PLAN.json`
- `formal_eval/metrics_step_*.json`
- `dropout_ablation/VERDICT.md`
"""
    (REP / "NEXT_STEPS_COMPLETE.md").write_text(md, encoding="utf-8")
    log("wrote NEXT_STEPS_COMPLETE.md")


def run_stage_a_phase(st: dict) -> bool:
    """One poll iteration for Stage A. Returns True when A phase finished."""
    step = current_step()
    st["current_step"] = step
    st["train_alive"] = train_alive()
    st["keepalive_alive"] = keepalive_alive()
    save_supervisor(st)

    for ms in EVAL_MILESTONES:
        if step >= ms and ms not in st["eval_done"]:
            ck = resolve_ckpt_for_step(ms)
            if ck is None:
                log(f"milestone {ms}: no ckpt yet")
                continue
            rec = run_official_eval(ms, ck)
            if rec:
                st["eval_done"].append(ms)
                ft_l1 = rec.get("ft_cnstyle", {}).get("L1_mean")
                a_l1 = rec.get("stageA_formal", {}).get("L1_mean")
                entry = {"step": ms, "ft_l1": ft_l1, "stageA_l1": a_l1}
                if ft_l1 is not None and a_l1 is not None:
                    entry["delta"] = ft_l1 - a_l1
                hist = st.setdefault("eval_history", [])
                hist.append(entry)
                best = min(
                    (h for h in hist if h.get("stageA_l1") is not None),
                    key=lambda h: h["stageA_l1"],
                    default=None,
                )
                if best:
                    st["best_eval"] = best
                save_supervisor(st)

    done = step >= MAX_STEPS and not train_alive()
    stopped = (REP / "STOP_FORMAL").exists() and not train_alive()
    if done or stopped:
        if stopped:
            log("STOP_FORMAL — finalize Stage A eval")
        final_ms = MAX_STEPS if done else step
        if final_ms not in st["eval_done"]:
            ck = resolve_ckpt_for_step(final_ms) or (FORMAL_OUT / "last.pt")
            if ck and ck.exists():
                rec = run_official_eval(final_ms, ck)
                if rec:
                    st["eval_done"].append(final_ms)
                    ft_l1 = rec.get("ft_cnstyle", {}).get("L1_mean")
                    a_l1 = rec.get("stageA_formal", {}).get("L1_mean")
                    entry = {"step": final_ms, "ft_l1": ft_l1, "stageA_l1": a_l1}
                    if ft_l1 is not None and a_l1 is not None:
                        entry["delta"] = ft_l1 - a_l1
                    hist = st.setdefault("eval_history", [])
                    hist.append(entry)
                    best = min(
                        (h for h in hist if h.get("stageA_l1") is not None),
                        key=lambda h: h["stageA_l1"],
                        default=None,
                    )
                    if best:
                        st["best_eval"] = best
        st["phase"] = "stage_a_complete"
        st["final_step"] = step
        write_next_steps(st)
        save_supervisor(st)
        log("Stage A complete — handoff to Stage B")
        return True

    log(f"poll A step={step}/{MAX_STEPS} train={st['train_alive']} evals={st['eval_done']}")
    return False


def run_stage_b_phase(st: dict) -> bool:
    """Returns True when Stage B finished."""
    # Require at least one successful eval before Stage B
    if not st.get("best_eval") and not st.get("eval_history"):
        log("Stage B waiting — no eval metrics yet; run Stage A evals first")
        for ms in EVAL_MILESTONES:
            if ms not in st.get("eval_done", []):
                ck = resolve_ckpt_for_step(ms)
                if ck and ck.exists():
                    rec = run_official_eval(ms, ck)
                    if rec:
                        st.setdefault("eval_done", []).append(ms)
                        ft_l1 = rec.get("ft_cnstyle", {}).get("L1_mean")
                        a_l1 = rec.get("stageA_formal", {}).get("L1_mean")
                        entry = {"step": ms, "ft_l1": ft_l1, "stageA_l1": a_l1}
                        if ft_l1 is not None and a_l1 is not None:
                            entry["delta"] = ft_l1 - a_l1
                        hist = st.setdefault("eval_history", [])
                        hist.append(entry)
                        best = min(
                            (h for h in hist if h.get("stageA_l1") is not None),
                            key=lambda h: h["stageA_l1"],
                            default=None,
                        )
                        if best:
                            st["best_eval"] = best
                        save_supervisor(st)
                        break
        if not st.get("best_eval") and not st.get("eval_history"):
            return False

    plan = st.get("stage_b_plan") or load_json(REP / "STAGE_B_PLAN.json")
    if not plan:
        plan = plan_stage_b(st)
        st["stage_b_plan"] = plan
        save_supervisor(st)
    if st.get("phase") == "stage_a_complete":
        start_stage_b(plan)
        st["phase"] = "stage_b"
        st["stage_b_max"] = int(plan.get("max_steps", STAGE_B_DEFAULT))
        save_supervisor(st)

    sb_step = stage_b_step()
    st["stage_b_step"] = sb_step
    st["stage_b_train_alive"] = stage_b_train_alive()
    st["stage_b_keepalive_alive"] = stage_b_keepalive_alive()
    save_supervisor(st)

    target = int(st.get("stage_b_max", STAGE_B_DEFAULT))
    if sb_step >= target and not stage_b_train_alive():
        st["phase"] = "all_complete"
        st["stage_b_final_step"] = sb_step
        write_complete_handoff(st)
        save_supervisor(st)
        log("Stage B complete — pipeline done")
        return True

    if (REP / "STOP_STAGE_B").exists() and not stage_b_train_alive():
        st["phase"] = "stage_b_stopped"
        st["stage_b_final_step"] = sb_step
        write_complete_handoff(st)
        save_supervisor(st)
        return True

    log(f"poll B step={sb_step}/{target} train={st['stage_b_train_alive']}")
    return False


def main() -> None:
    REP.mkdir(parents=True, exist_ok=True)
    st = load_json(REP / "SUPERVISOR.json")
    st.setdefault("phase", "formal_v2")
    st.setdefault("eval_done", [])
    st.setdefault("milestones", EVAL_MILESTONES)
    st["max_steps"] = MAX_STEPS

    log("supervisor start (A → auto B)")
    if st.get("phase") == "complete":
        st["phase"] = "stage_a_complete"
        save_supervisor(st)
        log("migrated phase complete → stage_a_complete for Stage B")
    if st.get("phase") == "stage_b" and not st.get("best_eval"):
        st["phase"] = "stage_a_complete"
        st.pop("stage_b_plan", None)
        save_supervisor(st)
        log("reset premature stage_b → stage_a_complete (evals pending)")
    if st.get("phase") in ("formal_v2", "stage_a_complete", None):
        start_keepalive()

    while True:
        phase = st.get("phase", "formal_v2")
        if phase in ("formal_v2",):
            if run_stage_a_phase(st):
                st = load_json(REP / "SUPERVISOR.json")
                continue
        elif phase in ("stage_a_complete", "stage_b"):
            if run_stage_b_phase(st):
                log("supervisor ALL COMPLETE")
                return
        elif phase in ("all_complete", "stage_b_stopped", "complete", "stopped"):
            log(f"supervisor already done phase={phase}")
            return
        else:
            log(f"unknown phase={phase}")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("supervisor interrupted")
        sys.exit(0)
