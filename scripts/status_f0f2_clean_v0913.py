#!/usr/bin/env python3
"""Refresh F0/F2 dirty-vs-v0913_clean STATUS for overnight monitoring.

Writes:
  reports/f0f2_clean_v0913/STATUS.md
  reports/f0f2_clean_v0913/STATUS.json

Does not start training. Safe to run under watch/cron.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
OUT = ROOT / "reports/f0f2_clean_v0913"
CONTRACT = ROOT / "reports/EXPERIMENT_F0F2_CLEAN_V0913_20260914.md"

DIRTY_F0 = ROOT / "runs/F0-RSIFREE-FT-A-S3407"
DIRTY_F2 = ROOT / "runs/F2-DELTARSI-A-S3407"
CLEAN_F0 = ROOT / "runs/F0-CLEAN-V0913-A-S3407"
CLEAN_F2 = ROOT / "runs/F2-CLEAN-V0913-A-S3407"
CLEAN_ES = ROOT / "artifacts/f0_clean_v0913/es_spatial"
CLEAN_EC = ROOT / "artifacts/f0_clean_v0913/ec_multiscale"
DIRTY_ES = ROOT / "artifacts/f0/es_spatial_f0"
DIRTY_EC = ROOT / "artifacts/f0/ec_multiscale_f0"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def milestones(run: Path) -> list[int]:
    steps = []
    if not run.is_dir():
        return steps
    for p in run.glob("global_step_*"):
        if (p / "unet.pth").is_file():
            try:
                steps.append(int(p.name.split("_")[-1]))
            except ValueError:
                pass
    return sorted(steps)


def resolve_best(run: Path) -> dict:
    best = run / "best"
    out = {"exists": best.exists(), "target": None, "step": None}
    if best.is_symlink():
        out["target"] = str(best.resolve())
        name = best.resolve().name
        if name.startswith("global_step_"):
            out["step"] = int(name.split("_")[-1])
    elif best.is_dir() and (best / "unet.pth").is_file():
        out["target"] = str(best)
    return out


def f0_train_points(run: Path) -> list[dict]:
    """Prefer reports training_logs; else parse fontdiffuser_training.log."""
    alt = ROOT / "reports/training_logs" / run.name / "train_loss.jsonl"
    rows = read_jsonl(alt)
    if rows:
        return [{"step": r.get("step"), "loss": r.get("train_loss", r.get("loss"))} for r in rows if r.get("step") is not None]

    log = run / "fontdiffuser_training.log"
    if not log.is_file():
        return []
    out = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r"Global Step (\d+) .* train_loss = ([0-9.eE+-]+)", line)
        if m:
            out.append({"step": int(m.group(1)), "loss": float(m.group(2))})
    return out


def f0_val_points(run: Path) -> list[dict]:
    hist = ROOT / "reports/training_logs" / run.name / "val_loss_history.json"
    data = read_json(hist)
    if isinstance(data, dict) and "milestones" in data:
        rows = data["milestones"]
        return [{"step": int(r["step"]), "val_loss": float(r["val_loss"])} for r in rows if "val_loss" in r]
    if isinstance(data, list):
        return [{"step": int(r["step"]), "val_loss": float(r.get("val_loss", r.get("loss")))} for r in data]
    # F2-style val_log also used if present
    return [
        {"step": int(r["step"]), "val_loss": float(r["val_loss"])}
        for r in read_jsonl(run / "val_log.jsonl")
        if "val_loss" in r
    ]


def f2_series(run: Path) -> dict:
    train = read_jsonl(run / "train_log.jsonl")
    val = read_jsonl(run / "val_log.jsonl")
    hb = read_json(run / "heartbeat.json")
    done = read_json(run / "DONE.json")
    last_train = None
    for r in reversed(train):
        if "loss" in r or "train_loss" in r:
            last_train = {"step": r.get("step"), "loss": r.get("loss", r.get("train_loss")), "lr": r.get("lr")}
            break
    val_pts = [{"step": r["step"], "val_loss": r["val_loss"]} for r in val if "val_loss" in r]
    best_val = None
    if val_pts:
        best_val = min(val_pts, key=lambda x: x["val_loss"])
    return {
        "exists": run.is_dir(),
        "done": done,
        "heartbeat": hb,
        "milestones": milestones(run),
        "best_link": resolve_best(run),
        "last_train": last_train,
        "val": val_pts,
        "best_val": best_val,
        "n_train_log": len(train),
    }


def f0_series(run: Path) -> dict:
    done = read_json(run / "DONE.json")
    meta = read_json(run / "launch_meta.json")
    train = f0_train_points(run)
    val = f0_val_points(run)
    last_train = train[-1] if train else None
    best_val = min(val, key=lambda x: x["val_loss"]) if val else None
    return {
        "exists": run.is_dir(),
        "done": done,
        "meta": {k: meta.get(k) for k in ("run_id", "dataset_id", "max_steps", "lr", "seed", "gpu", "created_at")} if meta else None,
        "milestones": milestones(run),
        "best_link": resolve_best(run),
        "last_train": last_train,
        "val": val,
        "best_val": best_val,
        "n_train_log": len(train),
    }


def cache_info(path: Path) -> dict:
    man = read_json(path / "manifest.json")
    if not man:
        return {"exists": False, "path": str(path)}
    return {
        "exists": True,
        "path": str(path),
        "ckpt_dir": man.get("ckpt_dir"),
        "entries": man.get("entries"),
        "kind": man.get("kind"),
    }


def align_val(dirty: list[dict], clean: list[dict]) -> list[dict]:
    dmap = {r["step"]: r["val_loss"] for r in dirty}
    out = []
    for r in clean:
        s = r["step"]
        if s in dmap:
            out.append(
                {
                    "step": s,
                    "dirty_val": dmap[s],
                    "clean_val": r["val_loss"],
                    "delta_clean_minus_dirty": r["val_loss"] - dmap[s],
                }
            )
    return out


def phase_guess(payload: dict) -> str:
    if payload["clean_f2"].get("done") and payload["clean_f2"]["done"].get("status") == "completed":
        return "F2_done_await_eval"
    if payload["clean_f2"]["exists"] and payload["clean_f2"]["milestones"]:
        return "F2_training"
    if payload["clean_es"]["exists"] and payload["clean_ec"]["exists"]:
        return "cache_ready_await_F2"
    if payload["clean_f0"].get("done") and payload["clean_f0"]["done"].get("status") == "completed":
        return "F0_done_await_cache"
    if payload["clean_f0"]["exists"] and payload["clean_f0"]["milestones"]:
        return "F0_training"
    if payload["clean_f0"]["exists"]:
        return "F0_started"
    return "not_started"


def md_table(rows: list[dict], keys: list[str], headers: list[str]) -> str:
    if not rows:
        return "_（尚无数据）_\n"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dirty_f0 = f0_series(DIRTY_F0)
    clean_f0 = f0_series(CLEAN_F0)
    dirty_f2 = f2_series(DIRTY_F2)
    clean_f2 = f2_series(CLEAN_F2)

    payload = {
        "updated_at": utc(),
        "contract": str(CONTRACT.relative_to(ROOT)),
        "dirty_f0": dirty_f0,
        "clean_f0": clean_f0,
        "dirty_f2": dirty_f2,
        "clean_f2": clean_f2,
        "dirty_es": cache_info(DIRTY_ES),
        "dirty_ec": cache_info(DIRTY_EC),
        "clean_es": cache_info(CLEAN_ES),
        "clean_ec": cache_info(CLEAN_EC),
        "f0_val_aligned": align_val(dirty_f0["val"], clean_f0["val"]),
        "f2_val_aligned": align_val(dirty_f2["val"], clean_f2["val"]),
        "do_not_overwrite": [
            str(DIRTY_F0),
            str(DIRTY_F2),
            str(DIRTY_ES),
            str(DIRTY_EC),
        ],
    }
    payload["phase"] = phase_guess(payload)

    # compact sparkline-style last points for md
    def last_n(rows, n=8):
        return rows[-n:] if rows else []

    f0_cmp = [
        {
            "step": r["step"],
            "dirty": f"{r['dirty_val']:.6f}",
            "clean": f"{r['clean_val']:.6f}",
            "Δ(c-d)": f"{r['delta_clean_minus_dirty']:+.6f}",
        }
        for r in last_n(payload["f0_val_aligned"], 12)
    ]
    f2_cmp = [
        {
            "step": r["step"],
            "dirty": f"{r['dirty_val']:.6f}",
            "clean": f"{r['clean_val']:.6f}",
            "Δ(c-d)": f"{r['delta_clean_minus_dirty']:+.6f}",
        }
        for r in last_n(payload["f2_val_aligned"], 12)
    ]

    md = []
    md.append("# F0/F2 clean@v0913 · STATUS\n")
    md.append(f"- 更新：`{payload['updated_at']}`")
    md.append(f"- 阶段：**{payload['phase']}**")
    md.append(f"- 合同：[`{payload['contract']}`](../EXPERIMENT_F0F2_CLEAN_V0913_20260914.md)")
    md.append("")
    md.append("## 读数规则")
    md.append("- F0：看 val16（扫描后）；`best` 按最小 val，平局更小 step。")
    md.append("- F2：主对比 **@40k**；附录 @80k / run best。")
    md.append("- Δ(c−d)<0 → 干净侧 val 更低（更好）。尚未开跑则表空。")
    md.append("- **禁止**覆盖 `do_not_overwrite` 列表中的脏臂路径。")
    md.append("")
    def dirty_val_ref(rows, n=8):
        return [
            {"step": r["step"], "dirty": f"{r['val_loss']:.6f}"}
            for r in (rows[-n:] if rows else [])
        ]

    md.append("## F0")
    md.append(
        f"- 脏：exists={dirty_f0['exists']} milestones={dirty_f0['milestones'][-5:] if dirty_f0['milestones'] else []} "
        f"best_link={dirty_f0['best_link']} curve_min={dirty_f0['best_val']}"
    )
    md.append(
        f"- 净：exists={clean_f0['exists']} milestones={clean_f0['milestones'][-5:] if clean_f0['milestones'] else []} "
        f"best_link={clean_f0['best_link']} last_train={clean_f0['last_train']} curve_min={clean_f0['best_val']}"
    )
    md.append("")
    md.append("### F0 val 同 step 对照（有干净点才并排）")
    md.append(md_table(f0_cmp, ["step", "dirty", "clean", "Δ(c-d)"], ["step", "dirty val", "clean val", "Δ(c−d)"]))
    if not f0_cmp and dirty_f0["val"]:
        md.append("### 脏 F0 val 参考（干净未开始）")
        md.append(md_table(dirty_val_ref(dirty_f0["val"]), ["step", "dirty"], ["step", "dirty val"]))
    md.append("## F2")
    md.append(
        f"- 脏：exists={dirty_f2['exists']} milestones 含 40k/80k="
        f"{40_000 in dirty_f2['milestones']}/{80_000 in dirty_f2['milestones']} "
        f"DONE.best={dirty_f2.get('done')} curve_min={dirty_f2['best_val']} hb={dirty_f2['heartbeat']}"
    )
    md.append(
        f"- 净：exists={clean_f2['exists']} milestones={clean_f2['milestones'][-5:] if clean_f2['milestones'] else []} "
        f"DONE={clean_f2.get('done')} curve_min={clean_f2['best_val']} "
        f"last_train={clean_f2['last_train']} hb={clean_f2['heartbeat']}"
    )
    md.append("")
    md.append("### F2 val 同 step 对照（有干净点才并排）")
    md.append(md_table(f2_cmp, ["step", "dirty", "clean", "Δ(c-d)"], ["step", "dirty val", "clean val", "Δ(c−d)"]))
    if not f2_cmp and dirty_f2["val"]:
        md.append("### 脏 F2 val 参考（干净未开始）")
        md.append(md_table(dirty_val_ref(dirty_f2["val"]), ["step", "dirty"], ["step", "dirty val"]))
    md.append("## Cache")
    md.append(f"- 脏 Es: `{payload['dirty_es']}`")
    md.append(f"- 脏 Ec: `{payload['dirty_ec']}`")
    md.append(f"- 净 Es: `{payload['clean_es']}`")
    md.append(f"- 净 Ec: `{payload['clean_ec']}`")
    md.append("")
    md.append("## 刷新")
    md.append("```bash")
    md.append("python3 scripts/status_f0f2_clean_v0913.py")
    md.append("# 或: watch -n 300 python3 scripts/status_f0f2_clean_v0913.py")
    md.append("```\n")

    (OUT / "STATUS.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "STATUS.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT / 'STATUS.md'}")
    print(f"phase={payload['phase']}")


if __name__ == "__main__":
    main()
