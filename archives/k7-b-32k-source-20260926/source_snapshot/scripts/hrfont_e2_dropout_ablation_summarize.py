#!/usr/bin/env python3
"""Summarize dropout ablation eval JSONs into VERDICT.md + summary.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
REP = ROOT / "reports/hrfont_overnight/dropout_ablation"

RUNS = [
    ("R0_formal_ref", "formal @80k (reference)", None, "GPU0 main — see SUPERVISOR.json"),
    ("R1_cfg01_d01", "cfg0.1 + Δ0.1", "both", "Δ 与官方同率"),
    ("R2_cfg01_d00", "cfg0.1 + Δ0.0", "both", "Δ 从不 drop"),
    ("R3_style025_d025", "style0.25 + Δ0.25", "style_only", "昨晚 v1 式（不 drop content）"),
    ("R4_cfg025_d025", "cfg0.25 + Δ0.25", "both", "全 0.25"),
]

REF = ("R0_default", "cfg0.1 + Δ0.25 (formal default)", "both", "当前 formal 默认")


def load_eval(run_id: str) -> dict | None:
    if run_id in ("R0_formal_ref", "R0_default"):
        for name in ("metrics_step_80000.json", "metrics_step_10000.json", "metrics.json"):
            p = ROOT / "reports/hrfont_overnight/formal_eval" / name
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        return None
    p = REP / f"eval_{run_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    REP.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_id, label, mode, note in [REF] + [r for r in RUNS if r[0] != "R0_formal_ref"]:
        ev = load_eval(run_id if run_id != "R0_default" else "R0_formal_ref")
        if run_id == "R0_default" and ev is None:
            rows.append({"run_id": "R0", "label": label, "status": "pending (formal main @10k+)"})
            continue
        if ev is None:
            rows.append({"run_id": run_id, "label": label, "status": "pending"})
            continue
        ft = ev.get("ft_cnstyle", {}).get("L1_mean") or ev.get("results", {}).get("ft_cnstyle", {}).get("L1_mean")
        a = ev.get("stageA_formal", {}).get("L1_mean") or ev.get("results", {}).get("stageA_formal", {}).get("L1_mean")
        rows.append({
            "run_id": run_id,
            "label": label,
            "drop_mode": mode,
            "note": note,
            "ft_l1": ft,
            "stageA_l1": a,
            "beats_ft": (a < ft) if (a is not None and ft is not None) else None,
            "delta_vs_ft": (ft - a) if (a is not None and ft is not None) else None,
        })

    (REP / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    md = """# Dropout 消融 — 结论草案

## 要验证什么

| 设定 | 含义 | 若成立说明 |
|------|------|------------|
| **content+style @0.1** | 官方 CFG：cond/uncond 成对训练 | 推理 CFG=7.5 有对应 uncond 分支 |
| **Δ @0.25** | HR-Font §6.2：20–30% 关掉 Δ 特征 | 模型学会「无 Δ 引导」时不乱靠库混形变 |
| **R1 Δ@0.1** | 全分支统一 0.1 | 若 ≈ R0 → 0.25 非必须 |
| **R2 Δ@0.0** | 从不 drop Δ | 若差于 R0 → Δ dropout 有必要 |
| **R3 style_only** | 不 drop content | 若差于 R0 → content CFG 有必要 |

## 结果 @10k（官方 eval：DejaVu + DPM++ + CFG）

| Run | 配置 | StageA L1 | ft L1 | 胜 ft? |
|-----|------|-----------|-------|--------|
"""
    for r in rows:
        md += f"| {r.get('run_id','?')} | {r.get('label', r.get('note',''))} | {r.get('stageA_l1', '—')} | {r.get('ft_l1', '—')} | {r.get('beats_ft', '—')} |\n"

    md += """
## 读法

- **R0（formal 默认）** 应在 10k 时已有趋势；80k 为最终主结论。
- 消融 **control 变量**：仅改 dropout，其余（DejaVu、VGG、reinit、init）相同。
- 若 R1≈R0 且 R2明显更差 → **保留 Δ@0.25** 有依据；若 R1更好 → formal 可改 Δ@0.1。
- 若 R3差于 R0 → **content+style@0.1** 必要；R3≈R3说明 content drop 影响小。

协议：`formal_eval/eval_*.json` · 训练 10k step · GPU2/3
"""
    (REP / "VERDICT.md").write_text(md, encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
