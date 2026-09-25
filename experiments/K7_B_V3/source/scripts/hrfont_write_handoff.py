#!/usr/bin/env python3
"""Write READ_ME_FIRST.md and NEXT_STEPS_COMPLETE.md when pipeline finishes."""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
REP = ROOT / "reports/hrfont_overnight"
STAGEB = ROOT / "runs/e2_stageB96"
FE = REP / "formal_eval"


def main() -> None:
    met = json.loads((FE / "metrics_step_80000.json").read_text())
    plan = json.loads((REP / "STAGE_B_PLAN.json").read_text()) if (REP / "STAGE_B_PLAN.json").exists() else {}
    sb_step = 0
    if (STAGEB / "last.pt").exists():
        import torch

        sb_step = int(torch.load(STAGEB / "last.pt", map_location="cpu", weights_only=False).get("step", 0))

    ft, a = met["ft_cnstyle"]["L1_mean"], met["stageA_formal"]["L1_mean"]
    sb_done = sb_step >= int(plan.get("max_steps", 25000))

    b_l1 = None
    mb = FE / "metrics_stageB_25000.json"
    if mb.exists():
        b_l1 = json.loads(mb.read_text()).get("stageB", {}).get("L1_mean")

    gap_note = ""
    gp = FE / "gap_stratified.json"
    if gp.exists():
        s = json.loads(gp.read_text()).get("summary_by_bucket", {})
        parts = []
        for bk in ("low", "mid", "high"):
            if bk in s:
                parts.append(f"{bk}: A胜率{100*s[bk]['a_win_rate']:.0f}%")
        gap_note = " · ".join(parts)

    read_me = f"""# 回来后先看这个

更新时间：{time.strftime("%Y-%m-%d %H:%M:%S")}

## 主结果（Formal 协议 · n=976）

| 方法 | L1 |
|------|-----|
| ft_cnstyle@25k | **{ft:.4f}** |
| Stage A Δ-RSI @80k | **{a:.4f}** ({(a/ft-1)*100:+.1f}%) |
| Stage B +Support | **{f"{b_l1:.4f}" if b_l1 else "待 eval"}** |

## Gap 分层（Stage A vs ft）

{gap_note or "见 formal_eval/gap_stratified.md"}

## 页面与文档

- **Formal 预览：** `formal_preview/index.html`（DejaVu 协议，替代 mentor_preview）
- **完整审查：** `REVIEW_SUMMARY_AND_PLAN.md`
- **Dropout：** `dropout_ablation/VERDICT.md`

## Stage B

- step **{sb_step}** / {plan.get("max_steps", 25000)} {'✅' if sb_done else '⏳'}

## 停止

`touch reports/hrfont_overnight/STOP_ALL`
"""
    (REP / "READ_ME_FIRST.md").write_text(read_me, encoding="utf-8")

    b_row = f"| Stage B | {b_l1:.4f} | {'胜 A' if b_l1 and b_l1 < a else '—'} |" if b_l1 else "| Stage B | 待 eval | — |"

    complete = f"""# HR-Font 无人值守完成报告

时间：{time.strftime("%Y-%m-%d %H:%M:%S")}

## L1 主表

| 方法 | L1 | 备注 |
|------|-----|------|
| ft_cnstyle | {ft:.4f} | 对照 |
| Stage A @80k | {a:.4f} | {'接近 ft' if a > ft else '胜 ft'} (+{(a/ft-1)*100:.1f}%) |
{b_row}

## 产出文件

- `formal_eval/metrics_step_80000.json`
- `formal_eval/metrics_stageB_25000.json`
- `formal_eval/gap_stratified.json` + `.md`
- `formal_eval/style_metrics.json`
- `formal_preview/index.html`

## 下一步（需 review）

1. 设计师 user study（风格合理性，非 L1）
2. 对照臂：RSI 仍看汉字 @10k
3. 统一渲染协议文档化

详见 `REVIEW_SUMMARY_AND_PLAN.md`。
"""
    (REP / "NEXT_STEPS_COMPLETE.md").write_text(complete, encoding="utf-8")
    print(f"handoff sb={sb_step} b_l1={b_l1}")


if __name__ == "__main__":
    main()
