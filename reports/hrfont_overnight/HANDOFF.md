# 过夜任务 — 你关笔记本后自动跑完（含 Stage B）

**无需在线。** `night_watch` + `supervisor` 会串行完成 **Stage A → Stage B**。

---

## 流水线

1. **Stage A formal** (80k) — GPU0，keepalive 保活  
2. **里程碑 eval** — 10k/25k/50k/70k/80k 官方协议  
3. **Dropout 消融** — GPU2/3，~3h  
4. **Stage B 自动启动** — 根据 A 的结果决定步数：

| Stage A vs ft | Stage B 步数 |
|---------------|-------------|
| A 胜 ft | **40k** |
| A 接近 ft (≤8%) | **25k** |
| A 落后 ft | **15k**（短训验证支撑） |

5. **完成报告** — `NEXT_STEPS_COMPLETE.md`

---

## 回来后

```bash
cat reports/hrfont_overnight/STATUS_OVERNIGHT.md
cat reports/hrfont_overnight/NEXT_STEPS_COMPLETE.md
```

---

## Stage B 实现

- 仅训 **SupportAdapter**（冻 UNet + encoder）  
- gap≥0.35 喂支撑，cover+MMR 选字，20% CFG 丢 Support  
- 输出：`runs/e2_stageB96/last.pt`

---

## 停止

```bash
touch reports/hrfont_overnight/STOP_ALL
```
