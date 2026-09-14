# G 系 1/2/4/8-shot 评测方案（旧 Ref）— 待 Review
- 更新：2026-09-14T16:30:03.664224+08:00
- 状态：**已开跑**（复用旧板 test 1/8；补 2/4 + train/val + 新模型）

## 1. 相对上一版的变更
| 项 | 上一版（已确认过） | **本版（请 review）** |
|------|-------------------|----------------------|
| Ref8 | `山水云月金石心文` | **改回** `永和书风骨韵天地` |
| Shot | 1/2/4/8 取自新 Ref | **仍 1/2/4/8，取自旧 Ref 前缀** |
| 采样 | DPM++20 CFG7.5 s3407 | **不变** |
| Content | stratified47 全量 | **不变** |
| 加速 | 单卡 batch≈8 | **不变（实现后再跑）** |

## 2. Ref / Shot 定义（可比）
- Ref8 全集：`永和书风骨韵天地`
- 每个 shot **k** = 该串的前 k 字；**Es 与 Δ 同一组**（true-k，非 Mode D）。

| k | Ref 子集 |
|---|----------|
| 1 | `永` |
| 2 | `永和` |
| 4 | `永和书风` |
| 8 | `永和书风骨韵天地` |

## 3. Content（生成目标字）
- 集合：**stratified47**（与 F/`f03` 相同），共 47 字。
- 内容：`0123456789AGMQRWBCOaodpqbegcilnuàéüāěあかさんアカンㄅㄆㄚ`
- 看板 HTML 可只展示 CURATED 17 字；**磁盘推理跑满 47**。

## 4. 字体
- **test16** 全量：`FZBuGTJW, FZChuangHJW_DB, FZCuanBZBKSJW, FZDeSHJW_515H, FZDouNTJW_Te, FZFeiHHJW-H, FZFengYKSJ, FZHanWZKJW, FZHuoYYJW-T, FZJianLTJW_Te, FZJingLTJW-H, FZJingYLLTJW, FZLingFKSJW-B, FZLiuGQKJW-L, FZMaWDBSJW, FZTJLSJW`
- **train5**：`FZPTYJW, FZDuHJW_Cu, FZShuLTJW-H, FZYiMSJW-T, FZFeiSJW-EL`
- **val5**：`FZYouHK_511M, FZLTHProGBK_H, FZJunYTJW-H, FZBangSKKXJW, FZKANGJW`

## 5. 模型与 shot
| 模型 | ckpt | 跑哪些 shot |
|------|------|-------------|
| G0b@10k | `runs/G0b-F0-V0913-BS256-A-S3407/global_step_10000` | [1] |
| G0c@20k | `runs/G0c-F0-V0913-BS256-A-S3407/global_step_20000` | [1] |
| G1@10k | `runs/G1-F1-V0913-A-S3407/global_step_10000` | [1, 2, 4, 8] |
| G2@10k | `runs/G2-F2-V0913-A-S3407/global_step_10000` | [1, 2, 4, 8] |
| G2-RL@10k | `runs/G2-RL-V0913-A-S3407/global_step_10000` | [1, 2, 4, 8] |
| pilot@best1k | `runs/G-RL-pilot-V0913-A-S3407/global_step_1000` | [1, 2, 4, 8] |
| pilot8@best2500 | `runs/G-RL-pilot-8gpu-V0913-A-S3407/best` | [1, 2, 4, 8] |
| TC-G2@best2500 | `runs/G-TC-G2-8gpu-V0913-A-S3407/best` | [1, 2, 4, 8] |
| TC-G2RL@best2500 | `runs/G-TC-G2RL-8gpu-V0913-A-S3407/best` | [1, 2, 4, 8] |

- G0 仅 1-shot（单 style 图条件）。
- pilot / TC 用 **best**（均为 @2500）。
- **本轮不含** dirty F2/F2-RL（除非你加回来）。

## 6. 采样协议（不变）
- `dpmsolver++` · **20 steps** · **CFG 7.5** · **seed 3407** · order2 multistep
- 与现有 `g_v0913_shot` / `f03_test16_strat` 一致。

## 7. 工程与产出
- 输出目录建议：`reports/g_v0913_shot_k1248/`（不覆盖现有 1/8 旧板）
- 8 卡并行 + **每卡 batch≈8**（OOM 则 4→1）
- 工作量：约 **36,660** 张（26 字体 × 47 × 30 臂）
- ETA：纯推理约 **25–50 min**；含实现/组板约 **40–70 min**

## 8. 请你拍板
1. 旧 Ref + 上表 shot 前缀 — 是否 OK？
2. train5/val5 名单 — 是否 OK？
3. 模型名单（含 pilot8 / TC-G2 / TC-G2RL，不含 dirty）— 是否 OK？
4. 新看板目录 `g_v0913_shot_k1248` — 是否 OK？
5. 确认后是否立刻开跑？
