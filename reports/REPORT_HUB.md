# HR-Font 报告导航

> 所有可视化报告的入口。精简介绍写在 REPORT_CATALOG.json；详情页负责深度内容。新增报告只改目录再 build。

生成：`2026-09-14T16:55:21Z` · 源：`reports/REPORT_CATALOG.json` · 重建：`python3 scripts/build_report_hub.py`

## 怎么用（防漏更新）

1. **新增/改报告** → 只改 `REPORT_CATALOG.json` 一条（title / blurb / path / port / status）
2. **重建导航** → `python3 scripts/build_report_hub.py`
3. **查漏** → `python3 scripts/build_report_hub.py --check`（未登记 HTML 会列在 `hub/unregistered.json`）
4. **打开** → `reports/hub/index.html` 或 http://127.0.0.1:8780/

## 今晚实验（2026-09-14）

| 报告 | 简介 | 打开 |
|------|------|------|
| **通宵实验看板 · F0/F2-CLEAN + E12-b** | 收敛曲线、脏/净 val 对照、E12 φ/membership；数据自动重建。 | [文件](tonight_20260914_dashboard/index.html) · [:8790](http://127.0.0.1:8790/index.html) · ✓ |
| **F2 中间测试图 · 脏 vs CLEAN** | test16×TIMELINE_CHARS 中途 ckpt 并排；协议同 DPM++20/CFG7.5/seed3407。进度更新不跳字体。 | [文件](f03_test16_strat/timeline_f2_clean.html) · [:8791](http://127.0.0.1:8791/timeline_f2_clean.html) · ✓ |
| **F2C 中间图（挂在 8790 目录）** | 与 f2c_mid_board 同内容镜像，方便从通宵看板同端口打开。 | [文件](tonight_20260914_dashboard/timeline_f2_clean.html) · [:8790](http://127.0.0.1:8790/timeline_f2_clean.html) · ✓ |
| **通宵 MORNING 文本看板** | watchdog 阶段快照；事故见 incidents.jsonl。 | [文件](watchdog_tonight_20260914/MORNING.md) · ✓ |
| **F0/F2 clean STATUS.md** | 脏/净对齐 val 表与 cache 状态。 | [文件](f0f2_clean_v0913/STATUS.md) · ✓ |
| **E12-b STATUS.md** | φ best AUC + membership val/test。 | [文件](e12_b/STATUS.md) · ✓ |

## 正式评测与导师板

| 报告 | 简介 | 打开 |
|------|------|------|
| **F0/F2/F3 Glyph Board（test16 主评测）** | 协议 A 终点图板 + L1/SSIM/LPIPS 诊断；正式比较主入口。 | [文件](f03_test16_strat/index.html) · [:8791](http://127.0.0.1:8791/index.html) · ✓ |
| **Demo-8 · 1-shot / 8-shot 对照** | 核心模型冻结 Demo-8 一眼板；与 test16 全量互补。 | [文件](f03_test16_strat/core_shot_board.html) · [:8791](http://127.0.0.1:8791/core_shot_board.html) · ✓ |
| **Multi-shot 眼板** | 多样本条件对照；与 Demo-8 / test16 互补。 | [文件](f03_test16_strat/multi_shot_board.html) · [:8791](http://127.0.0.1:8791/multi_shot_board.html) · ✓ |
| **对照组精选看图** | PI 精选字形对。 | [文件](f03_test16_strat/pi_highlights.html) · [:8791](http://127.0.0.1:8791/pi_highlights.html) · ✓ |
| **F2Vec 眼板** | F2 向量侧诊断看图。 | [文件](f03_test16_strat/f2vec_eye_board.html) · [:8791](http://127.0.0.1:8791/f2vec_eye_board.html) · ✓ |
| **F0/F2/F3 Val16 分层板** | val16 分层评测入口（非 test16 正式终点）。 | [文件](f03_val16_strat/index.html) · ✓ |
| **Compare Portal（方法状态总览）** | 各臂 ready/queued 与指标摘要。 | [文件](compare_portal/index.html) · ✓ |
| **多实验并排对照** | 跨 run / 方法一眼对照。 | [文件](multi_exp_compare/index.html) · ✓ |
| **F3 ckpt 视觉对照** | Official / F0 / F3 视觉并排。 | [文件](f3_ckpt_dashboard/index.html) · ✓ |
| **导师简报 2026-09-07** | 阶段性导师汇报页（历史）。 | [文件](mentor_briefing_20260907/index.html) · ✓ |
| **补充测试 6 套 · Paper6 OOD（肉眼）** | 不在原260的6套新增测试字体；未查中西文风格一致性。可肉眼观察，算指标须再确认。 | [文件](paper6_0914_f0f2/index.html) · [:8780](http://127.0.0.1:8780/) · ✓ |

## 训练监控

| 报告 | 简介 | 打开 |
|------|------|------|
| **F0–F3 训练看板** | 各臂 step/loss/ETA/GPU。 | [文件](f123_dashboard/index.html) · ✓ |
| **F2 脏臂训练过程时间线** | 脏 F2 从 5k→75k 的中间结果（历史基线）。 | [文件](f03_test16_strat/timeline_f2.html) · [:8791](http://127.0.0.1:8791/timeline_f2.html) · ✓ |
| **F2P 40k 对照** | F2P 40k 视觉/指标对照。 | [文件](f2p_40k_compare/index.html) · ✓ |

## E12 打分器

| 报告 | 简介 | 打开 |
|------|------|------|
| **E12 v5.1 眼板（train228）** | 同域打分器错误样例；v51 冻结基线。 | [文件](e12_eye_probe_v51_train228_s3407/index.html) · ✓ |
| **E12 v4 眼板（外部池）** | 外部 OFL 池诊断；与 E12-b 不同合同。 | [文件](e12_eye_probe_v4_s3407/index.html) · ✓ |
| **E12-b STATUS.md** | φ best AUC + membership val/test。 | [文件](e12_b/STATUS.md) · ✓ |

## 工具与探针

| 报告 | 简介 | 打开 |
|------|------|------|
| **字符集挑选器** | 选字工具页。 | [文件](charset_picker/index.html) · ✓ |
| **Style / Domain 探针** | 风格域分布探针板。 | [文件](style_domain_probe/index.html) · ✓ |
| **Delta Retrieve 板** | 检索差量可视化。 | [文件](f03_test16_strat/delta_retrieve/index.html) · ✓ |

## 历史归档（少用）

| 报告 | 简介 | 打开 |
|------|------|------|
| **通用 timeline.html（旧）** | 早期通用时间线；优先用 timeline_f2 / timeline_f2_clean。 | [文件](f03_test16_strat/timeline.html) · [:8791](http://127.0.0.1:8791/timeline.html) · ✓ |
| **Series750 权重浏览** | 750 系列权重浏览页。 | [文件](series750_weight_browse/index.html) · ✓ |
| **离线协作包入口** | 离线带走的协作导航；含 f123_board。 | [文件](collab_offline/index.html) · ✓ |
| **离线包 · F123 板** | collab_offline 内嵌训练板副本。 | [文件](collab_offline/f123_board/index.html) · ✓ |
| **E1 FT v2 看板** | 早期 E1 fine-tune 看板（历史）。 | [文件](e1_ft_v2_dashboard/index.html) · ✓ |
| **通宵正式预览（旧）** | 早期 overnight formal preview。 | [文件](hrfont_overnight/formal_preview/index.html) · ✓ |
| **StageA MVP 视觉（旧）** | 早期 StageA 视觉页。 | [文件](hrfont_stagea_mvp_visual/index.html) · ✓ |
| **FD Protocol 报告（retrain_v2）** | 协议报告历史页。 | [文件](retrain_v2/fd_protocol_report/index.html) · ✓ |

## 未分组

- **干净 vs 脏 · F0/F2 匹配指标** — P1 / 脏F0·F2 / 净F0·F2 同字符 L1·SSIM + E12 mem；看清洗与 Delta-F2 有没有用。 (`f03_test16_strat/clean_dirty_compare.html`)
- **全西文 592 · test16 重测** — test16×37西文(digit+拉丁+扩展)；E12/Oursφ固定；CLIP/DINO/Alex/LPIPS vs GT（f03协议）。 (`f03_test16_strat/western_all_metrics.html`)
- **G 系 v0913 · 1/8-shot 看板** — G0b/G0c/G1/G2/G2-RL/pilot vs dirty F2/F2-RL；合作者入口 http://172.19.45.13:19000/g_shot/ (`g_v0913_shot/index.html`)

## ⚠ 未登记 HTML（可能漏更新目录）

- `e12_paper/human_board/index.html`
- `font_probe_FZBangSKLTJW/index.html`
- `font_probe_FZBangSKLTJW/timeline_f2.html`
- `font_probe_FZBangSKLTJW/timeline_f2_clean.html`
- `g_v0913_shot/ref8_proposal/index.html`
- `g_v0913_shot_k1248/index.html`
- `paper_fonts_0914_screen/index.html`
- `v100_hub/index.html`
