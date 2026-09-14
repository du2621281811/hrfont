# 报告网页管理约定

## 问题

报告页散落在 `reports/**`，端口也不一（8790 / 8791 / 8777…）。找页靠记路径，更新时容易漏改入口。

## 约定（从现在起）

| 角色 | 文件 | 谁改 |
|------|------|------|
| **唯一目录源** | `reports/REPORT_CATALOG.json` | 人手改 |
| **生成导航** | `reports/hub/index.html` + `REPORT_HUB.md` | 脚本生成，勿手改 |
| **查漏列表** | `reports/hub/unregistered.json` | 脚本扫描 |

### 新增一个报告页时

1. 写出 HTML（或 md）
2. 在 `REPORT_CATALOG.json` 的 `pages` 加一条：`id / title / blurb / path / port? / tags / status`
3. 放进合适的 `groups[].page_ids`
4. 运行：`python3 scripts/build_report_hub.py`
5. （可选）`python3 scripts/build_report_hub.py --check` 确认没有未登记页

### 打开导航

- 文件：`reports/hub/index.html`
- 或：http://127.0.0.1:8780/hub/
- 顶层：`reports/index.html` → 跳转到 hub

### 为什么能防漏

- **介绍只写在 catalog**，hub / REPORT_HUB.md 都是生成的，不会出现「改了页忘了改索引」的双份文案。
- **`--check`** 扫描 `*board*.html` / `timeline*.html` / `index.html`，未登记就失败并写入 `unregistered.json`。
- 生成脚本可挂在「出新 board 的脚本末尾」：`subprocess` 调一次 build。

### 不做什么

- 不强制所有旧页立刻登记完；`--check` 先当提醒，逐步消化 `unregistered.json`。
- 不把大段实验结论塞进 hub——hub 只放**一句话 blurb + 链接**。
