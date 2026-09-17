**默认推理：Train / Val / Test × 128候选字 × 1/2/4/8-shot**

用户于2026-09-17要求将同时包含训练、验证、测试集的多shot方案作为默认，并增加字符数。配置入口是 `experiments/DEFAULT_INFERENCE.json`；执行脚本 `scripts/k_default_eval.py`，对照页面 `scripts/build_k_default_board.py`。

沿用原47字面板的 Train5 / Val5 / Test16 字体，候选字符扩至128（西文80、假名32、注音16），包含原47字。先按clean pair过滤，不跨split寻找替代图。实际每模型：Train2304、Val2112、Test7232，共11648图。每个合法字体字符组合都有1/2/4/8-shot。固定参考 `永和书风骨韵天地` 的嵌套前缀，同字体同字符在各shot及各模型共享初始噪声。

保持K原始采样 CFG1、DPM++20/order2。旧G/I展示的CFG7.5不能直接混入同一matched比较。本版batch4；K1预检12例覆盖三split及四shot，与原单图推理的归一化平均像素误差均<0.0002。批量与单图不声称字节完全相同；K0/K1全量均使用同一新版批量规格，旧K冻结TEST2816/VAL192保留。

远端输出：`/root/projects/hrfont/reports/k_default_k1248_20260917`，实际写到外部盘 `/root/data1/hrfont_default_inference_20260917`。`train_manifest.json`、`val_manifest.json`、`test_manifest.json` 和 `PROTOCOL.json` 锁定具体字符、字体、refs、seed、clean SHA及无效pair原因。后续兼容模型必须复用这些清单，不能按新模型效果换字。Train仅作拟合诊断。

页面同屏显示 Content、GT、K0和候选模型的四种shot，支持split、字体、脚本和多个字符筛选。当前默认执行适配K0/K1；新增K3 checkpoint后按同一manifest增加模型适配，不能只在文档上假定旧加载器支持新模型。

执行顺序：K原定推理完成 → 本版K0/K1全量推理与对照页 → E12-c修复增强与冻结排序实验。E12正式排序仍使用已批准的原704个test字体字符×四shot×两模型，不用新增展示面板反复调评测器。
