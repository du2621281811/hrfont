# K5-B 分支消融验收

192/192 PNG SHA256 通过。48 个 fresh original-forward FP32 baseline 与之前 chunk FP32 Top10 的最大像素差为 1/255；权重、字符、参考、seed、donor 列表一致。浏览器图像加载与筛选通过，GT 对比抽检未见空心细节改善。

去除普通字体重复后，共 36 个字体/字符/seed 配对。完整配对结果见 PAIRED_AUDIT.json；SUMMARY.csv 保留原始 48 case 权重，不用于跨字体总平均。

- 仅 Ec 或仅训练 gate 后的 Es：平均像素改变量为 0.012–0.034 个 uint8 灰度级，局部最大可达 99，说明不是严格无影响，但没有稳定质量改善。两个空心字体的 L1 改善方向不一致，最大绝对均值变化约 0.0000215。
- 将组合 offset 直接置零：平均像素改变量 0.00017–0.00052，最大 2 个灰度级。这比先前 Delta-off 明显小；二者不是同一个干预（关闭完整 Delta 路径不等于仅将 offset 置零），不能混称。
- 结合 18 轨迹的强相消证据，当前固定面板支持“训练后的组合 offset 对输出影响极弱”。但单分支解除相消也没有恢复所需效果，因此不能把修复简化成删 Ec、删 Es 或调 gate。

建议保留当前训练权重与配方，后续定位 offset 后的采样/读取敏感性及 Delta-off 实际改变的其他路径。此次结果不证明 loss 导致相消，也不解释 K5-A 全部不敏感现象。尚未覆盖新字符与 1/2/8shot，不外推整个测试集。K6 仍等待科学修订 review。

看板：outputs/K5_BRANCH_ABLATION_20260920/index.html（引用相邻 K5_BANK_DIAGNOSTIC_20260919/assets）。

## Delta-off attribution correction
Remote KModel.denoise sets zero_convs._k_active from structure enabled. Therefore no_delta gates the entire structural residual, not just offsets.

Frozen FZBangSKLTJW/A, 4shot, seeds 3407/93407: zeroing zero_conv outputs while retaining donor/offset computation produces exactly identical PNGs to no_delta (both seeds). Offset-only zero differs from no_delta by mean 3.02488/2.03852 uint8 levels. All eight new PNG SHA256 checks pass.

Thus the residual pathway matters in this case; previous Delta-off differences do not establish effective donor-dependent guidance. Offset-only weak sensitivity remains supported. Do not generalize this one-character control to all fonts or K5-A. No training changes.
