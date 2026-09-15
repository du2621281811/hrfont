# I2 中文中性内容扩展

这里保存338张新生成的中性汉字PNG及`COMPLETE.json`，用于审阅I2训练输入，不是模型推理结果。冻结Ec特征`features.pt`仅存服务器，哈希与绑定编码器在manifest中。

- 固定Noto字体文件哈希、83px字号、96px画布，Pillow12.2/FreeType2.14.3。
- 338张图通过非空、前景边界检查；目标风格图仍从清洗后的训练字体StyleImage取，ref排除目标字符。
- V0913主数据不变。本扩展不宣称逐像素复现旧西文ContentImage；对全部295旧字符的重绘误差审计保存在`renderer_checks`，86字符完全一致，少数字符差异较明显，原因未确定。
- 中文分支关闭Delta，仅辅助共享生成器和在线局部TC学习。同版本新锚点应用于全部训练字体，不混入目标字体风格。

检查示例：[永](u6C38.png)、[和](u548C.png)、[书](u4E66.png)、[风](u98CE.png)。完整字符与PNG哈希见 [manifest](COMPLETE.json)。
