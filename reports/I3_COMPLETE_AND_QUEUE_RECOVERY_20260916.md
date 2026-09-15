# I3完成归档与E12-c接续

2026-09-16小时巡检。I3完成10,000成功更新，attempt=10,000、八rank AMP跳步0、DDP参数差0；最终formal4096和matched4888全部完成。没有因为归档故障重新训练或重新推理。

## 已验收结果

- 2k/4k/6k/8k各192张固定验证图；10k formal4096和matched4888分别归档。
- 本地传输后逐文件SHA及native96图像解码通过：共11,831个登记文件、11,741张PNG（包括GT/ref/content，不等于生成图数量）。I0/I1/I2/I3的matched键完全一致，共4888键。
- [逐项验收与test指标](experiments/I/I3/LOCAL_VERIFICATION.json)、[formal4096浏览](experiments/I/I3/step00010000/review.html)、[matched4888浏览](experiments/I/I3/step00010000_k1248/review.html)。

初步像素指标：I3相对I1的test edge_l1在四个shot均略低，但L1/SSIM变化混合，未呈现一致的大幅收益；I2的test L1/SSIM仍更好。不能据此宣称笔触/特效问题已经解决，也不因这些结果自动改变已批准的I4实验。完整视觉review与E12-c/人评另行进行。

## 本次故障与修复

原队列在I3两套推理完成后，尝试封存2k验证归档时退出：执行快照漏部署 `scripts/seal_i_eval_archive.py`。这是部署遗漏，不是训练/推理数值错误。约04:06 +08停止，04:40 +08恢复。

补齐该已存在于Git的CPU验收脚本，新增独立 `scripts/resume_i34_archive_20260916.py`。先取得同一queue.lock并确认旧协调器和worker退出，再校验已有副本与源逐字节一致，补完所有归档；没有覆盖不一致副本、改训练代码、换checkpoint或删除旧模型。恢复dry-run已通过，依赖SHA与旧故障记录保存在 [恢复回执](experiments/I/I3/recovery/ARCHIVE_RECOVERY_READY.json)。

后台恢复协调器PID3476802，E12-c子进程PID3476814。已实测E12-c阶段A到370步，loss2.5219、AMP跳步0，root约15.03GiB、checkpoint盘约120.77GiB。完成A2000+B1000及自动验证后仍由同一恢复协调器启动I4；不跳过E12-c。E12-c训练和自动验证不等于独立人评验证。

巡检继续读取原 `probe_i34_compact.py` 和同一队列状态目录，旧failure已保留为 `failure_before_archive_fix.json`；不根据旧PID或旧failure重新启动队列。小时巡检频率及当前模型未变。
