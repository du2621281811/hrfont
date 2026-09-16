# I4 完成归档，I5 进入 GPU 预检

2026-09-16 14:01 +08 更新：**I4完整归档已推送f7015d3f8；I5预检通过，正式8卡训练进程已启动。** 数据仍为0913。下文13:57/13:59记录为过程快照。

## I4

已完成10,000成功更新、formal4096与matched4888；此前资源阻塞已恢复，未重复训练或首套推理。完整推理图像、GT、参考字、协议、指标、离线HTML及哈希已从执行机同步到 `reports/experiments/I/I4/`。

传输后逐项核验：**11,831个登记文件SHA256一致，11,741张PNG解码成功且为96×96**；各归档metrics行数分别为2k/4k/6k/8k各192、10k formal4096、matched4888，指标有限。未将图片存在或哈希通过当作风格质量验证。

- [formal4096原图](experiments/I/I4/step00010000/review.html)
- [matched4888原图](experiments/I/I4/step00010000_k1248/review.html)
- [matched指标，test/train/val与shot分开](experiments/I/I4/step00010000_k1248/metrics.json)

I4最终训练记录skips=0、ddp_spread=0。此次是完成与归档交付；是否改善笔触/特效须在同协议比较与视觉审阅后判断，不以训练loss直接宣称有效。

## I5

恢复进程3533245已自动交接到原I5队列；不继承I4或smoke权重。梯度校准完成，冻结render_weight=0.1。

|候选权重|加权render/base梯度比中位数|p95|结果|
|---|---:|---:|---|
|0.1|0.27748|1.41946|通过并选用|
|0.2|0.55496|2.83893|p95超过2|
|0.5|1.38739|7.09732|超过准入范围|

比例来自原执行规格中明确的共享参数子集，不是全模型梯度范数；没有放宽门槛。[完整校准](experiments/I/preflight_i56/calibration.json)。

8卡固定复杂批次已完成300步，skips=0、DDP差0、中文aux=0；正在执行300→304恢复检查。首阶段末步loss=0.0377377，reader/encoder/router梯度均非零。随后仍需实际生成前后诊断与预检门槛；**尚不能宣布预检通过，也没有I5正式训练或泛化效果结果**。

每小时巡检继续。I6不启动；正式I5仅在预检通过后从同一I0独立初始化，执行既定最多10k及4k复核。

13:59补充：恢复304阶段已退出并进入DIAGNOSTIC_AFTER；最终逐rank与图像门槛尚待统一判定，不能仅以恢复进程退出宣称全部预检通过。

## 14:01 预检判定与正式派发

[PREFLIGHT_RESULT](experiments/I/preflight_i56/PREFLIGHT_RESULT.json) 为PASSED，全部8rank达到304/304，零AMP跳步、DDP差0、aux=0；恢复无步骤重放。16个固定训练episode的TC区域误差比为0.817373（下降18.26%），实际DPM生成误差比为0.982827（下降1.72%），9/16例改善；关闭局部通路的中位像素差为0.00183668，图像有限且未全白/全黑。所有原门槛均满足，未修改阈值。

已实际查看[训练前面板](experiments/I/preflight_i56/before/review.png)与[训练后面板](experiments/I/preflight_i56/after/review.png)：TC从接近均匀灰色开始形成字形读出；最终生成变化较小，部分断笔/形状仍保留，少数块状字体出现额外黑块。不能据均值改善宣称空心/笔触问题解决；该伪影作为2k/4k验证重点，并保留既定4k质量门槛。

队列已进入I5_TRAIN，协调PID3533245、正式torchrun PID3537397，run=I5-V0916-S3407，8GPU。正式从I0独立初始化，不继承预检304步；此处确认启动进程，不把初始化阶段写成已完成成功更新。后续小时巡检检查正式步数、梯度/AMP及里程碑。

14:02补充：正式日志已确认10/10成功更新、skips0、DDP差0、aux0。末条loss=0.174330，grad_norm=0.639414，AMP scale1024；reader/encoder/router梯度非零，warmup和gate按计划生效。首10步不是收敛或效果结论。
