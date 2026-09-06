# F0–F3 训练看板

实时页（训练机）：[http://127.0.0.1:8787/](http://127.0.0.1:8787/)  
静态快照：本目录，已有 `python -m http.server 8765 --directory reports` 时打开 `/f123_dashboard/`。

启动：

```bash
bash scripts/start_f123_monitor.sh
```

监控进程会：

- 每 10s 读 heartbeat / train_log / GPU / 内存 / 磁盘
- 把 `train_log.jsonl` 拷到 `reports/training_logs/`（runs/ 不进 Git）
- F2/F3 进程消失且无 STOP/DONE 时，从 `last_state` 自动续跑（恢复 optimizer + RNG）
- **不**自动续跑 F1（有 STOP，等空闲 GPU）
- 页面不加载任何 CDN，训练本身也不走外网

5k checkpoint + val 时心跳可能停数分钟，看板会标 `ckpt_val`，不要当成死锁。
