# Training status — 2026-09-09 15:15 CST

## Cause of earlier stop
Machine reboot (~23 min uptime when discovered). Watchdog had no systemd autostart, so F1/F3b did not resume until manually restarted.

## Live (session-independent)

| Job | GPU | State | Progress | Notes |
|---|---|---|---|---|
| **F1** | 2 | relaunched → running | resume from `last_state` (~26500/80k) | watchdog relaunch 07:12Z |
| **F3b** | 0 | relaunched → running | resume from `last_state` (~9400/80k) | watchdog relaunch 07:13Z |
| **E12 φ** | — | **DONE** | early-stop@2250 best_auc≈0.9009 | |
| **E12 membership** | — | **DONE** | test ROC≈0.937 PR≈0.917 | |
| **Watchdog** | — | setsid | pid file `reports/watchdog_f1_f3b_e12/watchdog.pid` | `bash scripts/start_watchdog_f1_f3b_e12.sh` |

## Continuity
- Watchdog: `scripts/watchdog_f1_f3b_e12.py` (SIGHUP ignored, 60s poll)
- Relaunch: `bash scripts/start_watchdog_f1_f3b_e12.sh`
