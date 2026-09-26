"""Read-only compact monitor; no model loading or image scans on hourly ticks."""
import json
import os
from pathlib import Path
import shutil
import time

ROOT=Path('/root/projects/hrfont');OUT=ROOT/'reports/i34_20260916'
def read(p):
    try:return json.loads(p.read_text())
    except FileNotFoundError:return None
def alive(pid):return bool(pid and Path(f'/proc/{pid}').exists())
def main():
    now=time.time();state=read(OUT/'state.json') or {};result=dict(time=now,queue=state,
        coordinator_alive=alive(state.get('pid')),child_alive=alive(state.get('child_pid')),
        failure=read(OUT/'failure.json'),root_free_gib=round(shutil.disk_usage(ROOT).free/2**30,2),
        checkpoint_free_gib=round(shutil.disk_usage('/root/data1/hrfont_i34_20260915').free/2**30,2),runs={})
    for arm in ('I3','E12C','I4'):
        folder=ROOT/'runs'/f'{arm}-V0916-S3407';h=read(folder/'heartbeat.json') or {}
        result['runs'][arm]=dict(heartbeat={k:h[k] for k in ('state','stage','step','attempt','loss','skips','ddp_spread','seconds','time') if k in h},
            done=read(folder/'DONE.json'),k1248=read(folder/'K1248_DONE.json'),
            stopped=(folder/'STOP').exists(),archive_ready=read(OUT/(arm+'_ARCHIVE_READY.json')))
    print(json.dumps(result,separators=(',',':')))
if __name__=='__main__':main()
