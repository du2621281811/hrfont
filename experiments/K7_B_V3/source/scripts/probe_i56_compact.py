"""Read-only low-context I4 -> I5 status, without loading weights or image archives."""
import json
import os
from pathlib import Path
import shutil
import time

ROOT=Path('/root/projects/hrfont');OUT=ROOT/'reports/i56_20260916';PREP=ROOT/'artifacts/i56_20260916'
def read(p):
    try:return json.loads(p.read_text())
    except FileNotFoundError:return None
def alive(pid):
    if not pid:return False
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
def main():
    state=read(OUT/'state.json') or {}
    result=dict(time=time.time(),queue=state,coordinator_alive=alive(state.get('pid')),
        child_alive=alive(state.get('child_pid')),failure=read(OUT/'failure.json'),
        root_free_gib=round(shutil.disk_usage(ROOT).free/2**30,2),
        checkpoint_free_gib=round(shutil.disk_usage('/root/data1').free/2**30,2),runs={})
    for arm in ('I4','I5'):
        r=ROOT/'runs'/f'{arm}-V0916-S3407';h=read(r/'heartbeat.json') or {}
        result['runs'][arm]=dict(heartbeat={k:h[k] for k in ('state','step','attempt','loss','skips','ddp_spread','time') if k in h},
            done=read(r/'DONE.json'),k1248=read(r/'K1248_DONE.json'),stopped=read(r/'STOPPED.json'),
            review4k=read(r/'REVIEW_4K.json'),archive_ready=read(ROOT/'reports'/('i34_20260916' if arm=='I4' else 'i56_20260916')/(arm+'_ARCHIVE_READY.json')))
    calibration=read(PREP/'calibration.json') or {}
    result['calibration']={k:calibration[k] for k in ('status','render_weight','candidates') if k in calibration}
    pre=read(PREP/'PREFLIGHT_RESULT.json') or {}
    result['preflight']={k:pre[k] for k in ('status','checks','tc_ratio','actual_generation_ratio','improved_episodes','local_intervention_median_l1') if k in pre}
    print(json.dumps(result,separators=(',',':')))
if __name__=='__main__':main()
