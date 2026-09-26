"""After primary K queue: full frozen VAL192 at 10k, including outline glyphs.

This supplementary review never changes training, checkpoint selection, or the
primary clean test metric. It uses all 192 validation rows, not cherry-picks.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1]
ROOT=Path('/root/projects/hrfont')
CONTROL=ROOT/'reports/k_original_queue_20260917'
RUN=ROOT/'runs/K1-ORIGINAL-V0917-S3407'


def main():
    import fcntl
    lock=(CONTROL/'final_val192.LOCK').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(stage,**extra):
        p=CONTROL/'final_val192.status.tmp'
        p.write_text(json.dumps(dict(stage=stage,time=time.time(),
            helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),**extra),indent=2))
        p.replace(CONTROL/'final_val192.status.json')
    status('WAITING_FOR_PRIMARY_QUEUE')
    deadline=time.time()+36*3600
    while time.time()<deadline:
        if (CONTROL/'STOP').exists() or (RUN/'STOP').exists():status('STOPPED');return
        state=json.loads((CONTROL/'status.json').read_text())['stage']
        if state=='COMPLETED':break
        if state in ['FAILED','STOPPED','RUN_STOP_PENDING'] or state.startswith('STOPPED_BEFORE_'):
            status('PRIMARY_QUEUE_NEEDS_ATTENTION',primary_stage=state);return
        time.sleep(15)
    else:status('WAIT_TIMEOUT');return
    checkpoint=RUN/'global_step_10000'
    assert json.loads((RUN/'DONE.json').read_text())['step']==10000
    out=ROOT/'reports/k1_final_val192_20260917'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',PYTHONWARNINGS='ignore::FutureWarning')
    command=[str(Path(sys.executable).parent/'torchrun'),'--standalone','--nproc_per_node=8',
        'scripts/k_eval.py','--arm','K1','--checkpoint',str(checkpoint),
        '--manifest',str(ROOT/'experiments/K/K_VAL192.json'),'--out',str(out)]
    if not (out/'DONE.json').exists():
        status('FINAL_VAL192_INFERENCE')
        with (CONTROL/'FINAL_VAL192.log').open('a') as log:
            result=subprocess.run(command,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:status('FAILED',returncode=result.returncode);return
    from k_runtime import code_identity
    identity=code_identity()
    from k_eval import make_board
    board=ROOT/'reports/k1_final_val192_review'
    make_board(ROOT/'reports/k0_original_val192_20260917',out,board)
    status('COMPLETED',code_commit=identity['commit'],rows=192,review=str(board/'index.html'))


if __name__=='__main__':main()
