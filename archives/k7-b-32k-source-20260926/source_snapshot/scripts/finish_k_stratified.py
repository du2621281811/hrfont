"""One-shot CPU statistics dependency for the already authorized K3 pipeline."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    control=Path('/root/projects/hrfont/reports/k3_stratified_20260917')
    control.mkdir(exist_ok=True)
    lock=(control/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(stage,**kw):
        p=control/'status.json';q=p.with_suffix('.tmp')
        q.write_text(json.dumps(dict(stage=stage,time=time.time(),pid=os.getpid(),**kw),indent=2));q.replace(p)
    status('WAIT_FOR_K3_PAPER_METRICS')
    deadline=time.time()+24*3600
    while time.time()<deadline:
        p=Path('/root/projects/hrfont/reports/k3_publish_20260917/status.json')
        s=json.loads(p.read_text())
        if s['stage']=='COMPLETED':break
        if s['stage'] in ['FAILED','DEPENDENCY_FAILED','DEPENDENCY_STOPPED']:
            status('DEPENDENCY_FAILED',dependency=s);return
        time.sleep(30)
    else:
        status('TIMEOUT');return
    status('COMPUTE_STRATA')
    out=Path('/root/data1/hrfont_k3_stratified_results_20260917')
    command=[sys.executable,str(Path(__file__).with_name('k_stratified_report.py')),
        '--metrics','/root/data1/hrfont_k_paper_metrics_20260917','/root/data1/hrfont_k3_paper_metrics_20260917',
        '--e12','/root/data1/hrfont_e12c_r2_20260917',
        '--difficulty','/root/data1/hrfont_k_stratified_20260917',
        '--legacy-difficulty','/root/projects/hrfont/artifacts/i34_20260915/difficulty_manifest.json',
        '--out',str(out)]
    with (control/'statistics.log').open('a') as f:
        rc=subprocess.call(command,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT)
    if rc:status('FAILED',returncode=rc);return
    status('COMPLETED',review=str(out/'index.html'))


if __name__=='__main__':main()
