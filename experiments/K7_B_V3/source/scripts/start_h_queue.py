import os
import subprocess
from pathlib import Path
ROOT=Path('/root/projects/hrfont')
CODE=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/h_20260915'
if (OUT/'queue.pid').exists():
    raise RuntimeError('queue already launched: inspect, never duplicate')
with (OUT/'queue.log').open('x') as log:
    p=subprocess.Popen(['/root/miniforge3/envs/boogu/bin/python',str(CODE/'scripts/queue_h_20260915.py'),'--execute'],
        cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1'))
(OUT/'queue.pid').write_text(str(p.pid)+'\n')
print('H_QUEUE_STARTED',p.pid,CODE)
