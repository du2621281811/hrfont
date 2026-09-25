"""Start one detached eight-GPU cache build after the G switch."""
import os
import subprocess
from pathlib import Path

ROOT = Path('/root/projects/hrfont')
CODE = Path(__file__).resolve().parents[1]
PY = '/root/miniforge3/envs/boogu/bin/python'
out = ROOT / 'reports/h_20260915'
out.mkdir(parents=True, exist_ok=True)
if (out/'cache.pid').exists():
    raise RuntimeError('cache launch already recorded; inspect before retry')
with (out/'cache.log').open('x') as log:
    p = subprocess.Popen([PY, '-m', 'torch.distributed.run', '--nproc_per_node=8',
        '--master_port=29571', str(CODE/'scripts/prepare_h_cache.py')], cwd=CODE,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', OMP_NUM_THREADS='1', NCCL_IB_DISABLE='1', PYTHONUNBUFFERED='1'))
(out/'cache.pid').write_text(str(p.pid)+'\n')
print('H_CACHE_STARTED', p.pid, CODE)
