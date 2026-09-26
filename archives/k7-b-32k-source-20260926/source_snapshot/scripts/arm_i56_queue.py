"""Seal CPU checks and code fingerprint; this is NOT GPU preflight acceptance."""
import json
import os
from pathlib import Path
import subprocess
import time
from queue_i56_20260916 import CODE,PREP,ROOT,PY,sha,write


def main():
    assert not (PREP/'ARMED.json').exists(),'Existing seal is immutable; inspect before recovery'
    assert not (ROOT/'runs/I5-V0916-S3407').exists()
    assert json.loads((PREP/'detail_manifest.json').read_text())['status']=='REVIEWED'
    required=('train_i56.py','eval_i56_k1248.py','calibrate_i56.py','diagnose_i56.py','verify_i56_preflight.py',
        'i56_validation_gate.py','archive_i34_k1248.py','seal_i_eval_archive.py','queue_i56_20260916.py')
    assert all((CODE/'scripts'/n).is_file() for n in required)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    commands=[[PY,'-m','unittest','scripts.test_i56_components','-v'],
        [PY,'scripts/train_i56.py','--help'],[PY,'scripts/eval_i56_k1248.py','--help'],
        [PY,'scripts/calibrate_i56.py','--help'],[PY,'scripts/diagnose_i56.py','--help']]
    checks=[]
    for i,cmd in enumerate(commands):
        proc=subprocess.run(cmd,cwd=CODE,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log=PREP/f'cpu_check_{i}.log'
        with log.open('x') as f:f.write(proc.stdout)
        assert proc.returncode==0,(cmd,proc.stdout[-3000:])
        checks.append(dict(command=cmd,exit_code=proc.returncode,log_sha256=sha(log)))
    write(PREP/'ARMED.json',dict(status='CPU_READY_GPU_PENDING',time=time.time(),authorized_arms=['I5'],
        run_id='I5-V0916-S3407',i6='Implemented shared interface; not dispatched or authorized for training',
        cpu_tests_passed=True,checks=checks,detail_sha256=sha(PREP/'detail_manifest.json'),
        code_sha256={str(p.relative_to(CODE)):sha(p) for p in sorted(CODE.rglob('*.py'))},
        sequence='WaitI4 -> gradient calibration -> before diagnostic -> DDP300 -> resume304 -> after diagnostic -> gate -> independent I0 I5<=10k -> both inference protocols -> archive',
        notice='No claim of GPU training success until PREFLIGHT_RESULT and live update evidence exist'))
    print('CPU_READY_GPU_PENDING',flush=True)


if __name__=='__main__':main()
