"""Eight-rank injected FP16 overflow, same-batch replay, RNG and failure tests."""
import os,json,tempfile
from pathlib import Path
import torch
import torch.distributed as dist
from scripts.k4_numerics import guarded_forward
from k4_runtime import STORE,code_identity,atomic_json

def main():
    rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']));dist.init_process_group('nccl')
    out=STORE/'control/numerical_test';out.mkdir(exist_ok=True,parents=True)
    x=torch.tensor(2.,device='cuda',requires_grad=True);calls=[];randoms=[]
    def objective(amp):
        calls.append(amp);randoms.append(torch.rand(2,device='cuda').clone())
        # One rank fails in FP16. All ranks must replay before any backward.
        dtype=torch.float16 if amp else torch.float32
        factor=torch.tensor(40000. if rank==6 else 1.,device='cuda',dtype=dtype)
        loss=x.to(dtype)*factor
        return loss,None,dict(loss=loss)
    loss,_,_=guarded_forward(objective,out,dict(test='rank6_overflow'))
    assert calls==[True,False] and torch.equal(*randoms)
    loss.backward();assert torch.isfinite(x.grad) and float(x.grad)==(40000. if rank==6 else 1.)
    def bad(amp):
        v=x*float('nan') if rank==3 else x
        return v,None,dict(loss=v)
    failed=False
    try:guarded_forward(bad,out,dict(test='persistent_failure'))
    except RuntimeError:failed=True
    assert failed
    calls.clear()
    def healthy(amp):
        calls.append(amp);return x*2,None,dict(loss=x*2)
    guarded_forward(healthy,out,dict(test='finite_fast_path'));assert calls==[True]
    dist.barrier()
    if rank==0:atomic_json(STORE/'control/NUMERICAL_RECOVERY_TEST_PASSED.json',dict(status='PASS',ranks=dist.get_world_size(),single_rank_overflow_replayed_on_all_ranks=True,same_rng=True,finite_backward=True,persistent_failure_synchronized=True,healthy_path_unchanged=True,identity=code_identity()))
    dist.destroy_process_group()
if __name__=='__main__':main()
