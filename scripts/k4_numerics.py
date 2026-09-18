"""Rank-synchronous same-batch FP32 recovery; never silently drop an episode."""
import json
import torch
import torch.distributed as dist


def finite_on_all_ranks(loss):
    good=torch.isfinite(loss.detach()).all().to(dtype=torch.int32)
    dist.all_reduce(good,op=dist.ReduceOp.MIN)
    return bool(good.item())


def guarded_forward(forward, output, metadata):
    # forward(amp) returns (scalar loss, arbitrary payload, diagnostic scalars).
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
    result=forward(True)
    if finite_on_all_ranks(result[0]):return result
    original={k:float(v.detach()) for k,v in result[2].items()}
    del result
    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
    result=forward(False)
    good=finite_on_all_ranks(result[0])
    record=dict(**metadata,rank=dist.get_rank(),fp16=original,
                fp32={k:float(v.detach()) for k,v in result[2].items()},fp32_all_ranks_finite=good)
    with (output/f'numerical_recovery_rank{dist.get_rank()}.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n')
    if not good:raise RuntimeError('Nonfinite loss persists in synchronized FP32 replay; see numerical_recovery logs')
    return result
