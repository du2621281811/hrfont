"""Retry/review H checkpoint inference without rerunning training (torchrun x8)."""
import argparse
import datetime
import json
import os
import torch
import torch.distributed as dist
from h_runtime import T, model_for, DataContext, dataset
from h_eval import evaluate
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--quick',action='store_true')
    a=ap.parse_args()
    rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=45))
    assert dist.get_world_size()==8
    metadata=json.loads((a.checkpoint/'checkpoint.json').read_text())
    assert metadata['complete']
    device=torch.device('cuda',rank)
    model,args=model_for(metadata['arm'],device,a.out/f'load_rank{rank}')
    model.load_train_state(torch.load(a.checkpoint/'ema.pth',map_location=device,weights_only=True))
    data=DataContext(args,device)
    evaluate(model,data,dataset(args,'val'),T.build_ddpm_scheduler(args),a.out,metadata['step'],standard=not a.quick)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
