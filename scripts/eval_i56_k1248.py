"""I5/I6 adapter for the unchanged 26-font x47-char matched-shot protocol."""
import argparse
import datetime
import json
import torch
import torch.distributed as dist
from PIL import Image
import i34_k1248_shared as shared
from i56_runtime import ROOT, model_for, atomic_json, sha256_file


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['I5','I6'],required=True)
    ap.add_argument('--run-id',required=True);a=ap.parse_args()
    run=ROOT/'runs'/a.run_id
    done=json.loads((run/'DONE.json').read_text())
    assert done['arm']==a.arm and done['step']==10000 and done['inference_complete'] and not done['smoke']
    ckpt=run/'global_step_10000';meta=json.loads((ckpt/'checkpoint.json').read_text())
    assert meta['complete'] and meta['step']==10000 and meta['arm']==a.arm
    def load(arm,device,scratch):
        model,args=model_for(arm,device,scratch/'load')
        model.load_train_state(torch.load(ckpt/'ema.pth',map_location=device,weights_only=True))
        return model,args,10000
    shared.load_model=load
    dist.init_process_group('nccl',timeout=datetime.timedelta(hours=3))
    assert dist.get_world_size()==8
    try:
        shared.run_arm(a.arm,(1,2,4,8),False)
        dist.barrier()
        if dist.get_rank()==0:
            jobs=shared.jobs_for(a.arm,(1,2,4,8));assert len(jobs)==4888
            for job in jobs:
                with Image.open(job['out']) as im: im.load();assert im.size==(96,96)
            atomic_json(run/'K1248_DONE.json',dict(status='completed',images=4888,
                weight_sha256=sha256_file(ckpt/'ema.pth'),protocol_sha256=sha256_file(shared.OUT/'PROTOCOL.json')))
        dist.barrier()
    finally:dist.destroy_process_group()


if __name__=='__main__':main()

