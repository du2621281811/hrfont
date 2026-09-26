"""Real-model inference and completed eight-rank smoke acceptance, before dispatch."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from i34_runtime import ROOT,CODE,model_for,DataContext,dataset,batch_to,T,sha256_file,atomic_json
from i_eval import episodes,sample

PREP=ROOT/'artifacts/i34_20260915'


def main():
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device)
    evidence={}
    for arm,limit in (('I3',84),('I4',10)):
        run=ROOT/'runs'/f'{arm}-PREFLIGHT-V0916-S3407'
        done=json.loads((run/'DONE.json').read_text());assert done['smoke'] and done['step']==limit
        ranks=[]
        for rank in range(8):
            rows=[json.loads(line) for line in (run/f'rank{rank}.jsonl').read_text().splitlines()]
            last=rows[-1];assert last['step']==limit and last['skips']==0 and last['ddp_spread']==0
            for key in ('loss','reader_grad','encoder_grad','router_grad','token_projection_grad','ink_readout_grad'):
                assert math.isfinite(last[key]) and last[key]>0,(arm,rank,key,last[key])
            if arm=='I4':assert last['aux_loss_rank']>0
            ranks.append(last)
        logs=[json.loads(line) for line in (run/'train_log.jsonl').read_text().splitlines()]
        first=np.median([r['loss'] for r in logs[:10]]);last=np.median([r['loss'] for r in logs[-10:]])
        if arm=='I3':
            assert last<first
            assert [r['step'] for r in logs]==list(range(1,85)), 'Resume must continue at 81 without replay'
            assert json.loads((run/'config.json').read_text())['resume'] is not None
        model,args=model_for(arm,device,PREP/f'{arm}_inference_load')
        model.load_train_state(torch.load(run/f'global_step_{limit}/ema.pth',map_location=device,weights_only=True));model.eval()
        data=DataContext(args,device);ds=dataset(args,'val');job=episodes(ds)[0]
        row=ds[job['index']];row['ref_chars']=job['refs'][:1]
        row['ref_image_paths']=[str(ds.style_by_font_char[job['font']][c]) for c in row['ref_chars']]
        batch=batch_to([row],device);mask=torch.zeros(1,dtype=torch.bool,device=device)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            style,refs,q,keep,_,_=data.conditions(batch,mask,mask)
            tok1,_=model.reader(refs,q,keep)
            tok8,_=model.reader(refs.repeat(1,8,1,1,1),q,keep.repeat(1,8))
            torch.testing.assert_close(tok1,tok8,rtol=.02,atol=.002)
            pred=sample(model,data,batch,T.build_ddpm_scheduler(args),3407,gate=1.)
        Image.fromarray(pred).save(PREP/f'{arm}_smoke_inference.png')
        assert pred.shape==(96,96,3) and pred.dtype==np.uint8
        evidence[arm]=dict(updates=limit,ranks=8,skips=0,ddp_spread=0,
            loss_first10_median=float(first),loss_last10_median=float(last),
            repeated_ref_max_error=float((tok1-tok8).abs().max()),
            peak_mib=max(r['peak_mib'] for r in ranks),
            checkpoint_sha256=sha256_file(run/f'global_step_{limit}/ema.pth'),
            train_config_sha256=sha256_file(run/'config.json'))
        del model,data;torch.cuda.empty_cache()
    e12run=ROOT/'runs/E12C-PREFLIGHT-V0916-S3407'
    e12=json.loads((e12run/'DONE.json').read_text())
    assert e12['smoke'] and e12['A_steps']==20 and e12['B_steps']==20
    metrics=json.loads((e12run/'internal_test.json').read_text())
    assert len(metrics['AUC_cells'])==12 and all(math.isfinite(v) for v in metrics['AUC_cells'].values())
    evidence['E12c']=dict(smoke=e12,internal_metrics=metrics,note='Plumbing check only; smoke quality is not a paper result')
    evidence['difficulty_sha256']=sha256_file(PREP/'difficulty_manifest.json')
    evidence['e12c_manifest_sha256']=sha256_file(PREP/'e12c_manifest.json')
    evidence['code_sha256']={str(p.relative_to(CODE)):sha256_file(p) for p in sorted((CODE/'scripts').glob('*.py'))}
    evidence['status']='PASSED'
    atomic_json(PREP/'PREFLIGHT_PASSED.json',evidence)
    print(json.dumps(evidence,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
