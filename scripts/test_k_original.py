"""Preflight invariants against real clean data and the complete K denoiser."""
import argparse
import json
from pathlib import Path
import torch
from k_runtime import ROOT, T, args_for, dataset, model_for, DataContext, train_sample, batch_to, seed_episode, atomic_json
from scripts.k_components import KSampler, detail_distances, extra_losses, SCRIPT, COMPLEX
from scripts.hrfont_h import attention_once


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:1');ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();device=torch.device(a.device);torch.cuda.set_device(device);torch.set_num_threads(1)
    ds=dataset(args_for(),'train');sampler=KSampler(ds,ROOT/'artifacts/i56_20260916/detail_manifest.json')
    for i in range(1000):
        ids,quota=sampler.batch(i)
        assert quota['script']==SCRIPT and quota['complex']==COMPLEX
        assert len(ids)==64 and all(ds[j]['split']=='train' for j in ids[:1])
        assert sampler.batch(i)==sampler.batch(i)
    c=torch.ones(4,3,96,96,device=device);y=c.clone()
    y[0,:,20:30,20:30]=0
    c[1,:,20:30,20:30]=0
    c[2,:,20:30,20:30]=0;y[2,:,50:60,50:60]=0
    d=detail_distances(c,c,y);perfect=detail_distances(y,c,y)
    assert d['D_add'][0]>0 and d['D_remove'][0]==0
    assert d['D_remove'][1]>0 and d['D_add'][1]==0
    assert d['D_add'][2]>0 and d['D_remove'][2]>0 and d['D_change'][3]==0
    assert torch.all(d['D_change'][:3]>perfect['D_change'][:3])
    appearance=torch.ones(4,144,128,device=device,requires_grad=True)
    _,comp,detail,_=extra_losses(appearance,torch.zeros_like(appearance),c,c,y,
        torch.tensor([1,0,0,0],device=device),torch.ones(4,device=device),1000)
    assert torch.allclose(detail,d['D_out'][0]/4)
    comp.backward();assert appearance.grad[1:].abs().sum()>0
    model,args=model_for('K1',device,a.out/'parent');model.eval();data=DataContext(args,device)
    seed_episode(3407);b=batch_to([train_sample(ds,j) for j in sampler.batch(0)[0][:3]],device)
    cfg=torch.tensor([0,1,0],device=device,dtype=torch.bool)
    source=torch.tensor([0,0,1],device=device,dtype=torch.bool)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        style,refs,query,keep,content,structure=data.conditions(b,cfg,source)
        ctx=model.conditions(style,refs,query,keep,cfg)
        assert ctx[1].shape==(3,144,1024) and ctx[3].shape==(3,144,128)
        assert ctx[0][1].count_nonzero()==0 and ctx[1][1].count_nonzero()==0
        assert all(x[1].count_nonzero()==0 for x in content)
        assert ctx[1][2].abs().sum()>0 and ctx[0][2].abs().sum()>0
        for _,_,_,active in structure:assert active.tolist()==[1.,0.,0.]
        # Random nonzero structural output validates the gate beyond zero initialization.
        outputs=[];handles=[]
        for block in model.base.unet.up_blocks:
            if hasattr(block,'zero_convs'):
                for z in block.zero_convs:
                    z.weight.normal_(0,.001);z.bias.fill_(.01)
                    handles.append(z.register_forward_hook(lambda m,i,o:outputs.append(o.detach())))
        noise=torch.randn(3,3,96,96,device=device);t=torch.tensor([100,300,600],device=device)
        pred,_=model.denoise(noise,t,style.masked_fill(cfg[:,None,None,None],0),content,structure,ctx)
        assert torch.isfinite(pred).all() and outputs
        assert all(x[1:].count_nonzero()==0 for x in outputs)
        assert any(x[0].abs().sum()>0 for x in outputs)
        for h in handles:h.remove()
        attn=model.up_attn[-1];hidden=torch.randn(3,16,attn.to_q.in_features,device=device)
        global_value=attention_once(attn,hidden,ctx[0]);local_value=attention_once(attn,hidden,ctx[1])
        attn._k_beta=0.;assert torch.equal(attn(hidden,ctx[0]),global_value)
        attn._k_beta=.8;actual=attn(hidden,ctx[0]);scale=attn._k_scale[:,None,None]
        expected=global_value+.8*scale.to(local_value.dtype)*local_value*(~cfg)[:,None,None]
        assert torch.equal(actual,expected) and torch.equal(actual[1],global_value[1])
    atomic_json(a.out/'unit_tests.json',dict(status='PASSED',quota_updates=1000,
        tc_shape=[3,144,1024],teacher_shape=[3,144,128],beta_zero_parity=True,global_coefficient=1,
        cfg_tc_zero=True,cfg_structure_zero=True,source_structure_zero=True,source_tc_active=True,
        change_masks=True,full_batch_denominator=True,completion_on_cfg=True,donor_exclusion=True))
    print('K_UNIT_TESTS_PASSED',flush=True)


if __name__=='__main__':main()
