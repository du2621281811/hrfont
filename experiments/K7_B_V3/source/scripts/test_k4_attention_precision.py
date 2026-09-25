"""Real overflow regression plus full-model FP16 forward/backward on saved batch."""
import json,types
from pathlib import Path
import torch
from k4_runtime import *
from scripts.k_components import KSampler

def main():
    torch.set_num_threads(1);device=torch.device('cuda:0');out=STORE/'control/attention_probe'
    report=json.loads((out/'ROOT_CAUSE.json').read_text());saved=torch.load(out/'bad_attention_inputs.pt',map_location=device,weights_only=True)
    model,args=model_for('K4-A',device,out/'fixed_provenance');model.load_train_state(torch.load(out/'model.pth',map_location=device,weights_only=True))
    model.train();model.base.style_encoder.eval();model.base.content_encoder.eval()
    module=model.get_submodule(report['module']);q,k,v=[saved[n].detach().requires_grad_(True) for n in ['query','key','value']]
    with torch.autocast('cuda',dtype=torch.float16):
        old_scores=(q@k.transpose(-1,-2))*module.scale
        assert not torch.isfinite(old_scores).all()
        got=module._attention(q,k,v)
    with torch.autocast('cuda',enabled=False):
        oracle=module.reshape_batch_dim_to_heads(((q.float()@k.float().transpose(-1,-2)*module.scale).softmax(-1)@v.float()).to(q.dtype))
    torch.testing.assert_close(got,oracle,rtol=0,atol=0);assert torch.isfinite(got).all()
    got.float().square().mean().backward();assert all(torch.isfinite(t.grad).all() for t in [q,k,v])
    mask=torch.ones(q.shape[0],1,k.shape[1],device=device,dtype=torch.bool);mask[:,:,-1]=False
    module._slice_size=1
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        masked=module._attention(q,k,v,mask)
        sliced=module._sliced_attention(q,k,v,q.shape[1],q.shape[-1]*module.heads,mask)
        torch.testing.assert_close(sliced,masked,rtol=.002,atol=.002)
    module._slice_size=None
    data=DataContext(args,device);train=dataset(args,'train');sampler=KSampler(train,ASSETS/'detail_manifest.json');scheduler=T.build_ddpm_scheduler(args)
    attempt,rank=report['attempt'],report['rank'];seed_episode(3407+attempt*1000+rank);ids,_=sampler.batch(attempt)
    samples=batch_to([train_sample(train,i) for i in ids[rank*8:(rank+1)*8]],device)
    source=torch.rand(8,device=device)<.05;cfg=torch.rand(8,device=device)<.02;cond=data.conditions(samples,cfg,source,need_refs=True)
    noise=torch.randn_like(samples['target_image']);ts=torch.randint(0,1000,(8,),device=device);noisy=scheduler.add_noise(samples['target_image'],noise,ts)
    with torch.autocast('cuda',dtype=torch.float16):pred,offset,appearance=model(noisy,ts,*cond,cfg,1.)
    assert all(torch.isfinite(t).all() for t in [pred,offset,appearance])
    loss=(pred.float()-noise.float()).square().mean()+.25*offset.float()+.01*appearance.float().square().mean()
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    result=dict(status='PASS',identity=code_identity(),root_cause=report,real_overflow_reproduced=True,attention_fp32_oracle_exact=True,attention_backward_finite=True,sliced_masked_parity=True,full_model_fp16_outputs_and_gradients_finite=True,full_model_loss=float(loss),batch_attempt=attempt,batch_rank=rank)
    atomic_json(STORE/'control/ATTENTION_PRECISION_PASSED.json',result);print('ATTENTION_PRECISION_PASS',float(loss),flush=True)
if __name__=='__main__':main()
