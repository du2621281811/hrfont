"""Real I0 checkpoint: gradient, padding/order, CFG, DPM checks. Not quality evidence."""
import torch
from PIL import Image
from i_runtime import ROOT,T,model_for,dataset,DataContext,batch_to,atomic_json
from i_eval import sample

def main():
    torch.set_num_threads(1)
    device=torch.device('cuda:0')
    out=ROOT/'reports/i_20260915/integration'; out.mkdir(parents=True,exist_ok=True)
    model,args=model_for('I1',device,out/'parent')
    model.eval(); ds=dataset(args,'val'); data=DataContext(args,device)
    chosen=next(i for i,p in enumerate(ds.target_images) if p.endswith('+u0041.png'))
    s=ds[chosen]; s['ref_chars']=s['ref_chars'][:1]; s['ref_image_paths']=s['ref_image_paths'][:1]
    b=batch_to([s],device); cfg=torch.zeros(1,device=device,dtype=torch.bool)
    style,refs,q,keep,content,structure=data.conditions(b,cfg,cfg)
    print('DONORS',[(float(d.std()),float(d.std(1).mean()),a.tolist(),v.tolist()) for d,a,c,v in structure],flush=True)
    x=torch.randn(1,3,96,96,device=device); t=torch.tensor([200],device=device)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        one=model(x,t,style,refs,q,keep,content,structure,cfg,1.)[0]
        eight=model(x,t,style,refs.expand(-1,8,-1,-1,-1),q,keep.expand(-1,8),content,structure,cfg,1.)[0]
        torch.testing.assert_close(one,eight,atol=.003,rtol=.003)
        reversed_set=[(d.flip(1),a.flip(1),c,v) for d,a,c,v in structure]
        perm=model(x,t,style,refs,q,keep,content,reversed_set,cfg,1.)[0]
        torch.testing.assert_close(one,perm,atol=.003,rtol=.003)
        # Independent unconditional passes with different reference images must agree.
        zero_content=[torch.zeros_like(c) for c in content]
        disabled=[(d,a,c,torch.zeros_like(v)) for d,a,c,v in structure]
        un1=model(x,t,style,refs,q,keep,zero_content,disabled,~cfg,1.)[0]
        un2=model(x,t,style,refs.neg(),q,keep,zero_content,disabled,~cfg,1.)[0]
        torch.testing.assert_close(un1,un2,atol=1e-5,rtol=1e-5)
        img=sample(model,data,b,T.build_ddpm_scheduler(args),3407)
        Image.fromarray(img).save(out/'untrained_smoke.png')
    # Nonzero RSI gates for a synthetic gradient-path test (not saved to training).
    for name,p in model.named_parameters():
        if 'zero_convs.' in name and name.endswith('weight'):
            with torch.no_grad(): p.add_(.001)
    with torch.autocast('cuda',dtype=torch.float16):
        y,offset,appearance=model(x,t,style,refs,q,keep,content,structure,cfg,1.)
        loss=y.float().square().mean()+.01*appearance.float().square().mean()+.01*offset.float()
    (loss*1024).backward()
    grads={}
    for label,select in [('online_encoder',lambda n:n.startswith('reader.blocks.')),
                         ('reader',lambda n:n.startswith('reader.')),
                         ('router',lambda n:'sc_interpreter_offsets.' in n and '.original.' not in n)]:
        grads[label]=sum(float((p.grad.float()/1024).square().sum()) for n,p in model.named_parameters() if select(n) and p.grad is not None)**.5
        assert grads[label]>0 and torch.isfinite(torch.tensor(grads[label])), (label,grads)
    assert all(p.grad is None for p in model.base.style_encoder.parameters())
    atomic_json(out/'PASS.json',dict(duplicate8_max_abs=float((one-eight).abs().max()),
        permutation_max_abs=float((one-perm).abs().max()),cfg_max_abs=float((un1-un2).abs().max()),
        gradients=grads,DPM20='PASS',note='interface/gradient smoke, not visual quality; RSI gate temporarily .001'))
    print('I_INTEGRATION_PASS',grads,flush=True)

if __name__=='__main__': main()
