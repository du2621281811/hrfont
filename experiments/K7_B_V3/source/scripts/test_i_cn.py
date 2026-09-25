"""CPU I2 episode/gradient smoke while I1 owns the GPUs; DDP20 follows in queue."""
import torch
from i_runtime import ROOT,T,model_for,DataContext,dataset,atomic_json,seed_episode

def main():
    torch.set_num_threads(1);seed_episode(3407)
    out=ROOT/'reports/i_20260915/cn_preflight';out.mkdir(parents=True,exist_ok=True)
    model,args=model_for('I2',torch.device('cpu'),out/'parent')
    data=DataContext(args,torch.device('cpu'));data.enable_cn()
    ds=dataset(args,'train');b=data.cn_batch(ds,1)
    cfg=torch.zeros(1,dtype=torch.bool)
    style,refs,q,keep,content,structure=data.conditions(b,cfg,cfg)
    assert all(not active.any() for d,a,c,active in structure)
    assert b['char_cp'][0] not in b['ref_chars'][0]
    assert b['font_stem'][0] in {__import__('pathlib').Path(p).parent.name for p in ds.target_images}
    model.train();model.base.style_encoder.eval();model.base.content_encoder.eval()
    y,offset,pred=model(torch.randn_like(b['target_image']),torch.tensor([300]),style,refs,q,keep,content,structure,cfg,1.)
    loss=(y-b['target_image']).square().mean()+pred.square().mean()*.01
    loss.backward()
    grads=sum(float(p.grad.square().sum()) for n,p in model.named_parameters() if n.startswith('reader.blocks.') and p.grad is not None)**.5
    assert grads>0 and torch.isfinite(loss) and float(offset)==0
    atomic_json(out/'PASS.json',dict(font=b['font_stem'][0],target=b['char_cp'][0],refs=b['ref_chars'][0],
        excluded_target_from_ref=True,delta_offset=float(offset),online_encoder_grad=grads,
        loss=float(loss),note='CPU interface/gradient test, synthetic loss; not quality or DDP proof'))
    print('I_CN_CPU_PASS',grads)

if __name__=='__main__':main()
