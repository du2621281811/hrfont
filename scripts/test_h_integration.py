"""Actual-parent handoff, N invariance, and DPM inference smoke on V100."""
import json
from pathlib import Path
import torch
from PIL import Image
from h_runtime import ROOT, T, model_for, dataset, DataContext, batch_to, atomic_json
from hrfont_h import attention_once
from h_eval import sample, episodes


def main():
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    out = ROOT/'reports/h_20260915/integration'
    out.mkdir(parents=True,exist_ok=True)
    h0,args = model_for('H0',device,out/'H0')
    h3,_ = model_for('H3',device,out/'H3')
    h0.eval(); h3.eval()
    ds = dataset(args,'val')
    assert len(episodes(ds,False)) == 192
    assert len(episodes(ds,True)) == 4096
    data = DataContext(args,device)
    s = ds[0]
    s['ref_chars'] = s['ref_chars'][:1]
    b = batch_to([s],device)
    cfg = torch.zeros(1,device=device,dtype=torch.bool)
    style,refs,q,keep,content,structure = data.conditions(b,cfg,cfg)
    x = torch.randn(1,3,96,96,device=device)
    t = torch.tensor([200],device=device)
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
        y0 = h0(x,t,style,refs,q,keep,content,structure,cfg,0.)[0]
        y3 = h3(x,t,style,refs,q,keep,content,structure,cfg,0.)[0]
        torch.testing.assert_close(y0,y3,atol=1e-5,rtol=1e-5)
        one = h3(x,t,style,refs,q,keep,content,structure,cfg,1.)[0]
        eight = h3(x,t,style,refs.expand(-1,8,-1,-1),q,keep.expand(-1,8),content,structure,cfg,1.)[0]
        torch.testing.assert_close(one,eight,atol=.002,rtol=.002)
        attention = h0.up_attn[0]
        hidden = torch.randn(1,5,attention.to_q.in_features,device=device)
        g = torch.randn(1,9,1024,device=device)
        for count in (16,144):
            ref = torch.randn(1,count,1024,device=device)
            ctx1 = torch.cat([g,ref],1)
            ctx8 = torch.cat([g,ref.repeat(1,8,1)],1)
            a = attention_once(attention,hidden,ctx1,torch.ones(ctx1.shape[:2],device=device,dtype=torch.bool),count)
            z = attention_once(attention,hidden,ctx8,torch.ones(ctx8.shape[:2],device=device,dtype=torch.bool),count)
            torch.testing.assert_close(a,z,atol=.001,rtol=.001)
        image = sample(h3,data,b,T.build_ddpm_scheduler(args),3407,1.)
        assert image.shape==(96,96,3)
        Image.fromarray(image).save(out/'untrained_h3_inference_smoke.png')
    atomic_json(out/'PASS.json',dict(gate0_max_abs=float((y0-y3).abs().max()),
        duplicate8_max_abs=float((one-eight).abs().max()), N_16_144='PASS', DPM20='PASS',
        quick_episodes=192, standard_episodes=4096, note='untrained interface tests, not quality evidence'))
    print('H_INTEGRATION_PASS', flush=True)


if __name__=='__main__':
    main()
