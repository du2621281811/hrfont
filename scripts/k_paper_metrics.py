"""Frozen E12-c and paired LPIPS diagnostics for existing K images; no fitting."""
import argparse, collections, concurrent.futures, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT=Path('/root/projects/hrfont')
STORE=Path('/root/data1/hrfont_k_paper_metrics_20260917')
E12=Path('/root/data1/hrfont_e12c_r2_20260917')
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE/'e12c_r2'))
from model import Encoder

def atomic(p,d):
    p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2));q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_image(p):
    with Image.open(p) as im:
        assert im.size==(96,96),(p,im.size)
        return torch.from_numpy(np.array(im.convert('RGB'),copy=True)).permute(2,0,1).float()/255

def main():
    global STORE
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['K0_K1','K3'],default='K0_K1');args=parser.parse_args()
    if args.arm=='K3':STORE=Path('/root/data1/hrfont_k3_paper_metrics_20260917')
    torch.set_num_threads(4);device=torch.device(os.environ.get('K_METRICS_DEVICE','cuda:0'))
    STORE.mkdir(exist_ok=True);start=time.time()
    if (STORE/'DONE.json').exists():print('ALREADY_DONE');return
    lock=json.loads((E12/'MODEL_LOCK.json').read_text());assert lock['primary']=='R1_cosine'
    refs=[f'u{ord(c):04X}' for c in '永和书风骨韵天地'];rows=[];sources={}
    specs=[('default128',arm,sp,ROOT/'reports/k_default_k1248_20260917'/arm/sp) for arm in ['K0','K1'] for sp in ['train','val','test']]
    specs += [('original47','K0','test',ROOT/'reports/k0_original_test_20260917'),('original47','K1','test',ROOT/'runs/K1-ORIGINAL-V0917-S3407/eval_step_10000')]
    if args.arm=='K3':
        specs=[('default128','K3',sp,ROOT/'reports/k_default_k1248_20260917/K3'/sp) for sp in ['train','val','test']]
        specs += [('original47','K3','test',ROOT/'runs/K3-PURENOISE-V0917-S3407/eval_step_10000')]
    for protocol,arm,sp,d in specs:
        p=d/'metrics.json';sources[str(p)]=sha(p)
        for r in json.loads(p.read_text())['rows']:
            refdir=Path(r['refs_paths'][0]).parent
            fixed=[str(refdir/(r['font']+'+'+c+'.png')) for c in refs]
            assert all(Path(p).is_file() for p in fixed)
            rows.append(dict(**r,protocol=protocol,arm=arm,prediction=str(d/r['png']),e12_refs=fixed))
    e=Encoder(False,False).to(device)
    e.load_state_dict(torch.load(E12/'R1/A_best.pt',map_location=device,weights_only=False)['model']);e.eval().requires_grad_(False)
    paths=sorted({r['prediction'] for r in rows}|{r['target'] for r in rows}|{r['content'] for r in rows}|{p for r in rows for p in r['e12_refs']})
    lookup={p:i for i,p in enumerate(paths)};features=[]
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    pool=concurrent.futures.ThreadPoolExecutor(8)
    with torch.inference_mode():
        for st in range(0,len(paths),128):
            x=torch.stack(list(pool.map(load_image,paths[st:st+128]))).to(device)
            with torch.autocast(device.type,enabled=device.type=='cuda',dtype=torch.float16):z=e((x-mean)/std)
            features.append(z.float().cpu())
            if st%2048==0:atomic(STORE/'heartbeat.json',dict(stage='E12_EMBED',done=st,total=len(paths),time=time.time()))
    feats=torch.cat(features);del e
    import lpips
    net=lpips.LPIPS(net='alex',version='0.1').to(device).eval().requires_grad_(False)
    with torch.inference_mode():
        for st in range(0,len(rows),32):
            rr=rows[st:st+32];a=torch.stack(list(pool.map(load_image,[r['prediction'] for r in rr]))).to(device)
            b=torch.stack(list(pool.map(load_image,[r['target'] for r in rr]))).to(device)
            lp=net(a*2-1,b*2-1).flatten().cpu().tolist()
            for r,d in zip(rr,lp):
                ref=F.normalize(feats[[lookup[p] for p in r['e12_refs']]].mean(0),dim=0)
                r.update(lpips_alex=float(d))
                for tag,key in [('family','prediction'),('gt_family','target'),('content_family','content')]:
                    c=float(feats[lookup[r[key]]]@ref);r[tag+'_cosine']=c;r[tag+'_s01']=(1+c)/2
            if st%1024==0:atomic(STORE/'heartbeat.json',dict(stage='LPIPS',done=st,total=len(rows),time=time.time()))
    pool.shutdown()
    for protocol,arm,sp,d in specs:
        rr=[r for r in rows if (r['protocol'],r['arm'],r['split'])==(protocol,arm,sp)]
        atomic(STORE/(protocol+'_'+arm+'_'+sp+'.json'),dict(rows=rr))
    provenance=dict(status='completed',rows=len(rows),seconds=time.time()-start,device=str(device),
        evaluator='E12-c R1 cosine; fixed 8 Chinese refs for every generation shot',
        e12_weights_sha256=sha(E12/'R1/A_best.pt'),e12_model_lock_sha256=sha(E12/'MODEL_LOCK.json'),
        source_metrics_sha256=sources,script_sha256=sha(__file__),lpips='official lpips 0.1 AlexNet, native96 RGB [-1,1], no resize',
        missing={'identity':'No usable identity checkpoint found on this execution host; no OCR number fabricated.',
                 'human_style_preference':'No blinded human judgments collected.', 'K2':'No completed K2 checkpoint found.', 'K3':'Scored completed outputs.' if args.arm=='K3' else 'Training has not completed.'})
    atomic(STORE/'DONE.json',provenance);print(json.dumps(provenance,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
