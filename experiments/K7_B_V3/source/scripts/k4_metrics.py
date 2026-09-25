"""Frozen R1/LPIPS scoring; stratified v2 reports, paired cluster intervals."""
import argparse,collections,csv,hashlib,json,os,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from k4_runtime import STORE,ASSETS,DATA,ROOT,read_unit,atomic_json,sha256_file,RUNS
from scripts.e12c_r2.model import Encoder

ARMS=['K0','K1','K3','K4-C','K4-B','K4-A']
E12=Path('/root/data1/hrfont_e12c_r2_20260917')
METRICS=['l1','ssim','lpips_alex','D_region','D_change','D_high','family_s01','gt_family_s01','content_family_s01']

def score(arm):
    import lpips
    rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    torch.cuda.set_device(rank);torch.set_num_threads(2);device=torch.device('cuda',rank)
    if world>1:dist.init_process_group('nccl')
    out=STORE/'metrics'/arm;out.mkdir(parents=True,exist_ok=True)
    lock=json.loads((E12/'MODEL_LOCK.json').read_text());assert lock['primary']=='R1_cosine'
    e=Encoder(False,False).to(device);e.load_state_dict(torch.load(E12/'R1/A_best.pt',map_location=device,weights_only=False)['model'])
    e.eval().requires_grad_(False);lp=lpips.LPIPS(net='alex',version='0.1').to(device).eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    pool=ThreadPoolExecutor(8);prototype={};sources={};rows=[]
    for sp in ['val','test']:
        path=STORE/'inference'/arm/sp/'metrics.json';sources[str(path)]=sha256_file(path)
        rows+=json.loads(path.read_text())['rows']
    def state_hash(net):
        h=hashlib.sha256()
        for k,v in sorted(net.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    identity=dict(arm=arm,sources=sources,e12_sha256=sha256_file(E12/'R1/A_best.pt'),lock_sha256=sha256_file(E12/'MODEL_LOCK.json'),lpips_sha256=state_hash(lp),refs='fixed valid per-font ref8, independent of generation shot',script_sha256=sha256_file(Path(__file__)))
    if rank==0:
        p=out/'PROTOCOL.json'
        if p.exists():assert json.loads(p.read_text())==identity
        else:atomic_json(p,identity)
    if world>1:dist.barrier()
    progress=out/f'rank{rank}.jsonl';existing={}
    if progress.exists():
        for l in progress.read_text().splitlines():
            try:r=json.loads(l)
            except json.JSONDecodeError:continue
            existing[r['prediction']]=r
    shard=rows[rank::world];done=[];start=time.time()
    with torch.inference_mode():
        for st in range(0,len(shard),32):
            chunk=shard[st:st+32];pending=[]
            for r in chunk:
                old=existing.get(r['prediction'])
                if old is not None and old['prediction_sha256']==r['prediction_sha256']:done.append(old)
                else:pending.append(r)
            if not pending:continue
            for r in pending:
                key=(r['split'],r['font'])
                if key in prototype:continue
                paths=[DATA/'v2'/r['split']/'StyleImage'/r['font']/f"{r['font']}+{c}.png" for c in r['ref8']]
                x=torch.stack(list(pool.map(read_unit,paths))).to(device)
                with torch.autocast('cuda',dtype=torch.float16):z=e((x-mean)/std)
                prototype[key]=F.normalize(z.float().mean(0),dim=0)
            a=torch.stack(list(pool.map(read_unit,[r['prediction'] for r in pending]))).to(device)
            b=torch.stack(list(pool.map(read_unit,[r['target'] for r in pending]))).to(device)
            c=torch.stack(list(pool.map(read_unit,[r['content'] for r in pending]))).to(device)
            with torch.autocast('cuda',dtype=torch.float16):z=e((torch.cat([a,b,c])-mean)/std).float()
            ref=torch.stack([prototype[(r['split'],r['font'])] for r in pending]);n=len(pending)
            cosine=[(x*ref).sum(1).cpu().tolist() for x in z.split(n)]
            distances=lp(a*2-1,b*2-1).flatten().cpu().tolist()
            scored=[]
            for i,r in enumerate(pending):
                assert sha256_file(Path(r['prediction']))==r['prediction_sha256']
                row=dict(r,arm=arm,lpips_alex=float(distances[i]))
                for k,v in zip(['family','gt_family','content_family'],cosine):row[k+'_cosine']=float(v[i]);row[k+'_s01']=(1+float(v[i]))/2
                scored.append(row)
            with progress.open('a') as f:f.write(''.join(json.dumps(r)+'\n' for r in scored))
            done+=scored
            atomic_json(out/f'progress_rank{rank}.json',dict(done=len(done),total=len(shard),seconds=time.time()-start))
    atomic_json(out/f'rank{rank}.json',dict(rows=done));pool.shutdown()
    if world>1:dist.barrier()
    if rank==0:
        rr=sum([json.loads((out/f'rank{i}.json').read_text())['rows'] for i in range(world)],[])
        assert len(rr)==len(rows)==len({r['prediction'] for r in rr})
        atomic_json(out/'metrics.json',dict(rows=rr));atomic_json(out/'DONE.json',dict(status='completed',rows=len(rr),identity=identity))
    if world>1:dist.destroy_process_group()

def write_csv(path,rows):
    assert rows
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def report():
    out=STORE/'review';out.mkdir(parents=True,exist_ok=True)
    with (STORE/'difficulty/difficulty_font_script.csv').open() as f:diff={(r['split'],r['font'],r['script']):r['difficulty'] for r in csv.DictReader(f)}
    aliases=json.loads((ASSETS/'aliases.json').read_text());overlap=set(json.loads((ASSETS/'lineage_overlap.json').read_text())['fonts'])
    def subtype(cp):
        i=int(cp[1:],16)
        return 'digits' if 48<=i<=57 else 'uppercase' if 65<=i<=90 else 'lowercase' if 97<=i<=122 else 'latin_ext' if i<0x3000 else 'hiragana' if i<0x30A0 else 'katakana' if i<0x3100 else 'bopomofo'
    rows=[];sources={}
    for arm in ARMS:
        p=STORE/'metrics'/arm/'metrics.json';sources[str(p)]=sha256_file(p)
        for r in json.loads(p.read_text())['rows']:
            r['difficulty']=diff[r['split'],r['font'],r['script']];r['subscript']=subtype(r['cp'])
            r['known_lineage_overlap']=r['font'] in overlap;rows.append(r)
    fields=['arm','split','source','font','cp','k','script','subscript','difficulty','known_lineage_overlap',*METRICS,'prediction','prediction_sha256','reused']
    write_csv(out/'per_image.csv',[{k:r.get(k) for k in fields} for r in rows])
    aggregates=collections.defaultdict(list)
    for r in rows:
        for source in [r['source'],'all']:
            for script in [r['script'],'all']:
                for difficulty in [r['difficulty'],'all']:
                    aggregates[r['arm'],r['split'],source,r['k'],script,difficulty,'all',False].append(r)
        aggregates[r['arm'],r['split'],r['source'],r['k'],r['script'],r['difficulty'],r['subscript'],False].append(r)
        if not r['known_lineage_overlap']:
            aggregates[r['arm'],r['split'],r['source'],r['k'],r['script'],'all','all',True].append(r)
    summary=[];fontmeans={}
    for key,rr in sorted(aggregates.items()):
        byfont=collections.defaultdict(list)
        for r in rr:byfont[r['font']].append([r[m] for m in METRICS])
        fm={f:np.mean(v,axis=0) for f,v in byfont.items()};fontmeans[key]=fm
        macro=np.mean(list(fm.values()),axis=0);micro=np.mean([[r[m] for m in METRICS] for r in rr],axis=0)
        item=dict(zip(['arm','split','source','shot','script','difficulty','subscript','exclude_known_lineage_overlap'],key))
        summary.append(dict(**item,fonts=len(fm),pairs=len(rr),**{m+'_font_macro':float(v) for m,v in zip(METRICS,macro)},**{m+'_pair_micro':float(v) for m,v in zip(METRICS,micro)}))
    write_csv(out/'stratified_metrics.csv',summary)
    balanced=[]
    for arm in ARMS:
        for sp in ['val','test']:
            for k in ([1] if arm=='K0' else [1,2,4,8]):
                cells=[r for r in summary if r['arm']==arm and r['split']==sp and r['shot']==k and r['source']!='all' and r['script']!='all' and r['difficulty']=='all' and r['subscript']=='all' and not r['exclude_known_lineage_overlap']]
                balanced.append(dict(arm=arm,split=sp,shot=k,cells=len(cells),**{m:float(np.mean([r[m+'_font_macro'] for r in cells])) for m in METRICS}))
    write_csv(out/'source_script_balanced.csv',balanced)
    comparisons=[];rng=np.random.default_rng(3407)
    for candidate,baseline in [('K4-B','K4-C'),('K4-C','K1'),('K4-B','K3'),('K4-A','K1'),('K1','K0'),('K3','K0'),('K4-A','K0'),('K4-B','K0'),('K4-C','K0')]:
        for key,cm in fontmeans.items():
            arm,sp,src,k,sc,diffname,sub,exclude=key
            if arm!=candidate or sub!='all' or exclude or (baseline=='K0' and k!=1):continue
            bm=fontmeans.get((baseline,*key[1:]));assert bm and set(cm)==set(bm)
            fonts=sorted(cm);delta=np.array([cm[f]-bm[f] for f in fonts]);groups=collections.defaultdict(list)
            for i,f in enumerate(fonts):groups[aliases.get(f,f)].append(i)
            clusters=list(groups.values());boot=[]
            for _ in range(1000):
                ii=np.concatenate([clusters[i] for i in rng.integers(len(clusters),size=len(clusters))]);boot.append(delta[ii].mean(0))
            lo,hi=np.quantile(np.array(boot),[.025,.975],axis=0)
            for n,m in enumerate(MR for MR in METRICS if MR not in ['gt_family_s01','content_family_s01']):
                idx=METRICS.index(m)
                comparisons.append(dict(candidate=candidate,baseline=baseline,split=sp,source=src,shot=k,script=sc,difficulty=diffname,metric=m,delta=float(delta[:,idx].mean()),ci_low=float(lo[idx]),ci_high=float(hi[idx]),fonts=len(fonts),lineage_clusters=len(clusters)))
    write_csv(out/'paired_deltas_cluster_bootstrap.csv',comparisons)
    # Compact lazy-loading gallery: one query per row; K0 is always labelled 1-shot.
    queries=[json.loads(l) for l in (ASSETS/'v2_val_test_queries.jsonl').read_text().splitlines()]
    for name,path in [('inference',STORE/'inference'),('data',DATA/'v2')]:
        p=out/name
        if not p.exists():p.symlink_to(path,target_is_directory=True)
    html='''<!doctype html><meta charset="utf-8"><title>K family · v2 val/test</title>
<style>body{font:15px system-ui;margin:24px;background:#f3f5f8}select{margin:8px;padding:8px}td,th{padding:8px;border:1px solid #ddd}img{width:96px;height:96px}table{border-collapse:collapse;background:white}.refs img{width:48px;height:48px}</style>
<h1>K family · v2 val/test</h1><p>K0: 1-shot only. Other models: 1/2/4/8-shot. Shared v2 train donor bank. E12-c is auxiliary; no human preference claim.</p>
<p><a href="stratified_metrics.csv">Stratified metrics</a> · <a href="paired_deltas_cluster_bootstrap.csv">Paired differences</a> · <a href="per_image.csv">Per-image rows</a></p>
<select id="split"><option>val</option><option>test</option></select><select id="font"></select><select id="shot"><option>1</option><option>2</option><option>4</option><option>8</option></select><div id="board"></div>
<script>const queries=QUERIES;const arms=ARMS;const E=id=>document.getElementById(id);function fonts(){const fs=[...new Set(queries.filter(q=>q.split===E('split').value).map(q=>q.font))];E('font').replaceChildren(...fs.map(f=>new Option(f,f)));render()};function render(){const rr=queries.filter(q=>q.split===E('split').value&&q.font===E('font').value);const k=+E('shot').value;E('board').innerHTML='<table><tr><th>Character</th><th>Refs</th><th>Content</th><th>GT</th>'+arms.map(a=>'<th>'+a+' '+(a==='K0'?1:k)+' shot</th>').join('')+'</tr>'+rr.map(q=>{let base='data/'+q.split;let img=p=>'<img loading="lazy" src="'+p+'">';return '<tr><td>'+q.char+'<br>'+q.source+' / '+q.script+'</td><td class="refs">'+q.ref8.slice(0,k).map(c=>img(base+'/StyleImage/'+q.font+'/'+q.font+'+'+c+'.png')).join('')+'</td><td>'+img(base+'/ContentImage/'+q.cp+'.png')+'</td><td>'+img(base+'/TargetImage/'+q.font+'/'+q.font+'+'+q.cp+'.png')+'</td>'+arms.map(a=>'<td>'+img('inference/'+a+'/'+q.split+'/'+q.font+'__'+q.cp+'__k'+(a==='K0'?1:k)+'.png')+'</td>').join('')+'</tr>'}).join('')+'</table>'};E('split').onchange=fonts;E('font').onchange=render;E('shot').onchange=render;fonts();</script>'''
    (out/'index.html').write_text(html.replace('QUERIES',json.dumps(queries,ensure_ascii=False).replace('</','<\\/')).replace('ARMS',json.dumps(ARMS)))
    atomic_json(out/'DONE.json',dict(status='completed',images=len(rows),sources=sources,ci='1000 paired lineage-cluster bootstrap replicates; single training seed',human_judgments=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS);p.add_argument('--report',action='store_true');a=p.parse_args()
    if a.report:report()
    elif a.arm:score(a.arm)
    else:p.error('Specify --arm or --report')
