"""Lock on internal validation, then report every external five-candidate result."""
import collections,csv,json,math,shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageFilter
from data import STORE,ROOT,Images,atomic,sha
from model import Encoder,SetHead,cosine
from train import encode,auc
EPS=1e-6

def win(a,b):return 1. if a>b+EPS else 0. if b>a+EPS else .5
def tau_b(grades,scores):
    con=dis=tx=ty=0
    for i in range(len(grades)):
        for j in range(i):
            x=np.sign(grades[i]-grades[j]);d=scores[i]-scores[j];y=0 if abs(d)<=EPS else np.sign(d)
            if x and y:
                if x==y:con+=1
                else:dis+=1
            elif not x and y:tx+=1
            elif x and not y:ty+=1
    den=math.sqrt((con+dis+tx)*(con+dis+ty))
    return (con-dis)/den if den else None
def rank_metrics(s):
    valid=all(s.get(c) is not None for c in 'AG NFX'.replace(' ',''))
    result=dict(M5=None,same_char_top2=None,N_not_last=None,tau_GT=None)
    if valid:
        top=min(s['A'],s['G'])>max(s['N'],s['F'],s['X'])+EPS;nlast=s['N']>min(s['F'],s['X'])+EPS
        result.update(M5=float(top and nlast),same_char_top2=float(min(s['A'],s['G'])>max(s['N'],s['F'])+EPS),
            N_not_last=float(nlast),tau_GT=tau_b([3,2,1,3],[s[c] for c in 'ANFX']))
    for a,b in [('A','F'),('X','F'),('A','N'),('X','N'),('N','F')]:
        result[a+'_gt_'+b]=win(s[a],s[b]) if s.get(a) is not None and s.get(b) is not None else None
    others=[v for c,v in s.items() if c!='G' and v is not None]
    result['G_average_rank']=(1+sum(v>s['G']+EPS for v in others)+.5*sum(abs(v-s['G'])<=EPS for v in others)) if valid else None
    return result
def interval(rows,key):
    groups=collections.defaultdict(list)
    for r in rows:
        if r.get(key) is not None:groups[r['font']].append(r[key])
    if not groups:return dict(mean=None,ci95=None,clusters=0,n=0)
    vals=np.array([np.mean(v) for _,v in sorted(groups.items())]);rng=np.random.default_rng(20260917)
    boots=vals[rng.integers(0,len(vals),(10000,len(vals)))].mean(1)
    return dict(mean=float(vals.mean()),ci95=np.quantile(boots,[.025,.975]).tolist(),clusters=len(vals),n=sum(map(len,groups.values())))

def lock_model():
    values={}
    for arm in ['R1','R2','R3']:
        done=json.loads((STORE/arm/'DONE.json').read_text());assert not done['smoke']
        assert done['encoder_sha256']==sha(STORE/arm/'A_best.pt') and done['head_sha256']==sha(STORE/arm/'B_best.pt')
        v=json.loads((STORE/arm/'validation.json').read_text())
        for mode in (['cosine','head'] if arm!='R3' else ['head']):values[arm+'_'+mode]=v[mode+'_AUC']
    order=['R1_cosine','R2_cosine','R3_head','R1_head','R2_head'];best=max(values.values())
    primary=next(k for k in order if values[k]>=best-.005)
    obj=dict(primary=primary,selection='internal validation script-shot macro AUC; within .005 choose fixed simplicity order',values=values,
        candidates_sha256=sha(STORE/'candidates.json'),manifest_sha256=sha(STORE/'manifest.json'),
        model_sha256={arm:{f:sha(STORE/arm/f) for f in ['A_best.pt','B_best.pt']} for arm in ['R1','R2','R3']})
    p=STORE/'MODEL_LOCK.json'
    if p.exists():assert json.loads(p.read_text())==obj
    else:atomic(p,obj)
    return obj

def main():
    torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda:0');lock=lock_model()
    m=json.loads((STORE/'manifest.json').read_text());candidate=json.loads((STORE/'candidates.json').read_text());episodes=candidate['episodes']
    controls=STORE/'controls';controls.mkdir(exist_ok=True);requests=[];positions=[];gt=[];seen=set()
    def request(q,refs):
        if q is None:return None
        idx=len(requests);requests.append((q,tuple(refs)));return idx
    fonts=sorted({e['font'] for e in episodes});refs_by={e['font']:e['refs'] for e in episodes}
    for e in episodes:
        positions.append({c:request(e.get(c),e['refs']) for c in 'AGNFX'})
        key=e['font'],e['cp']
        if key in seen:continue
        seen.add(key);other=fonts[(fonts.index(e['font'])+1)%len(fonts)]
        p=controls/(sha(e['A'])+'_blur3.png')
        if not p.exists():Image.open(e['A']).convert('RGB').filter(ImageFilter.GaussianBlur(3)).save(p)
        cp={}
        for color in ['white','black']:
            path=controls/(color+'.png')
            if not path.exists():Image.new('RGB',(96,96),color).save(path)
            cp[color]=request(str(path),e['refs'])
        gt.append(dict(font=e['font'],cp=e['cp'],script=e['script'],episode=len(positions)-1,
            wrong_ref=request(e['A'],refs_by[other]),blur=request(str(p),e['refs']),**cp))
    paths=sorted({q for q,r in requests}|{x for q,r in requests for x in r});lookup={p:i for i,p in enumerate(paths)}
    registry={p:sha(p) for p in paths};atomic(STORE/'SCORED_FILES.json',registry)
    images=Images([dict(path=p) for p in paths]);qid=torch.tensor([lookup[q] for q,rr in requests]);rid=torch.tensor([[lookup[r] for r in rr] for q,rr in requests])
    results={};internal={};summary={};legacy=ROOT/'runs/E12C-V0916-S3407'
    for arm in ['R1','R2','R3','v1']:
        directory=legacy if arm=='v1' else STORE/arm;multi=arm in ['R2','R3'];dim=768 if multi else 512
        encoder=Encoder(multi,False).to(device);encoder.load_state_dict(torch.load(directory/'A_best.pt',map_location=device,weights_only=False)['model']);encoder.eval()
        head=SetHead(dim,arm=='R3').to(device);head.load_state_dict(torch.load(directory/'B_best.pt',map_location=device,weights_only=False)['head']);head.eval()
        features=encode(encoder,images,list(range(len(images))),device)
        modes=['head'] if arm=='R3' else ['cosine','head']
        if arm!='v1':
            mf=torch.load(directory/'features.pt',map_location='cpu',weights_only=True)['features']
            internal[arm]={mode:auc(head,mf,m['validation']['test'],device,mode) for mode in modes}
        for mode in modes:
            scores=[]
            with torch.no_grad():
                for st in range(0,len(requests),256):
                    q=features[qid[st:st+256]].to(device);rr=features[rid[st:st+256]].to(device);keep=torch.ones(rr.shape[:2],dtype=torch.bool,device=device)
                    y=cosine(q,rr,keep) if mode=='cosine' else head(q,rr,keep)[0];scores+=y.cpu().tolist()
            name=arm+'_'+mode;erows=[];grows=[]
            for e,pos in zip(episodes,positions):
                s={c:scores[i] if i is not None else None for c,i in pos.items()}
                erows.append(dict(id=e['id'],arm=e['arm'],font=e['font'],cp=e['cp'],k=e['k'],script=e['script'],status=e['status'],scores=s,**rank_metrics(s)))
            for c in gt:
                e=erows[c['episode']];s=e['scores'];grows.append(dict(font=c['font'],cp=c['cp'],script=c['script'],
                    **{key:e[key] for key in ['A_gt_F','X_gt_F','A_gt_N','X_gt_N','N_gt_F','tau_GT']},
                    correct_ref=win(s['A'],scores[c['wrong_ref']]),blur_reject=win(s['A'],scores[c['blur']]),
                    white_reject=win(s['A'],scores[c['white']]),black_reject=win(s['A'],scores[c['black']])))
            results[name]=dict(episodes=erows,GT=grows)
            ss=dict(GT={key:interval(grows,key) for key in grows[0] if key not in ['font','cp','script']},generation={})
            for generator in ['K0','K1']:
                rr=[r for r in erows if r['arm']==generator];valid=sum(r['M5'] is not None for r in rr)
                ss['generation'][generator]=dict(total=len(rr),resolved=valid,coverage=valid/len(rr),
                    conservative_M5=sum(r['M5'] or 0 for r in rr)/len(rr),
                    metrics={key:interval(rr,key) for key in ['M5','same_char_top2','N_not_last','G_average_rank']},
                    by_script={g:{key:interval([r for r in rr if r['script']==g],key) for key in ['M5','same_char_top2','N_not_last']} for g in ['western','kana','bopomofo']},
                    by_shot={str(k):{key:interval([r for r in rr if r['k']==k],key) for key in ['M5','same_char_top2','N_not_last']} for k in [1,2,4,8]})
            ss['GT_by_script']={g:{key:interval([r for r in grows if r['script']==g],key) for key in ss['GT']} for g in ['western','kana','bopomofo']}
            summary[name]=ss
            atomic(STORE/(name+'_scores.json'),results[name]);print('SCORED',name,flush=True)
    primary=lock['primary'];base={r['font']+r['cp']:r for r in results['R1_cosine']['GT']};delta=[]
    for r in results[primary]['GT']:
        b=base[r['font']+r['cp']];delta.append(dict(font=r['font'],**{k:r[k]-b[k] if r[k] is not None and b[k] is not None else None for k in r if k not in ['font','cp','script']}))
    comparison={k:interval(delta,k) for k in delta[0] if k!='font'}
    stats=summary[primary];targets={'A_gt_F':.9,'X_gt_F':.9,'correct_ref':.9,'blur_reject':.95,'white_reject':.95,'black_reject':.95,'N_gt_F':.8}
    met={k:stats['GT'][k]['mean'] is not None and stats['GT'][k]['mean']>=v for k,v in targets.items()}
    met['near_far_coverage']=stats['generation']['K1']['coverage']>=.7
    atomic(STORE/'internal_test.json',internal)
    atomic(STORE/'summary.json',dict(primary=primary,models=summary,primary_minus_R1_cosine=comparison,engineering_targets_met=met,
        verdict='MEETS_PROPOSED_TARGETS' if all(met.values()) else 'REVIEW_REQUIRED',human_correlation_validated=False))
    build_board(episodes,results,primary)
    atomic(STORE/'EVALUATION_DONE.json',dict(status='completed',episodes=len(episodes),models=list(results),model_lock_sha256=sha(STORE/'MODEL_LOCK.json'),
        candidate_sha256=sha(STORE/'candidates.json'),source_images_sha256=sha(STORE/'SCORED_FILES.json')))

def build_board(episodes,results,primary):
    review=STORE/'review';assets=review/'assets';assets.mkdir(parents=True,exist_ok=True)
    def asset(p):
        if p is None:return None
        name=sha(p)+'.png';dest=assets/name
        if not dest.exists():shutil.copyfile(p,dest)
        return 'assets/'+name
    rows=[]
    for i,e in enumerate(episodes):
        rows.append(dict(id=e['id'],arm=e['arm'],font=e['font'],cp=e['cp'],k=e['k'],script=e['script'],status=e['status'],
            images={c:asset(e.get(c)) for c in 'AGNFX'},refs=[asset(p) for p in e['refs']],
            scores={name:obj['episodes'][i]['scores'] for name,obj in results.items()}))
    payload=json.dumps(rows,ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><meta charset="utf-8"><title>E12-c 五候选排序</title><style>body{font:14px system-ui;margin:24px}img{width:96px;height:96px}.refs img{width:38px;height:38px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}select,button{padding:7px;margin:4px}small{display:block}</style><h1>E12-c 五候选排序</h1><p>A目标GT · G生成图 · N相似GT · F不相似GT · X同风格不同字。GT相似关系为独立代理；X是风格正例。模型选择已在内部验证集锁定。</p><div id="filters"></div><p id="count"></p><button id="prev">上一页</button><button id="next">下一页</button><table><tbody id="rows"></tbody></table><script>const rows=PAYLOAD,primary=PRIMARY;let state={model:primary,arm:'K1',font:'',k:'',script:'',page:0};for(const field of ['model','arm','font','k','script']){let s=document.createElement('select');if(field!=='model')s.add(new Option('全部 '+field,''));let vals=field==='model'?Object.keys(rows[0].scores):[...new Set(rows.map(r=>r[field]))].sort();for(const v of vals)s.add(new Option(v,v));s.value=state[field];s.onchange=()=>{state[field]=s.value;state.page=0;render()};document.getElementById('filters').appendChild(s)}const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));function render(){let rr=rows.filter(r=>['arm','font','k','script'].every(f=>!state[f]||String(r[f])===state[f]));const pages=Math.max(1,Math.ceil(rr.length/30));state.page=Math.min(state.page,pages-1);document.getElementById('count').textContent=rr.length+' 组，第 '+(state.page+1)+'/'+pages+' 页';document.getElementById('rows').innerHTML=rr.slice(state.page*30,(state.page+1)*30).map(r=>{let scores=r.scores[state.model];let order=Object.keys(scores).filter(c=>scores[c]!==null).sort((a,b)=>scores[b]-scores[a]);return '<tr><td>'+esc(r.id)+'<small>'+r.status+'</small><div class="refs">'+r.refs.map(p=>'<img loading="lazy" src="'+p+'">').join('')+'</div></td>'+order.map(c=>'<td>'+c+'<br><img loading="lazy" src="'+r.images[c]+'"><small>'+scores[c].toFixed(5)+'</small></td>').join('')+'</tr>'}).join('')}document.getElementById('prev').onclick=()=>{state.page=Math.max(0,state.page-1);render()};document.getElementById('next').onclick=()=>{state.page++;render()};render();</script>'''
    (review/'index.html').write_text(page.replace('PAYLOAD',payload).replace('PRIMARY',json.dumps(primary)))
    shutil.copyfile(STORE/'summary.json',review/'summary.json')
    with (review/'summary.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['model','generator','font','cp','shot','M5','same_char_top2','N_not_last','G_rank'])
        for name,obj in results.items():
            for r in obj['episodes']:w.writerow([name,r['arm'],r['font'],r['cp'],r['k'],r['M5'],r['same_char_top2'],r['N_not_last'],r['G_average_rank']])
    link=ROOT/'reports/e12c_r2_review_20260917'
    if not link.exists():link.symlink_to(review,target_is_directory=True)

if __name__=='__main__':main()
