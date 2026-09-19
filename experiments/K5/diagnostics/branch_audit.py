import pathlib,json,collections,numpy as np
from PIL import Image
p=pathlib.Path('outputs/K5_BRANCH_ABLATION_20260920'); old=pathlib.Path('outputs/K5_BANK_DIAGNOSTIC_20260919');groups=collections.defaultdict(list);seen=set();par=[]
def im(f):return np.asarray(Image.open(f)).astype(float)
for f in sorted(p.glob('K5-B/case*/baseline/result.json')):
 r=json.loads(f.read_text()); b=im(f.parent/'prediction.png');o=old/'K5-B'/f.parent.parent.name/'top10'
 prev=json.loads((o/'result.json').read_text())
 assert all(r[k]==prev[k] for k in ['font','cp','refs','seed','weights_sha256'])
 assert r['selection'][0]['donors']==prev['selection'][0]['donors']
 d=np.abs(b-im(o/'prediction.png'));par.append(dict(case=r['case'],seed=r['seed'],max=float(d.max()),mean=float(d.mean())))
 key=(r['font'],r['cp'],r['seed'])
 if key in seen:continue
 seen.add(key)
 for mode in ['ec_only','es_only','zero_offset']:
  x=json.loads((f.parent.parent/mode/'result.json').read_text());d=np.abs(b-im(f.parent.parent/mode/'prediction.png'))
  groups[r['font'],mode].append(dict(pixel_mean=float(d.mean()),pixel_max=float(d.max()),delta_l1=x['l1']-r['l1'],delta_ssim=x['ssim']-r['ssim']))
out=[dict(font=k[0],mode=k[1],n=len(v),**{m:float(np.mean([r[m] for r in v])) for m in v[0]},max_pixel_any=max(r['pixel_max'] for r in v)) for k,v in groups.items()]
res=dict(baseline_parity=par,unique_cases=len(seen),dedup_rule='sorted first case per font/character/seed; ordinary duplicate omitted',summary=out)
(p/'PAIRED_AUDIT.json').write_text(json.dumps(res,indent=2));print('baseline max',max(x['max'] for x in par));print(json.dumps(out,indent=2))
