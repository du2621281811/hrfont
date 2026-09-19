import pathlib,json,collections
import numpy as np
from PIL import Image
p=pathlib.Path('outputs/K5_BANK_DIAGNOSTIC_20260919');cases=collections.defaultdict(dict)
for f in p.glob('K5-*/case*/*/result.json'):
 d=json.loads(f.read_text());cases[d['arm'],d['font'],d['cp'],d['seed']][d['mode']]=f.parent
rows=[]
for (arm,font,cp,seed),m in sorted(cases.items()):
 b=np.asarray(Image.open(m['top10']/'prediction.png'),dtype=np.float32)
 for mode,path in m.items():
  if mode=='top10':continue
  diff=np.abs(np.asarray(Image.open(path/'prediction.png'),dtype=np.float32)-b)
  rows.append(dict(arm=arm,font=font,cp=cp,seed=seed,mode=mode,mean_pixel_levels=float(diff.mean()),max_pixel_levels=float(diff.max()),changed_fraction=float((diff>0).mean())))
g=collections.defaultdict(list)
for r in rows:g[r['arm'],r['mode']].append(r)
s=[]
for (a,m),rr in sorted(g.items()):s.append(dict(arm=a,mode=m,unique_cases=len(rr),mean_pixel_levels=float(np.mean([x['mean_pixel_levels'] for x in rr])),max_pixel_levels=max(x['max_pixel_levels'] for x in rr),exact_equal=sum(x['max_pixel_levels']==0 for x in rr)))
(p/'PIXEL_SENSITIVITY.json').write_text(json.dumps(dict(deduplicated=True,unit='uint8 pixel level, 0..255',summary=s,rows=rows),indent=2));print(json.dumps(s,indent=2))
