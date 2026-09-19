import json,pathlib,collections,math
import numpy as np
from PIL import Image
p=pathlib.Path('outputs/K5_BANK_DIAGNOSTIC_20260919');groups=collections.defaultdict(list);donors=collections.defaultdict(list)
for f in sorted(p.glob('K5-*/case*/*/result.json')):
 r=json.loads(f.read_text());r['local_png']=str(f.parent/'prediction.png');groups[r['arm'],r['font'],r['cp'],r['seed'],r['mode']].append(r)
 for s in r['selection']:
  a=s['alpha'];donors[r['arm'],r['mode']].append(dict(n=len(a),effective=math.exp(-sum(x*math.log(x) for x in a if x>0)),maximum=max(a)))
repeats=[]
for key,rs in groups.items():
 assert len({tuple(r['refs']) for r in rs})==1,key
 if len(rs)==1:continue
 x,y=[np.asarray(Image.open(r['local_png']),dtype=np.float32) for r in rs];d=np.abs(x-y)
 repeats.append(dict(key=key,mean_pixel_levels=float(d.mean()),max_pixel_levels=float(d.max()),same_hash=rs[0]['prediction_sha256']==rs[1]['prediction_sha256']))
out=dict(duplicate_groups=len(repeats),duplicate_protocol_refs_equal=True,repeat_results=repeats,selection=[dict(arm=a,mode=m,n=sorted({r['n'] for r in rs}),mean_effective=sum(r['effective'] for r in rs)/len(rs),mean_max_alpha=sum(r['maximum'] for r in rs)/len(rs)) for (a,m),rs in sorted(donors.items())])
(p/'SELECTION_AUDIT.json').write_text(json.dumps(out,indent=2));print('repeat max',max(x['max_pixel_levels'] for x in repeats),'repeat mean',sum(x['mean_pixel_levels'] for x in repeats)/len(repeats));print(out['selection'])
