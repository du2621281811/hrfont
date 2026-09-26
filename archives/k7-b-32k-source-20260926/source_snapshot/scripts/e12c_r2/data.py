"""Alias-grouped E12-c R2 data, excluding all main evaluation font groups."""
import collections,csv,hashlib,json,random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

ROOT=Path('/root/projects/hrfont')
STORE=Path('/root/data1/hrfont_e12c_r2_20260917')
DATA=ROOT/'data/fontdiffuser-p253-t295-s338-cn2west-v2'
CLEAN=ROOT/'manifests/v0913_clean'
SCRIPTS=('latin','kana','bopomofo')
REF8=[f'u{ord(c):04X}' for c in '永和书风骨韵天地']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

class Images(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.rows[i]['path']) as im:
            assert im.size==(96,96)
            x=torch.from_numpy(np.array(im.convert('RGB'),copy=True)).permute(2,0,1).float()/255
        return (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]

def collect():
    records=[];split_fonts={}
    for sp in ('train','val','test'):
        rows=list(csv.DictReader((CLEAN/f'pairs_{sp}.tsv').open(),delimiter='\t'))
        split_fonts[sp]=sorted({r['font'] for r in rows})
        records.extend(dict(font=r['font'],cp=r['cp'],script=r['script_group'],role='target',main_split=sp,
            path=str(DATA/sp/'TargetImage'/r['font']/f"{r['font']}+{r['cp']}.png")) for r in rows)
        for f in split_fonts[sp]:
            for p in sorted((DATA/sp/'StyleImage'/f).glob('*.png')):
                records.append(dict(font=f,cp=p.stem[len(f)+1:],script='han',role='style',main_split=sp,path=str(p)))
    def verify(r):
        p=Path(r['path'])
        with Image.open(p) as im:
            assert im.size==(96,96)
            pixels=im.convert('RGB').tobytes()
        return dict(r,sha256=sha(p),rgb_sha256=hashlib.sha256(pixels).hexdigest())
    with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(verify,records))
    return records,split_fonts

def grouped(records,family):
    fonts=sorted({r['font'] for r in records});parent={f:f for f in fonts}
    def find(f):
        while parent[f]!=f:parent[f]=parent[parent[f]];f=parent[f]
        return f
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    by=collections.defaultdict(list)
    for r in records:by[r['font'],r['role'],r['script']].append((r['cp'],r['rgb_sha256']))
    signature={};edges=[]
    for (f,role,script),pairs in sorted(by.items()):
        if len(pairs)<8:continue
        sig=(role,script,tuple(sorted(pairs)))
        if sig in signature:
            other=signature[sig];union(f,other);edges.append(dict(a=f,b=other,reason='exact_'+role+'_'+script,count=len(pairs)))
        else:signature[sig]=f
    families=collections.defaultdict(list)
    for f,ff in family.items():
        if f in parent:families[ff].append(f)
    for members in families.values():
        for f in members[1:]:union(f,members[0])
    return {f:find(f) for f in fonts},edges

def build():
    STORE.mkdir(parents=True,exist_ok=True)
    if (STORE/'manifest.json').exists():
        print('REUSE_MANIFEST',sha(STORE/'manifest.json'),flush=True);return
    allrows,mainfonts=collect()
    old=json.loads((ROOT/'artifacts/i34_20260915/e12c_manifest.json').read_text())
    known=old.get('known_families',{});unit,edges=grouped(allrows,known)
    deny={unit[f] for sp in ('val','test') for f in mainfonts[sp]}
    eligible=[f for f in mainfonts['train'] if unit[f] not in deny]
    keys=sorted({unit[f] for f in eligible});random.Random(3407).shuffle(keys)
    assert len(keys)>=40
    sp_by={u:'train' if i<int(.8*len(keys)) else 'val' if i<int(.9*len(keys)) else 'test' for i,u in enumerate(keys)}
    split={f:sp_by[unit[f]] for f in eligible}
    rows=[r for r in allrows if r['font'] in split]
    by={f:{g:[] for g in ('han',)+SCRIPTS} for f in split}
    lookup={}
    for i,r in enumerate(rows):by[r['font']][r['script']].append(i);lookup[r['font'],r['role'],r['cp']]=i
    validation={}
    for sp in ('val','test'):
        pool=sorted(f for f in split if split[f]==sp);gallery={f:[lookup[f,'style',c] for c in REF8] for f in pool}
        queries=[];episodes=[]
        for f in pool:
            for g in SCRIPTS:
                candidates=by[f][g];chosen=random.Random('3407:'+f+g).sample(candidates,min(4,len(candidates)))
                for idx in chosen:
                    cp=rows[idx]['cp'];legal=[other for other in pool if unit[other]!=unit[f] and (other,'target',cp) in lookup and
                        rows[lookup[other,'target',cp]]['rgb_sha256']!=rows[idx]['rgb_sha256']]
                    if not legal:continue
                    other=random.Random('3407:'+f+cp).choice(legal)
                    queries.append(dict(index=idx,font=f,script=g))
                    for k in (1,2,4,8):episodes.append(dict(pos=idx,neg=lookup[other,'target',cp],refs=gallery[f][:k],font=f,
                        negative_font=other,cp=cp,script=g,k=k))
        assert {(e['script'],e['k']) for e in episodes}=={(g,k) for g in SCRIPTS for k in (1,2,4,8)}
        validation[sp]=dict(gallery=gallery,queries=queries,episodes=episodes)
    for sp in ('train','val','test'):
        assert {g for f in split if split[f]==sp for g in SCRIPTS if len(by[f][g])>=2}==set(SCRIPTS)
    assert not {unit[f] for f in split}&deny
    atomic(STORE/'inventory_all.json',dict(rows=allrows,main_fonts=mainfonts,groups=unit,alias_edges=edges,known_families=known))
    m=dict(version='E12C-R2-ALIAS-GROUP-v1',split=split,rows=rows,byfont=by,groups={f:unit[f] for f in split},
        known_families=known,unknown_family_fonts=[f for f in split if f not in known],validation=validation,
        excluded_main_eval_fonts=mainfonts['val']+mainfonts['test'],excluded_train_aliases=sorted(set(mainfonts['train'])-set(eligible)),
        alias_edges=edges,inventory_sha256=sha(STORE/'inventory_all.json'),seed=3407,
        sources_sha256={f.name:sha(f) for f in CLEAN.glob('*') if f.is_file()},
        split_counts=dict(collections.Counter(split.values())),group_counts=dict(collections.Counter(sp_by.values())))
    atomic(STORE/'manifest.json',m)
    print(json.dumps({k:m[k] for k in ['split_counts','group_counts','excluded_train_aliases']})+' rows='+str(len(rows)),flush=True)

if __name__=='__main__':build()
