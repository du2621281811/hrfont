"""Explicit v2 manifests, partial-reference donors, immutable legacy encoders."""
import csv, functools, hashlib, json, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image
from k_runtime import (ROOT,CODE,CACHE,PARENT0,T,model_for as legacy_model_for,
                       batch_to,seed_episode,atomic_json,sha256_file,train_sample)
from scripts.hrfont_feature_cache import EsCache,EcCache,key_es,key_ec,ES_SPATIAL,EC_SHAPES
from scripts.k_components import KSampler
from scripts.k4_family import FamilyPolicy

DATA=Path('/root/data1/hrfont_dataset_v2_20260917')
STORE=Path('/root/data1/hrfont_k4_weight_20260918')
ASSETS=CODE/'experiments/K4'
RUNS={'K4-C':'K4-C-K1RECIPE-V2-WEIGHT-S3407','K4-B':'K4-B-K3RECIPE-V2-WEIGHT-S3407','K4-A':'K4-A-K1FT-0917-WEIGHT-S3407'}
K1_EMA=ROOT/'runs/K1-ORIGINAL-V0917-S3407/global_step_10000/ema.pth'

def tsv(path):
    with Path(path).open() as f:return list(csv.DictReader(f,delimiter='\t'))

def configure(args, data_name='v2', donor_name='v2'):
    assert data_name in ['v2','v0917'] and donor_name in ['v2','v0917','legacy']
    args.data_root=str(DATA/data_name)
    args.split_manifest=str(DATA/'manifests'/data_name/'split.json')
    args.v0913_clean_map=str(DATA/'manifests'/data_name)
    args.k4_donor_name=donor_name
    return args

def model_for(arm,device,output,evaluation=False):
    authorization=json.loads((ASSETS/'AUTHORIZATION.json').read_text())
    assert sha256_file(PARENT0/'unet.pth')==authorization['k0_parent_unet_sha256']
    model,args=legacy_model_for('K0' if arm=='K0' else 'K1',device,output)
    if arm=='K4-A' and not evaluation:
        expected=json.loads((ASSETS/'AUTHORIZATION.json').read_text())['k1_parent_ema_sha256']
        assert sha256_file(K1_EMA)==expected
        model.load_train_state(torch.load(K1_EMA,map_location=device,weights_only=True))
    model.arm=arm
    name='v0917' if arm=='K4-A' and not evaluation else 'v2'
    return model,configure(args,name,name)

def read_unit(path):
    with Image.open(path) as im:
        assert im.size==(96,96) and im.mode=='RGB',str(path)
        return torch.from_numpy(np.array(im,copy=True)).permute(2,0,1).float()/255.

def ref8(pool):
    order=[f'u{ord(c):04X}' for c in '永和书风骨韵天地山水人']
    refs=[c for c in order if c in pool][:8]
    assert len(refs)==8
    return refs

class Dataset:
    def __init__(self,args,split):
        self.root=Path(args.data_root);self.phase=split;self.rows=tsv(Path(args.v0913_clean_map)/f'pairs_{split}.tsv')
        stems=json.loads(Path(args.split_manifest).read_text())['stems'][split]
        pools=json.loads((Path(args.v0913_clean_map)/'style_pool.json').read_text())
        self.style_by_font_char={f:{c:self.root/split/'StyleImage'/f/f'{f}+{c}.png' for c in pools[f]} for f in stems}
        self.target_images=[str(self.root/split/'TargetImage'/r['font']/f"{r['font']}+{r['cp']}.png") for r in self.rows]
        self.lookup={(r['font'],r['cp']):i for i,r in enumerate(self.rows)}
        assert len(self.lookup)==len(self.rows) and all(r['split']==split and r['font'] in stems for r in self.rows)
    def __len__(self):return len(self.rows)
    def sample(self,i,refs=None):
        r=self.rows[i];f,c=r['font'],r['cp'];pool=self.style_by_font_char[f]
        if refs is None:refs=random.sample(sorted(pool),random.randint(1,8)) if self.phase=='train' else ref8(pool)
        assert c not in refs and len(set(refs))==len(refs) and all(x in pool for x in refs)
        target=read_unit(self.target_images[i]);content=read_unit(self.root/self.phase/'ContentImage'/f'{c}.png')
        return dict(content_image=(content-.5)/.5,target_image=(target-.5)/.5,
                    target_image_path=self.target_images[i],nonorm_target_image=target,
                    font_stem=f,char_cp=c,split=self.phase,ref_chars=refs,
                    ref_image_paths=[str(pool[x]) for x in refs])
    def __getitem__(self,i):return self.sample(i)

def dataset(args,split):return Dataset(args,split)

class Extension:
    def __init__(self,kind):
        self.root=STORE/'cache'/kind
        self.complete=json.loads((self.root/'COMPLETE.json').read_text())
        assert self.complete['status']=='COMPLETE'
        self.identity=self.complete['identity'];self.keys=(self.root/'keys.txt').read_text().splitlines()
        assert sha256_file(self.root/'keys.txt')==self.identity['keys_sha256']
        weight=PARENT0/('style_encoder.pth' if kind=='es' else 'content_encoder.pth')
        assert sha256_file(weight)==self.identity['encoder_sha256']
        for n,h in self.identity['source_sha256'].items():assert sha256_file(DATA/'manifests/v0917'/n)==h
        self.index={k:i for i,k in enumerate(self.keys)};assert len(self.index)==self.identity['entries']
        specs={'spatial':ES_SPATIAL,'pooled':(1024,)} if kind=='es' else {f's{i}':EC_SHAPES[i] for i in [1,2]}
        self.arrays={}
        for name,shape in specs.items():
            p=self.root/(name+'.dat');assert p.stat().st_size==len(self.keys)*int(np.prod(shape))*2
            self.arrays[name]=np.memmap(p,mode='r',dtype=np.float16,shape=(len(self.keys),*shape))

class MergedEs:
    def __init__(self):
        self.old=EsCache(ROOT/'artifacts/g0/es_spatial');self.new=Extension('es')
    def tensor(self,name,split,font,cp):
        key=key_es(split,font,cp)
        if key in self.new.index:array=self.new.arrays[name];i=self.new.index[key]
        else:array=getattr(self.old,name);i=self.old.table.index[key]
        return torch.from_numpy(np.array(array[i],copy=True)).float()
    def spatial_tensor(self,*key):return self.tensor('spatial',*key)
    def pooled_tensor(self,*key):return self.tensor('pooled',*key)

class MergedEc:
    def __init__(self):
        self.old=EcCache(ROOT/'artifacts/g0/ec_multiscale');self.new=Extension('ec')
        self.neutral={}
    def features_many(self,role,items):
        assert role in ['content','target']
        unique=list(dict.fromkeys(items));result={}
        if role=='content':
            for item in unique:
                if item not in self.neutral:self.neutral[item]=self.old.features(role,*item)
                result[item]=self.neutral[item]
            return result
        # The K architecture only consumes donor levels 1 and 2. Avoid reading
        # unused multiscale rows from NFS; consumed values remain byte-identical.
        for newer in [False,True]:
            selected=[item for item in unique if (key_ec(role,*item) in self.new.index)==newer]
            if not selected:continue
            index=self.new.index if newer else self.old.table.index
            ids=[index[key_ec(role,*item)] for item in selected]
            batches={s:torch.from_numpy(np.array((self.new.arrays[f's{s}'] if newer else self.old.scales[s])[ids],copy=True)).float() for s in [1,2]}
            for i,item in enumerate(selected):result[item]=[None,batches[1][i:i+1],batches[2][i:i+1],None,None]
        return result

class Library:
    def __init__(self,es,donor_map,pools):
        self.fonts=sorted({f for fs in donor_map.values() for f in fs})
        self.font_index={f:i for i,f in enumerate(self.fonts)}
        chars=sorted({c for f in self.fonts for c in pools[f]});self.char_index={c:i for i,c in enumerate(chars)}
        self.table=torch.zeros(len(self.fonts),len(chars),1024)
        self.present=torch.zeros(len(self.fonts),len(chars),dtype=torch.bool)
        for i,f in enumerate(self.fonts):
            cols=[self.char_index[c] for c in sorted(pools[f])]
            self.table[i,cols]=torch.stack([es.pooled_tensor('train',f,c) for c in sorted(pools[f])])
            self.present[i,cols]=True
        self.allowed={c:set(fs) for c,fs in donor_map.items()}
        self.bycp={c:torch.tensor([f in fs for f in self.fonts]) for c,fs in self.allowed.items()}
        self.family_policy=FamilyPolicy(ASSETS/'weight_groups.json',self.fonts)
    def select(self,cp,refs,query,font,alpha_cfg):
        cols=[self.char_index[c] for c in refs]
        valid=self.bycp[cp]&self.present[:,cols].all(1)
        valid=self.family_policy.exclude(font,valid)
        # Invalid prototypes are zero only as an intermediate representation;
        # they are explicitly excluded before alpha/top-k, never donor targets.
        excluded=(~valid).nonzero().flatten().tolist()
        idx,alpha,_=T.compute_alpha(query.cpu(),self.table[:,cols],self.font_index.get(font),alpha_cfg,extra_exclude=excluded)
        assert idx and all(valid[i] for i in idx)
        selected=[self.fonts[i] for i in idx]
        self.family_policy.verify(font,selected)
        assert bool(torch.isfinite(alpha).all()) and abs(float(alpha.sum())-1)<1e-5
        return selected,alpha

class DataContext:
    def __init__(self,args,device):
        self.args,self.device=args,device;self.es=MergedEs();self.ec=MergedEc()
        if args.k4_donor_name=='legacy':
            rows=tsv(ROOT/'manifests/v0913_clean/pairs_train.tsv');dm={}
            for r in rows:dm.setdefault(r['cp'],[]).append(r['font'])
        else:dm=json.loads((DATA/'manifests'/args.k4_donor_name/'donor_train_by_cp.json').read_text())
        pools=json.loads((DATA/'manifests/v2/style_pool.json').read_text())
        self.library=Library(self.es,dm,pools);self.fonts=self.library.fonts
        assert set(pools)==set(self.library.family_policy.groups)
        assert set(json.loads((ASSETS/'aliases.json').read_text()))==set(pools)  # separate auxiliary lineage
        self.alpha_cfg=T.DeltaConfig(tau=args.delta_tau,eps_alpha=args.delta_eps_alpha,k_max=args.delta_k_max,k_top=args.delta_k_top,mode=args.delta_mode,rng_seed=args.seed)
        self.pool=ThreadPoolExecutor(max_workers=4)
    @staticmethod
    def image(path):return (read_unit(path)-.5)/.5
    @torch.no_grad()
    def conditions(self,samples,cfg,source,no_delta=False,official=False,need_refs=True):
        assert not official
        style,queries,*_=T._style_conditions(self.es,samples,self.device)
        lengths=[len(r) for r in samples['ref_chars']]
        keep=torch.arange(max(lengths))[None]<torch.tensor(lengths)[:,None]
        refs=None
        if need_refs:
            images=list(self.pool.map(self.image,sum(samples['ref_image_paths'],[])))
            refs=torch.zeros(len(lengths),max(lengths),3,96,96);refs[keep]=torch.stack(images)
            refs=refs.pin_memory().to(self.device,non_blocking=True)
        neutral=self.ec.features_many('content',[('',c) for c in samples['char_cp']])
        content=[torch.cat([neutral[('',c)][s] for c in samples['char_cp']]).to(self.device) for s in range(5)]
        query=content[-1].clone();selections=[]
        if not no_delta:
            for f,c,r,q in zip(samples['font_stem'],samples['char_cp'],samples['ref_chars'],queries):
                selections.append(self.library.select(c,r,q,f,self.alpha_cfg))
            donors=self.ec.features_many('target',[(f,c) for (fs,_),c in zip(selections,samples['char_cp']) for f in fs])
        width=max((len(fs) for fs,_ in selections),default=1);alpha=torch.zeros(len(lengths),width)
        for i,(_,a) in enumerate(selections):alpha[i,:len(a)]=a
        if no_delta:alpha.fill_(1)
        alpha=alpha.to(self.device);active=(~(cfg|source)).float() if not no_delta else torch.zeros_like(cfg,dtype=torch.float32)
        structure=[]
        for s,c in enumerate(content):
            if s not in (1,2) or no_delta:d=c.new_zeros(len(lengths),width,*c.shape[1:])
            else:
                rows=[]
                for (fs,_),cp in zip(selections,samples['char_cp']):
                    row=torch.cat([donors[(f,cp)][s] for f in fs])-neutral[('',cp)][s]
                    if len(fs)<width:row=torch.cat([row,torch.zeros(width-len(fs),*row.shape[1:])])
                    rows.append(row)
                d=torch.stack(rows).pin_memory().to(self.device,non_blocking=True)
            structure.append((d,alpha,c,active))
        return style,refs,query,keep.to(self.device),[c.masked_fill(cfg[:,None,None,None],0) for c in content],structure

def data_identity(args):
    m=Path(args.v0913_clean_map)
    return dict(data={p.name:sha256_file(p) for p in sorted(m.glob('*')) if p.is_file()},
                donor_name=args.k4_donor_name,cache={k:sha256_file(STORE/'cache'/k/'COMPLETE.json') for k in ['es','ec']},
                detail_sha256=sha256_file(ASSETS/'detail_manifest.json'),aliases_sha256=sha256_file(ASSETS/'aliases.json'),
                donor_policy='exclude_self_and_explicit_weight_variants_before_alpha',family_groups_sha256=sha256_file(ASSETS/'weight_groups.json'))

def code_identity():
    meta=json.loads((CODE/'K4_CODE_IDENTITY.json').read_text())
    for name,h in meta['files'].items():assert sha256_file(CODE/name)==h,'Source drift: '+name
    return meta
