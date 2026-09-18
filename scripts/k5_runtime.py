"""K5 spatial Es Delta, independent K0 initialization, original V2 selection."""
import copy,collections,hashlib,json
from pathlib import Path
import torch
from torch import nn
import k4_runtime as K4
from k4_runtime import ROOT,CODE,CACHE,PARENT0,T,dataset,train_sample,batch_to,seed_episode,atomic_json,sha256_file,configure,data_identity
from scripts.hrfont_i import SetOffset
STORE=Path('/root/data1/hrfont_k5_20260919')
ASSETS=CODE/'experiments/K5'

class EsOffset(SetOffset):
    def __init__(self,original):
        super().__init__(original)
        c=original.style_proj_in.in_channels
        self.es_adapter=nn.Conv2d(c,c,1,bias=False)
        nn.init.eye_(self.es_adapter.weight[:,:,0,0])
    def forward(self,hidden,payload):
        d,a,n,active,ref,t=payload;b,m,c,h,w=d.shape
        d=self.es_adapter(d.flatten(0,1)).reshape(b,m,c,h,w)
        return super().forward(hidden,(d,a,n,active,ref,t))

class DualOffset(nn.Module):
    def __init__(self,ec_branch):
        super().__init__();self.ec_branch=ec_branch
        self.es_branch=EsOffset(copy.deepcopy(ec_branch.original))
        self.es_gate=nn.Parameter(torch.zeros(()))
    def forward(self,hidden,payload):
        d,a,n,active,ref,t=payload;c=n.shape[1]
        if d.shape[2]==c:
            assert not bool(active.any()), 'Active dual payload requires both branches'
            ec=es=d
        else:ec,es=d.split(c,dim=2)
        original=self.ec_branch(hidden,(ec,a,n,active,ref,t))
        # A near-zero gate must not underflow Es branch gradients in FP16.
        with torch.autocast(device_type=hidden.device.type,enabled=False):
            extra=self.es_branch(hidden.float(),(es.float(),a.float(),n.float(),active,ref.float(),t))
            combined=original.float()+self.es_gate.float().tanh()*extra.float()
        return combined.to(original.dtype)

def model_for(arm,device,output,evaluation=False):
    assert arm in ['K5-A','K5-B']
    model,args=K4.model_for('K4-C',device,output,evaluation=True)
    for block in model.base.unet.up_blocks:
        if hasattr(block,'sc_interpreter_offsets'):
            block.sc_interpreter_offsets=nn.ModuleList([EsOffset(x.original) if arm=='K5-A' else DualOffset(x) for x in block.sc_interpreter_offsets])
    model.arm=arm;model.to(device);args.k5_arm=arm
    return model,configure(args,'v2','v2')

class EsTargetFeatures:
    def __init__(self,old,encoder,device,arm):
        self.old,self.encoder,self.device,self.arm=old,encoder,device,arm
        self.cache=collections.OrderedDict();self.capacity=256
    @torch.no_grad()
    def features_many(self,role,items):
        if role=='content':return self.old.features_many(role,items)
        assert role=='target';unique=list(dict.fromkeys(items));keys=[('target',f,c) for f,c in unique]+[('content','',c) for c in sorted({c for f,c in unique})]
        missing=[k for k in keys if k not in self.cache];local={k:self.cache[k] for k in keys if k in self.cache}
        self.encoder.eval()
        for start in range(0,len(missing),16):
            kk=missing[start:start+16];paths=[K4.DATA/'v2/train'/('TargetImage' if role=='target' else 'ContentImage')/(f if role=='target' else '')/(f+'+'+cp+'.png' if role=='target' else cp+'.png') for role,f,cp in kk]
            x=torch.stack([K4.read_unit(p)*2-1 for p in paths]).to(self.device)
            vals=[]
            with torch.autocast(device_type=self.device.type,enabled=False):
                for blocks in list(self.encoder.blocks)[:2]:
                    for block in blocks:x=block(x)
                    vals.append(x.detach().cpu())
            assert [tuple(v.shape[1:]) for v in vals]==[(64,48,48),(128,24,24)]
            for i,k in enumerate(kk):local[k]=[v[i:i+1].clone() for v in vals]
        for k in keys:
            self.cache[k]=local[k];self.cache.move_to_end(k)
        while len(self.cache)>self.capacity:self.cache.popitem(last=False)
        neutral=self.old.features_many('content',[('',cp) for _,cp in unique])
        ec=self.old.features_many('target',unique) if self.arm=='K5-B' else None
        # K4.conditions subtracts Ec neutral afterwards; adding it here gives exact
        # Es difference there. B: concatenate Ec target and Es delta + Ec neutral.
        result={}
        for f,cp in unique:
            row=[None]*5
            for s in [1,2]:
                es=local[('target',f,cp)][s-1]-local[('content','',cp)][s-1]
                row[s]=es+neutral[('',cp)][s] if ec is None else torch.cat([ec[(f,cp)][s],es+neutral[('',cp)][s]],dim=1)
            result[(f,cp)]=row
        return result

class DataContext(K4.DataContext):
    def __init__(self,args,device,model):
        super().__init__(args,device);self.arm=args.k5_arm
        self.ec=EsTargetFeatures(self.ec,model.base.style_encoder,device,self.arm)
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
                    row=torch.cat([donors[(f,cp)][s] for f in fs])-(torch.cat([neutral[('',cp)][s]]*2,dim=1) if self.arm=='K5-B' else neutral[('',cp)][s])
                    if len(fs)<width:row=torch.cat([row,torch.zeros(width-len(fs),*row.shape[1:])])
                    rows.append(row)
                d=torch.stack(rows).pin_memory().to(self.device,non_blocking=True)
            structure.append((d,alpha,c,active))
        return style,refs,query,keep.to(self.device),[c.masked_fill(cfg[:,None,None,None],0) for c in content],structure

def code_identity():
    meta=json.loads((CODE/'K5_CODE_IDENTITY.json').read_text())
    for path,h in meta['files'].items():assert sha256_file(CODE/path)==h,'K5 source drift: '+path
    return meta
