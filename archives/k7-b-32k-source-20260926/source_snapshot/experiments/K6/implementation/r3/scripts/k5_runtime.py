"""K5 spatial Es Delta, independent K0 initialization, original V2 selection."""
import copy,collections,hashlib,json,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
import k4_runtime as K4
from k4_runtime import ROOT,CODE,CACHE,PARENT0,T,dataset,train_sample,batch_to,seed_episode,atomic_json,sha256_file,configure,data_identity
from scripts.hrfont_i import SetOffset
from scripts.hrfont_feature_cache import key_es,key_ec,ES_SPATIAL
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
    def __init__(self,old,encoder,device,arm,v0921=None):
        self.old,self.encoder,self.device,self.arm=old,encoder,device,arm
        self.v0921_data_root=getattr(v0921,'data_root',None)
        self.v0921_fonts=set(getattr(v0921,'fonts',()))
        self.cache=collections.OrderedDict();self.capacity=256
    @torch.no_grad()
    def features_many(self,role,items):
        if role=='content':return self.old.features_many(role,items)
        assert role=='target';unique=list(dict.fromkeys(items));keys=[('target',f,c) for f,c in unique]+[('content','',c) for c in sorted({c for f,c in unique})]
        missing=[k for k in keys if k not in self.cache];local={k:self.cache[k] for k in keys if k in self.cache}
        self.encoder.eval()
        for start in range(0,len(missing),16):
            kk=missing[start:start+16];paths=[]
            for item_role,f,cp in kk:
                root=(self.v0921_data_root if item_role=='target' and f in self.v0921_fonts else K4.DATA/'v2')
                paths.append(root/'train'/('TargetImage' if item_role=='target' else 'ContentImage')/(f if item_role=='target' else '')/(f+'+'+cp+'.png' if item_role=='target' else cp+'.png'))
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

class K7DynamicEs:
    """K7-only V2 Es fallback encoded by the authorized independent K0 encoder."""
    def __init__(self,extra,encoder,device,keys,eval_root=None):
        self.old=K4.EsCache(K4.ROOT/'artifacts/g0/es_spatial');self.extra=extra
        self.encoder,self.device=encoder,device
        self.eval_root=Path(eval_root) if eval_root else K4.DATA/'v2'
        missing=[]
        for key in dict.fromkeys(keys):
            split,font,cp=key.split('|')[1:4]
            if key not in self.old.table.index and (extra is None or key not in extra.es.table.index):missing.append((split,font,cp))
        self.dynamic_index={key_es(s,f,c):i for i,(s,f,c) in enumerate(missing)}
        self.spatial=np.empty((len(missing),*ES_SPATIAL),dtype=np.float16)
        self.pooled=np.empty((len(missing),1024),dtype=np.float16)
        # The old implementation made every DDP rank encode the complete
        # missing set.  Shard the one-time precompute across ranks and persist
        # the result so later resumes/A2 runs only read the immutable cache.
        world=dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank=dist.get_rank() if world > 1 else 0
        digest=hashlib.sha256('\n'.join(f'{s}|{f}|{c}' for s,f,c in missing).encode()).hexdigest()[:20]
        cache_dir=Path('/root/data1/k7_b_dynamic_es_cache')/digest
        complete=cache_dir/'COMPLETE.json'
        def barrier():
            if world > 1: dist.barrier()
        def load_shards():
            for r in range(world):
                sarr=np.load(cache_dir/f'spatial_rank{r}.npy',mmap_mode='r')
                parr=np.load(cache_dir/f'pooled_rank{r}.npy',mmap_mode='r')
                positions=list(range(r,len(missing),world))
                self.spatial[positions]=sarr
                self.pooled[positions]=parr
        if not complete.exists():
            cache_dir.mkdir(parents=True,exist_ok=True)
            owned=list(range(rank,len(missing),world))
            self.encoder.eval().requires_grad_(False);pool=ThreadPoolExecutor(max_workers=8)
            local_spatial=np.empty((len(owned),*ES_SPATIAL),dtype=np.float16)
            local_pooled=np.empty((len(owned),1024),dtype=np.float16)
            for st in range(0,len(owned),64):
                positions=owned[st:st+64]
                batch=[missing[i] for i in positions]
                paths=[
                    (K4.DATA/'v2' if s == 'train' else self.eval_root)
                    / s / 'StyleImage' / f / f'{f}+{c}.png'
                    for s,f,c in batch
                ]
                x=torch.stack(list(pool.map(lambda p:K4.read_unit(p)*2-1,paths))).to(device)
                with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):spatial,pooled,_=encoder(x)
                local_spatial[st:st+len(batch)]=spatial.detach().half().cpu().numpy()
                local_pooled[st:st+len(batch)]=torch.nn.functional.normalize(pooled.float(),dim=1).half().cpu().numpy()
            pool.shutdown()
            spatial_tmp=cache_dir/f'spatial_rank{rank}.npy.tmp.{os.getpid()}'
            pooled_tmp=cache_dir/f'pooled_rank{rank}.npy.tmp.{os.getpid()}'
            with open(spatial_tmp,'wb') as fh: np.save(fh,local_spatial)
            with open(pooled_tmp,'wb') as fh: np.save(fh,local_pooled)
            os.replace(spatial_tmp,cache_dir/f'spatial_rank{rank}.npy')
            os.replace(pooled_tmp,cache_dir/f'pooled_rank{rank}.npy')
            barrier()
            if rank==0:
                atomic_json(complete,{'key_count':len(missing),'world_size':world,'digest':digest})
            barrier()
        else:
            barrier()
        load_shards()
        self.dynamic_encoder_sha256=K4.sha256_file(K4.PARENT0/'style_encoder.pth')
        self.dynamic_entries=len(missing)
    def tensor(self,name,split,font,cp):
        key=key_es(split,font,cp)
        if self.extra is not None and key in self.extra.es.table.index:
            return self.extra.es.spatial_tensor(split,font,cp) if name=='spatial' else self.extra.es.pooled_tensor(split,font,cp)
        if key in self.old.table.index:
            row=self.old.table.index[key];value=self.old.spatial[row] if name=='spatial' else self.old.pooled[row]
            return torch.from_numpy(np.array(value,copy=True)).float()
        if key not in self.dynamic_index:raise KeyError(key)
        arr=self.spatial if name=='spatial' else self.pooled
        return torch.from_numpy(np.array(arr[self.dynamic_index[key]],copy=True)).float()
    def spatial_tensor(self,*key):return self.tensor('spatial',*key)
    def pooled_tensor(self,*key):return self.tensor('pooled',*key)

class K7DynamicEc:
    """Use immutable base rows and compute missing V2 donor rows online, bounded in RAM."""
    def __init__(self,extra,encoder,device,capacity=1024):
        self.old=K4.EcCache(K4.ROOT/'artifacts/g0/ec_multiscale');self.extra=extra
        self.encoder,self.device=encoder,device;self.cache=collections.OrderedDict();self.capacity=capacity
        self.dynamic_encoder_sha256=K4.sha256_file(K4.PARENT0/'content_encoder.pth')
    @torch.no_grad()
    def _encode(self,items):
        requested=list(dict.fromkeys(items))
        # `features_many`'s misses are relative to the immutable disk cache,
        # so some requested rows may already be present in this dynamic LRU.
        # Keep direct references to those hits before encoding can evict them.
        computed={item:self.cache[item] for item in requested if item in self.cache}
        missing=[item for item in requested if item not in self.cache]
        if not missing:return computed
        self.encoder.eval().requires_grad_(False);pool=ThreadPoolExecutor(max_workers=8)
        for st in range(0,len(missing),32):
            batch=missing[st:st+32]
            paths=[K4.DATA/'v2/train/TargetImage'/f/f'{f}+{c}.png' for f,c in batch]
            x=torch.stack(list(pool.map(lambda p:K4.read_unit(p)*2-1,paths))).to(self.device)
            with torch.autocast(device_type=self.device.type,enabled=False):_,residuals=self.encoder(x)
            for i,item in enumerate(batch):
                features=[None,residuals[1][i:i+1].detach().half().cpu().float(),
                    residuals[2][i:i+1].detach().half().cpu().float(),None,None]
                computed[item]=features
                self.cache[item]=features
                self.cache.move_to_end(item)
        pool.shutdown()
        while len(self.cache)>self.capacity:self.cache.popitem(last=False)
        # Return every requested row independently of the bounded LRU. A batch
        # can contain existing LRU hits and more misses than its capacity;
        # evicted rows must remain available to the current caller.
        return computed
    def features_many(self,role,items):
        unique=list(dict.fromkeys(items))
        if role=='content':return self.old.features_many(role,unique)
        if role!='target':raise AssertionError(role)
        result={};missing=[]
        for item in unique:
            key=key_ec(role,*item)
            if self.extra is not None and key in self.extra.ec.table.index:
                result[item]=self.extra.ec.features_many(role,[item])[item]
            elif key in self.old.table.index:result[item]=self.old.features(role,*item)
            else:missing.append(item)
        if missing:
            result.update(self._encode(missing))
        return result

class DataContext(K4.DataContext):
    def __init__(self,args,device,model):
        super().__init__(args,device);self.arm=args.k5_arm
        if getattr(args,'k7_v2_only',False):
            dm=json.loads((K4.DATA/'manifests/v2/donor_train_by_cp.json').read_text())
            pools=json.loads((K4.DATA/'manifests/v2/style_pool.json').read_text())
            train_fonts=json.loads((K4.DATA/'manifests/v2/split.json').read_text())['stems']['train']
            needed=set(train_fonts)|{f for fs in dm.values() for f in fs}
            keys=[key_es('train',f,c) for f in sorted(needed) for c in sorted(pools[f])]
            extra=self.es.extra
            eval_refs=getattr(args,'k7_eval_refs',None)
            eval_refs_by_font=getattr(args,'k7_eval_refs_by_font',{}) or {}
            eval_split=getattr(args,'k7_eval_split','val')
            if eval_refs:
                eval_stems=json.loads(Path(args.split_manifest).read_text())['stems'][eval_split]
                keys.extend(
                    key_es(eval_split,f,c)
                    for f in eval_stems
                    for c in eval_refs_by_font.get(f,eval_refs)
                )
            self.es=K7DynamicEs(
                extra,
                model.base.style_encoder,
                device,
                keys,
                getattr(args,'data_root',None),
            )
            self.ec=K7DynamicEc(extra,model.base.content_encoder,device)
            self.library=K4.Library(self.es,dm,pools);self.fonts=self.library.fonts
            if self.v0921 is not None:
                # This validation hashes tens of thousands of files.  Running
                # it independently on all DDP ranks saturated the NFS and
                # looked like a deadlock before step 1.  Validate once, keep
                # a receipt for later checkpoint segments, then synchronize.
                receipt_dir=Path('/root/data1/k7_b_v0921_validation')
                receipt=receipt_dir/f'{self.v0921.payload_sha256}.json'
                failed=receipt.with_suffix('.FAILED.json')
                if not receipt.exists() and dist.get_rank()==0:
                    try:
                        self.v0921.validate_runtime(
                            set(self.library.fonts),self.es.old.manifest,
                            self.ec.old.manifest,Path(args.data_root),
                        )
                        atomic_json(receipt,dict(spec_sha256=self.v0921.payload_sha256,
                            status='PASS'))
                    except Exception as exc:
                        atomic_json(failed,dict(spec_sha256=self.v0921.payload_sha256,
                            status='FAIL',error=repr(exc)))
                dist.barrier()
                if failed.exists():
                    raise RuntimeError(f'v0921 runtime validation failed: {failed.read_text()}')
                if not receipt.exists():
                    raise RuntimeError(f'v0921 runtime validation receipt missing: {receipt}')
                self.library.family_policy=K4.NonBankFamilyPolicy(self.library.family_policy,set(self.v0921.fonts))
            assert set(pools)==set(self.library.family_policy.groups)
            assert set(json.loads((K4.ASSETS/'aliases.json').read_text()))==set(pools)
        self.ec=EsTargetFeatures(self.ec,model.base.style_encoder,device,self.arm,self.v0921)
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
        content_rows=[self.content_features(f,c) for f,c in zip(samples['font_stem'],samples['char_cp'])]
        content=[torch.cat([row[s] for row in content_rows]).to(self.device) for s in range(5)]
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
                for i,((fs,_),cp) in enumerate(zip(selections,samples['char_cp'])):
                    row=torch.cat([donors[(f,cp)][s] for f in fs])-(torch.cat([content_rows[i][s]]*2,dim=1) if self.arm=='K5-B' else content_rows[i][s])
                    if len(fs)<width:row=torch.cat([row,torch.zeros(width-len(fs),*row.shape[1:])])
                    rows.append(row)
                d=torch.stack(rows).pin_memory().to(self.device,non_blocking=True)
            structure.append((d,alpha,c,active))
        return style,refs,query,keep.to(self.device),[c.masked_fill(cfg[:,None,None,None],0) for c in content],structure

def code_identity():
    meta=json.loads((CODE/'K5_CODE_IDENTITY.json').read_text())
    for path,h in meta['files'].items():assert sha256_file(CODE/path)==h,'K5 source drift: '+path
    return meta
