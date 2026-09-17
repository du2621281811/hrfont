"""Same-content, same-pure-noise reference interventions with full rollout gradients."""
import collections, contextlib, random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from scripts.i34_components import gray, sobel
from scripts.i56_components import region_detail_distance

PAIR_VERSION='K3-pure-noise-ddim8-gt-response-v1'

@contextlib.contextmanager
def isolated_rng(seed):
    py=random.getstate();npstate=np.random.get_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed)
        try:yield
        finally:random.setstate(py);np.random.set_state(npstate)

class PairSampler:
    def __init__(self,ds,sampler,groups):
        self.ds,self.base,self.groups=ds,sampler,groups
        self.by=collections.defaultdict(list)
        for i,(font,cp,script,flag) in enumerate(sampler.entries):self.by[script,cp,flag].append(i)
        self.cps={g:sorted(c for gg,c,flag in self.by if gg==g and flag and (g,c,False) in self.by) for g in ('western','kana','bopomofo')}

    def draw(self,event,rank,device,batch_to):
        # Every event contains 4 western, 3 kana, 1 bopomofo pair; rotate rank assignment.
        group=['western']*4+['kana']*3+['bopomofo']
        script=group[(rank+event)%8];rng=random.Random(93407+event*1009+rank)
        for retry in range(100):
            cp=rng.choice(self.cps[script]);ia=rng.choice(self.by[script,cp,True]);ib=rng.choice(self.by[script,cp,False])
            fa,fb=self.base.entries[ia][0],self.base.entries[ib][0]
            if fa==fb or self.groups.get(fa,fa)==self.groups.get(fb,fb):continue
            sa,sb=self.ds[ia],self.ds[ib]
            ga,gb=sa['nonorm_target_image'],sb['nonorm_target_image']
            if int(((gray(ga[None])-gray(gb[None])).abs()>.10).sum())<16:continue
            common=sorted((set(self.ds.style_by_font_char[fa])&set(self.ds.style_by_font_char[fb]))-{cp})
            k=rng.choice((1,2,4,8));assert len(common)>=8
            refs=rng.sample(common,k)
            for s,f in [(sa,fa),(sb,fb)]:
                assert s['split']=='train' and s['char_cp']==cp
                s['ref_chars']=refs;s['ref_image_paths']=[str(self.ds.style_by_font_char[f][c]) for c in refs]
            assert torch.equal(sa['content_image'],sb['content_image'])
            return batch_to([sa,sb],device),dict(fonts=[fa,fb],cp=cp,script=script,shot=k,refs=refs,retries=retry)
        raise RuntimeError('No nonduplicate GT style pair: '+script)

def pair_loss(pred,gt):
    p,y=gray(pred),gray(gt.detach())
    inkdiff=(y[0:1]-y[1:2]).abs()>.10
    edgediff=(sobel(y[0:1])-sobel(y[1:2])).abs().amax(1,keepdim=True)>.02
    mask=F.max_pool2d((inkdiff|edgediff).float(),3,1,1)
    residual=((p[0:1]-p[1:2])-(y[0:1]-y[1:2])).abs()
    difference=(residual*mask).sum()/mask.sum().clamp_min(1)
    reconstruction=region_detail_distance(pred,gt).sum()
    return reconstruction+difference,dict(reconstruction=reconstruction,difference=difference,
        mask_pixels=mask.sum(),prediction_difference=(p[0]-p[1]).abs().mean(),gt_difference=(y[0]-y[1]).abs().mean(),
        out_of_range=((pred<0)|(pred>1)).float().mean())

def rollout(model,data,batch,scheduler,gate,seed,trace=False):
    device=data.device;off=torch.zeros(2,device=device,dtype=torch.bool)
    style,refs,query,keep,content,structure=data.conditions(batch,off,off,need_refs=True)
    with torch.autocast('cuda',dtype=torch.float16):context=model.conditions(style,refs,query,keep,off)
    old,local,active,_=context
    # Reapply the complete context on every recomputation, avoiding mutable-attention staleness.
    def predict(x,t,old_tokens,local_tokens):
        with torch.autocast('cuda',dtype=torch.float16):
            pred,_=model.denoise(x,t,style,content,structure,(old_tokens,local_tokens,active,None),gate)
        return pred.float()
    z=torch.randn((1,3,96,96),device=device,generator=torch.Generator(device=device).manual_seed(seed))
    x=z.expand(2,-1,-1,-1).clone();alphas=scheduler.alphas_cumprod.to(device).float()
    times=torch.linspace(999,0,8,device=device).round().long();path=[]
    for i,t in enumerate(times):
        ts=t.expand(2)
        eps=checkpoint(predict,x,ts,old,local,use_reentrant=False,preserve_rng_state=True)
        if trace:eps.retain_grad();path.append(eps)
        a=alphas[t];ap=alphas[times[i+1]] if i+1<len(times) else x.new_tensor(1.)
        clean=(x-(1-a).sqrt()*eps)/a.sqrt()
        # Deterministic DDIM eta=0. No clipping/detaching the rollout gradient.
        x=ap.sqrt()*clean+(1-ap).sqrt()*eps
    return x/2+.5,path

def average_gradients(params):
    """One explicit average after main and multi-forward rollout accumulation.

    Avoids DDP reducer reentrancy for checkpointed, multi-step auxiliary graphs.
    Missing local gradients contribute zero; globally unused parameters stay None.
    """
    device=params[0].device;world=dist.get_world_size()
    used=torch.tensor([p.grad is not None for p in params],device=device,dtype=torch.int32)
    dist.all_reduce(used,op=dist.ReduceOp.SUM);used=used.tolist()
    bucket=[];size=0
    def flush(ps):
        flat=torch.cat([p.grad.reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1) for p in ps])
        dist.all_reduce(flat);flat.div_(world);offset=0
        for p in ps:
            part=flat[offset:offset+p.numel()].view_as(p)
            if p.grad is None:p.grad=part.clone()
            else:p.grad.copy_(part)
            offset+=p.numel()
    for p,n in zip(params,used):
        if not n:continue
        if bucket and size+p.numel()>8_000_000:flush(bucket);bucket=[];size=0
        bucket.append(p);size+=p.numel()
    if bucket:flush(bucket)
