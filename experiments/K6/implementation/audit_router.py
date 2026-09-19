import torch
import torch.nn.functional as F
def audited_forward(self,hidden,payload):
 d,a,n,active,ref,t=payload;b,m,c,h,w=d.shape
 if hasattr(self,'es_adapter'):d=self.es_adapter(d.flatten(0,1)).reshape(b,m,c,h,w)
 tt=t.float().reshape(b,1)/1000
 q=self.query(hidden)+self.content(n)+(self.ref(ref)+self.time(torch.cat([tt,tt.square()],1)))[:,:,None,None]
 key=self.key(d.flatten(0,1)).reshape(b,m,32,h,w)
 logits=(F.normalize(q.float(),dim=1)[:,None]*F.normalize(key.float(),dim=2)).sum(2)*self.strength.float()+a.float().clamp_min(1e-12).log()[:,:,None,None]
 logits=logits.masked_fill(a[:,:,None,None]<=0,-torch.inf);weights=logits.softmax(1)
 mix=(weights[:,:,None]*d.float()).sum(1)
 out=self.original(hidden,mix.to(hidden.dtype))*active[:,None,None,None]
 self.audit_sink.append(dict(layer=self.audit_name,t=t.detach().flatten().tolist(),strength=float(self.strength),alpha=a.detach().tolist(),router_mean=weights.mean((2,3)).detach().tolist(),router_spatial_std=weights.std((2,3)).detach().tolist(),prior_l1=float((weights-a[:,:,None,None]).abs().mean()),delta_norm=d.float().flatten(2).norm(dim=2).detach().tolist(),mixed_norm=float(mix.norm()),offset_norm=float(out.float().norm())))
 return out
