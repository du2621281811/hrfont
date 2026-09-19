"""Inference-only exact donor softmax reduction with bounded key/mix temporaries."""
import torch
import torch.nn.functional as F

def chunk_forward(self,hidden,payload,chunk=16):
 d,a,n,active,ref,t=payload;b,m,c,h,w=d.shape
 tt=t.float().reshape(b,1)/1000
 q=self.query(hidden)+self.content(n)+(self.ref(ref)+self.time(torch.cat([tt,tt.square()],1)))[:,:,None,None]
 # Es adapter is pointwise and can be applied per donor chunk identically.
 def part(i):
  x=d[:,i:i+chunk];k=x.shape[1]
  if hasattr(self,'es_adapter'):x=self.es_adapter(x.flatten(0,1)).reshape(b,k,c,h,w)
  keys=self.key(x.flatten(0,1)).reshape(b,k,32,h,w)
  with torch.autocast(device_type=hidden.device.type,enabled=False):
   logits=(F.normalize(q.float(),dim=1)[:,None]*F.normalize(keys.float(),dim=2)).sum(2)*self.strength.float()+a[:,i:i+k].float().clamp_min(1e-12).log()[:,:,None,None]
   logits=logits.masked_fill(a[:,i:i+k,None,None]<=0,-torch.inf)
  return x,logits
 maximum=None;den=None;num=None
 for i in range(0,m,chunk):
  x,l=part(i)
  with torch.autocast(device_type=hidden.device.type,enabled=False):
   mx=l.max(1).values
   if maximum is None:maximum=mx;den=torch.zeros_like(mx);num=torch.zeros(b,c,h,w,device=hidden.device)
   new=torch.maximum(maximum,mx);safe=torch.where(torch.isfinite(new),new,torch.zeros_like(new));scale=(maximum-safe).exp();wgt=(l-safe[:,None]).exp()
   num=num*scale[:,None]+(wgt[:,:,None]*x.float()).sum(1);den=den*scale+wgt.sum(1);maximum=new
 return self.original(hidden,(num/den[:,None]).to(hidden.dtype))*active[:,None,None,None]
