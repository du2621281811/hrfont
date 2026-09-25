import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18,ResNet18_Weights

class Encoder(nn.Module):
    def __init__(self,multiscale=False,pretrained=True):
        super().__init__();self.multiscale=multiscale
        self.net=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None);self.net.fc=nn.Identity()
        if multiscale:self.proj2=nn.Linear(256,128);self.proj3=nn.Linear(512,128)
    def train(self,mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m,nn.BatchNorm2d):m.eval()
        return self
    def forward(self,x):
        n=self.net;x=n.maxpool(n.relu(n.bn1(n.conv1(x))));x=n.layer1(x);l2=n.layer2(x);l3=n.layer3(l2);l4=n.layer4(l3)
        z=F.normalize(n.avgpool(l4).flatten(1).float(),dim=1)
        if not self.multiscale:return z
        stat=lambda f:torch.cat([f.float().mean((2,3)),f.float().var((2,3),unbiased=False).add(1e-6).sqrt()],1)
        return torch.cat([z*(.5**.5),F.normalize(self.proj2(stat(l2)).float(),dim=1)*.5,F.normalize(self.proj3(stat(l3)).float(),dim=1)*.5],1)

def branches(z):return [z] if z.shape[-1]==512 else [z[...,:512]/(.5**.5),z[...,512:640]*2,z[...,640:]*2]
def cosine(q,refs,keep):
    qs,rs=branches(q),branches(refs);weights=[1.] if len(qs)==1 else [.5,.25,.25]
    return sum(w*(qq*F.normalize(rr.masked_fill(~keep[:,:,None],0).sum(1)/keep.sum(1,keepdim=True),dim=1)).sum(1) for qq,rr,w in zip(qs,rs,weights))

class SetHead(nn.Module):
    def __init__(self,dim=512,residual=False):
        super().__init__();self.residual=residual
        if not residual:self.aggregator=nn.Sequential(nn.Linear(dim*2,256),nn.ReLU(),nn.Linear(256,dim))
        self.fusion=nn.Sequential(nn.Linear(dim*4,256),nn.ReLU(),nn.Linear(256,1))
        if residual:nn.init.zeros_(self.fusion[-1].weight);nn.init.zeros_(self.fusion[-1].bias)
    def forward(self,q,refs,keep):
        assert keep.dtype==torch.bool and keep.any(1).all() and refs.shape[:2]==keep.shape
        mean=refs.masked_fill(~keep[:,:,None],0).sum(1)/keep.sum(1,keepdim=True)
        if self.residual:r=F.normalize(mean,dim=1)
        else:r=F.normalize(self.aggregator(torch.cat([mean,refs.masked_fill(~keep[:,:,None],-torch.inf).amax(1)],1)),dim=1)
        delta=self.fusion(torch.cat([q,r,(q-r).abs(),q*r],1)).flatten();base=cosine(q,refs,keep)
        return (base+.25*torch.tanh(delta) if self.residual else delta),base

def multi_positive(a,b,inst,group,tau=.1):
    def loss(x,y):
        with torch.autocast(device_type=x.device.type,enabled=False):
            logits=x.float()@y.float().T/tau;pos=inst[:,None]==inst[None,:];valid=pos|(group[:,None]!=group[None,:])
            assert (valid&~pos).any(1).all()
            logp=logits-torch.logsumexp(logits.masked_fill(~valid,-torch.inf),1,keepdim=True)
            return -(logp.masked_fill(~pos,0).sum(1)/pos.sum(1)).mean()
    weights=[1.] if a.shape[-1]==512 else [1.,.25,.25]
    return sum(w*.5*(loss(x,y)+loss(y,x)) for x,y,w in zip(branches(a),branches(b),weights))
