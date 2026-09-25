"""Independent E12-c; real-font relation labels, no generator components."""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.net=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.net.fc=nn.Identity()

    def train(self, mode=True):
        super().train(mode)
        # Freeze only running statistics during A; affine parameters still train.
        for m in self.modules():
            if isinstance(m,nn.BatchNorm2d): m.eval()
        return self

    def forward(self,x): return F.normalize(self.net(x).float(),dim=1)


class SetHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregator=nn.Sequential(nn.Linear(1024,256),nn.ReLU(),nn.Linear(256,512))
        self.fusion=nn.Sequential(nn.Linear(2048,256),nn.ReLU(),nn.Linear(256,1))

    def forward(self,q,refs,keep):
        if refs.ndim!=3 or keep.dtype!=torch.bool or refs.shape[:2]!=keep.shape or not keep.any(1).all():
            raise ValueError('Invalid reference mask')
        mean=refs.masked_fill(~keep[:,:,None],0).sum(1)/keep.sum(1,keepdim=True)
        maximum=refs.masked_fill(~keep[:,:,None],-torch.inf).amax(1)
        r=F.normalize(self.aggregator(torch.cat([mean,maximum],1)),dim=1)
        score=self.fusion(torch.cat([q,r,(q-r).abs(),q*r],1)).flatten()
        return score,(q*F.normalize(mean,dim=1)).sum(1)


def multi_positive(cn,target,instances,families,tau=.1):
    with torch.autocast(device_type=cn.device.type,enabled=False):
        logits=cn.float()@target.float().T/tau
        positive=instances[:,None]==instances[None,:]
        candidate=positive | (families[:,None]!=families[None,:])
        if not positive.any(1).all() or not (candidate & ~positive).any(1).all():
            raise ValueError('Each anchor needs positives and legal negatives')
        def direction(x,p,mask):
            logprob=x-torch.logsumexp(x.masked_fill(~mask,-torch.inf),1,keepdim=True)
            return -(logprob.masked_fill(~p,0).sum(1)/p.sum(1)).mean()
        return .5*(direction(logits,positive,candidate)+direction(logits.T,positive.T,candidate.T))
