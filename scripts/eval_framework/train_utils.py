"""训练脚本共享逻辑。"""
from __future__ import annotations
import math, torch

def scheduler(optimizer, total_steps, warmup_steps):
    def factor(step):
        if warmup_steps and step < warmup_steps: return float(step+1)/warmup_steps
        progress=(step-warmup_steps)/max(1,total_steps-warmup_steps)
        return .5*(1+math.cos(math.pi*min(1.,max(0.,progress))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,factor)

def amp_context(device, enabled):
    return torch.autocast(device_type=device.type,dtype=torch.float16,enabled=enabled and device.type=="cuda")

def binary_auc(labels, scores):
    labels=torch.as_tensor(labels).long(); scores=torch.as_tensor(scores).float(); pos=(labels==1).sum().item(); neg=(labels==0).sum().item()
    if not pos or not neg: return float("nan")
    order=torch.argsort(scores); ranks=torch.empty_like(order,dtype=torch.float); ranks[order]=torch.arange(1,len(scores)+1,dtype=torch.float)
    return float((ranks[labels==1].sum()-pos*(pos+1)/2)/(pos*neg))

def pr_auc(labels, scores):
    y=torch.as_tensor(labels).float(); s=torch.as_tensor(scores); order=torch.argsort(s,descending=True); y=y[order]
    precision=y.cumsum(0)/torch.arange(1,len(y)+1); return float((precision*y).sum()/max(1.,float(y.sum())))

def ece(labels, probs, bins=10):
    y=torch.as_tensor(labels).float(); p=torch.as_tensor(probs).float(); total=0.
    for i in range(bins):
        mask=(p>=i/bins)&(p<=(i+1)/bins if i==bins-1 else p<(i+1)/bins)
        if mask.any(): total += float(mask.float().mean()*torch.abs(p[mask].mean()-y[mask].mean()))
    return total

def infinite(loader):
    while True:
        for batch in loader: yield batch
