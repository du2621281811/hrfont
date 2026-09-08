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
    order=torch.argsort(scores); _,counts=torch.unique_consecutive(scores[order],return_counts=True)
    ends=counts.cumsum(0); starts=ends-counts
    average_ranks=(starts+1+ends).float()/2
    ranks=torch.empty_like(scores); ranks[order]=torch.repeat_interleave(average_ranks,counts)
    return float((ranks[labels==1].sum()-pos*(pos+1)/2)/(pos*neg))

def pr_auc(labels, scores):
    y=torch.as_tensor(labels).float(); s=torch.as_tensor(scores); order=torch.argsort(s,descending=True); y=y[order]
    # Non-interpolated average precision: tied scores enter at one threshold.
    _,counts=torch.unique_consecutive(s[order],return_counts=True)
    ends=counts.cumsum(0); tp=y.cumsum(0)[ends-1]
    positive_increments=torch.diff(torch.cat((tp.new_zeros(1),tp)))
    return float(((tp/ends)*positive_increments).sum()/max(1.,float(y.sum())))

def ece(labels, probs, bins=10):
    y=torch.as_tensor(labels).float(); p=torch.as_tensor(probs).float(); total=0.
    for i in range(bins):
        mask=(p>=i/bins)&(p<=(i+1)/bins if i==bins-1 else p<(i+1)/bins)
        if mask.any(): total += float(mask.float().mean()*torch.abs(p[mask].mean()-y[mask].mean()))
    return total

def infinite(loader):
    while True:
        for batch in loader: yield batch
